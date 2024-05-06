"""
Fine-tune the CLIP model

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al. and Guillermo Ortiz-Jimenez et al.,
at https://github.com/mlfoundations/task_vectors and
https://github.com/gortizji/tangent_task_arithmetic
"""

import os
import time

import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.utils import LabelSmoothing, cosine_lr


def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    # Check if checkpoints already exist
    ft_path = (
        os.path.join(args.save, train_dataset, "linear_finetuned.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "finetuned.pt")
    )
    lora_path = (
        os.path.join(args.save, train_dataset, "linear_lora.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "lora.pt")
    )
    zs_path = (
        os.path.join(args.save, train_dataset, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "zeroshot.pt")
    )
    
    if os.path.exists(zs_path) and os.path.exists(ft_path) and not args.lora:
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path
    
    if os.path.exists(zs_path) and os.path.exists(lora_path) and args.lora:
        print(f"Skipping fine-tuning because {lora_path} exists.")
        return zs_path, lora_path

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt"):
        image_encoder = (
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    # Build the classification head with all classes, when the dataset only has one.
    if '_' in train_dataset:
        train_dataset_ = train_dataset.split('_')[-1]
    else:
        train_dataset_ = train_dataset
    classification_head = get_classification_head(args, train_dataset_)

    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 50

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    if args.lora and is_main_process():
        import minlora
        from functools import partial
        default_lora_config = {  # specify which layers to add lora to, by default only add to linear layers
            torch.nn.Linear: {
                "weight": partial(minlora.LoRAParametrization.from_linear, rank=4),
            },
        }

        minlora.add_lora(model.image_encoder, lora_config=default_lora_config)
        params = list(minlora.get_lora_params(model.image_encoder))
        print(f"Training {sum(p.numel() for p in params if p.requires_grad)} LoRA params ({sum(p.numel() for p in params if p.requires_grad)/sum(p.numel() for p in model.parameters() if p.requires_grad)*100.:.2f}%)")
        
    model = model.cuda()
    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )
    
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    if not args.lora:
        params = [p for p in ddp_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    # Saving zero-shot model
    if args.save is not None and is_main_process():
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        if not args.lora:
            ddp_model.module.image_encoder.save(model_path)        
        
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):        
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time
            with torch.autocast(device_type="cuda"):
                logits = ddp_model(inputs)
                
                loss = loss_fn(logits, labels)

            #loss.backward()
            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                #optimizer.step()
                optimizer.zero_grad()              

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = (
                    os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
                    if linearized_finetuning
                    else os.path.join(ckpdir, f"checkpoint_{step}.pt")
                )
                ddp_model.module.image_encoder.save(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )

        # Test the model each epoch 
        image_encoder = ddp_model.module.image_encoder
        if not args.lora:
            #I have not found a way to move the lora params back into training mode
            eval_single_dataset(image_encoder, train_dataset, dataset, args)
            
    if args.save is not None and is_main_process():
        zs_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        if args.lora:
            lora_state_dict = minlora.get_lora_state_dict(ddp_model)
            torch.save(lora_state_dict, os.path.join(ckpdir, "lora.pt"))
        else:
            ft_path = (
                os.path.join(ckpdir, "linear_finetuned.pt")
                if linearized_finetuning
                else os.path.join(ckpdir, "finetuned.pt")
            )
            image_encoder.save(ft_path)\
                
    from lightning.fabric import Fabric
    import gc;gc.collect();torch.cuda.empty_cache()
    fabric = Fabric(accelerator="cuda", devices=1, strategy="ddp", precision="32")
    fabric.launch()
    args.fabric = fabric

    eval_single_dataset(image_encoder, train_dataset, dataset, args, test=True)        
    cleanup_ddp()
    return zs_path, ft_path


if __name__ == "__main__":
    
    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 13,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
        "CIFAR10": 5,
        "CIFAR100": 6,
        "ImageNet": 10,
        "STL10": 4,
        "Food101": 15,
        "Caltech256": 8,
        "FGVCAircraft": 60,
        "Flowers102": 40,
        "OxfordIIITPet": 5,
        "CUB200": 20,
        "PascalVOC": 10,
        "Country211": 15,
        "UCF101": 20,
        "Caltech101":10,
    }

    args = parse_arguments()
    if args.datasets is not None:
        epochs =  {k:v for k,v in epochs.items() if k in args.datasets}
    
    for dataset in epochs:

        # HACK: Some command line arguments are overwritten by defaults here.
        args.lr = 1e-5
        args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.batch_size = 32 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 4 if args.model == "ViT-L-14" else 1

        if args.seed is not None and False:
            args.save = f"checkpoints_{args.seed}/{args.model}"
        else:
            args.save = f"checkpoints/{args.model}"
        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
