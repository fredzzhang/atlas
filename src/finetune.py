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
from src.utils import LabelSmoothing, cosine_lr, get_n_shots, apply_lora
from lightning.fabric import Fabric, seed_everything

def finetune(args):

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
        os.path.join(args.save, train_dataset, "linear_lora_{args.rank}.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, f"lora_{args.rank}{'_attn_only' if args.attn_only else ''}{'_mlp_only' if args.mlp_only else ''}.pt")
    )
    zs_path = (
        os.path.join(args.save, train_dataset, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "zeroshot.pt")
    )
    
    if os.path.exists(zs_path) and os.path.exists(ft_path) and not args.lora and args.semi is None:
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path
    
    if os.path.exists(zs_path) and os.path.exists(lora_path) and args.lora and args.semi is None:
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

    if args.lora and is_main_process():
        ori_params = sum(p.numel() for p in model.parameters() if p.requires_grad)        
        model.image_encoder.model, params = apply_lora(model.image_encoder.model, rank=args.rank, mlp=not args.attn_only, attn=not args.mlp_only)
        print(f"Training {sum(p.numel() for p in params if p.requires_grad)} LoRA params ({sum(p.numel() for p in params if p.requires_grad)/ori_params*100.:.2f}%)")
        
    model = model.cuda()
    # Distribute the data and model across the GPUs.   
       
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    if not args.lora:
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

        
    import gc;gc.collect();torch.cuda.empty_cache()
    fabric = Fabric(accelerator="cuda", devices=1, strategy="ddp", precision="16-mixed")
    fabric.launch()
    args.fabric = fabric

    ddp_model, optimizer = fabric.setup(model, optimizer)

    # Saving zero-shot model
    if args.save is not None and is_main_process() and args.semi is None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        if not args.lora:
            ddp_model.image_encoder.save(model_path)

    if args.semi is not None and is_main_process():
        to_keep, preds = get_n_shots(dataset.train_dataset, args.semi, classification_head.out_features, args)

        print(f"Got {len(to_keep)} trusted samples")
        r = len(to_keep) / args.batch_size
        if r < 10:
            over_sampling = 10/r
            over_sampling = int(over_sampling) + 1
            print(f"Oversampling {over_sampling} times")
            to_keep = torch.cat([to_keep] * over_sampling)
            
        sampler = torch.utils.data.SubsetRandomSampler(to_keep)
        data_loader = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    else:
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
        
    data_loader = fabric.setup_dataloaders(data_loader, use_distributed_sampler=False)

    num_batches = len(data_loader)
    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    for epoch in range(args.epochs):        
        ddp_model.train()

        for i, batch in enumerate(data_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            logits = ddp_model(inputs)            
            loss = loss_fn(logits, labels)
            
            fabric.backward(loss)

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
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
                ddp_model.image_encoder.save(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )

        # Test the model each epoch 
        image_encoder = ddp_model.image_encoder
        if False:
            eval_single_dataset(image_encoder, train_dataset, dataset, args)
            
    if args.save is not None and is_main_process() and args.semi is None:
        zs_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        if args.lora:
            import loratorch
            lora_state_dict = loratorch.lora_state_dict(ddp_model)
            torch.save(lora_state_dict, lora_path)
        else:
            ft_path = (
                os.path.join(ckpdir, "linear_finetuned.pt")
                if linearized_finetuning
                else os.path.join(ckpdir, "finetuned.pt")
            )
            image_encoder.save(ft_path)
            
    metrics = eval_single_dataset(image_encoder, train_dataset, dataset, args, test=True)
    if args.semi is not None:
        with open(os.path.join(args.fname, str(args.seed), "results.txt"), 'a') as f:
            f.writelines([f"{train_dataset.replace('Val','')}\n", f"Base accuracy 0.0, best accuracy {metrics['top1']}\n"])
    
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
        
    if args.semi is not None and args.fname is None:
        args.fname = os.path.join("results", args.model, f"{args.semi}_shots")
        print("--fname not found, defaulting to {args.fname}")
        
    if args.fname is not None:
        if os.path.isfile(os.path.join(args.fname, str(args.seed), "results.txt")):
            os.remove(os.path.join(args.fname, str(args.seed), "results.txt"))

        splt = os.path.join(args.fname, str(args.seed), "results.txt").split('/')
        for i in range(len(splt)-1):
            if not os.path.exists('/'.join(splt[:i+1])):
                os.mkdir('/'.join(splt[:i+1]))
        

    for dataset in epochs:

        # HACK: Some command line arguments are overwritten by defaults here.
        if args.lora:
            args.lr = 1e-3
        else:
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
        
        seed = int(torch.exp(torch.tensor(args.seed)) * 3.1415 * 1000)
        seed_everything(seed)

        finetune(args)
        
