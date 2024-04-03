"""Learn coefficients on a task vector for task negation,
with supervised objective on a combination of the target
dataset and the control dataset.

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""
import os
import time
import json
import torch

from torch.cuda.amp import GradScaler
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder, MultiHeadImageClassifier
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.composition import WeightedImageEncoder, WeightedLinearizedModel

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp

def main(rank, args):

    setup_ddp(rank, args.world_size, port=args.port)

    tgt_dataset = args.tgt_dataset
    ctr_dataset = args.ctr_dataset

    ckpdir = os.path.join(args.save, tgt_dataset)
    os.makedirs(ckpdir, exist_ok=True)

    assert args.finetuning_mode in [
        "linear", "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    ft_path = (
        os.path.join(args.save, f"{tgt_dataset}Val", "linear_finetuned.pt")
        if linearized_finetuning
        else os.path.join(args.save, f"{tgt_dataset}Val", "finetuned.pt")
    )
    zs_path = (
        os.path.join(args.save, f"{tgt_dataset}Val", "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, f"{tgt_dataset}Val", "zeroshot.pt")
    )
    if not os.path.exists(zs_path):
        raise ValueError(f"The checkpoint for the zero-shot model does not exist at {zs_path}.")
    if not os.path.exists(ft_path):
        raise ValueError(f"The checkpoint for the fine-tuned model does not exist at {ft_path}.")

    if args.finetuning_mode == "linear":
        task_vectors = [LinearizedTaskVector(zs_path, ft_path),]
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = WeightedLinearizedModel(
            image_encoder.model, task_vectors,
            device=rank, blockwise=args.blockwise_coef
        )
    else:
        task_vectors = [NonLinearTaskVector(zs_path, ft_path),]
        image_encoder = ImageEncoder(args)
        image_encoder = WeightedImageEncoder(
            image_encoder, task_vectors,
            device=rank, blockwise=args.blockwise_coef
        )

    tgt_classification_head = get_classification_head(args, tgt_dataset)
    ctr_classification_head = get_classification_head(args, ctr_dataset)
    model = MultiHeadImageClassifier(image_encoder,
        [tgt_classification_head, ctr_classification_head]
    )

    model.freeze_head()
    model = model.cuda()

    if args.partition == "trainval":
        tgt_dataset += "Val"
        ctr_dataset += "Val"
    preprocess_fn = model.train_preprocess
    tgt_dataloader = get_dataloader(
        get_dataset(
            tgt_dataset, preprocess_fn,
            location=args.data_location,
            batch_size=int(args.batch_size / 2),
            num_workers=2),
        is_train=True, args=args, image_encoder=None
    )
    ctr_dataloader = get_dataloader(
        get_dataset(
            ctr_dataset, preprocess_fn,
            location=args.data_location,
            batch_size=int(args.batch_size / 2),
            num_workers=2),
        is_train=True, args=args, image_encoder=None
    )
    num_batches = len(tgt_dataloader)
    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = int(num_batches / 4)
    else:
        print_every = args.print_every

    # Distribute the data and model across the GPUs.
    ddp_tgt_loader = distribute_loader(tgt_dataloader)
    ddp_ctr_loader = distribute_loader(ctr_dataloader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=False,
        output_device=rank,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if linearized_finetuning:
        head_path = os.path.join(ckpdir, "linear_learned_negation.pt")
        log_path = os.path.join(args.save, "linear_learned_negations.json")
        coef = ddp_model.module.image_encoder.model.coef
    else:
        head_path = os.path.join(ckpdir, "learned_negation.pt")
        log_path = os.path.join(args.save, "learned_negations.json")
        coef = ddp_model.module.image_encoder.coef

    scaler = GradScaler()
    tgt_zs_acc = args.zs_acc[tgt_dataset]
    best_acc = tgt_zs_acc
    ctr_zs_acc = args.zs_acc[ctr_dataset]
    if is_main_process():
        print(f"=> Zero-shot accuracy on {tgt_dataset} (target): {100*tgt_zs_acc:.2f}%.")
        print(f"=> Zero-shot accuracy on {ctr_dataset} (control): {100*ctr_zs_acc:.2f}%.")
        if os.path.exists(log_path):
            with open(log_path) as f:
                neg_acc = json.load(f)
        else:
            neg_acc = {"trainval": {}, "traintest": {}}

    for epoch in range(args.epoch):
    
        ddp_tgt_loader.sampler.set_epoch(epoch)
        ddp_ctr_loader.sampler.set_epoch(epoch)
        ctr_iter = iter(ddp_ctr_loader)
        for i, batch in enumerate(ddp_tgt_loader):
            ctr_batch = next(ctr_iter)
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            ctr_batch = maybe_dictionarize(ctr_batch)
            inputs = torch.cat([batch["images"].cuda(), ctr_batch["images"].cuda()])
            data_time = time.time() - start_time
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(inputs, [int(args.batch_size / 2)] * 2)
                labels = [batch["labels"].cuda(), ctr_batch["labels"].cuda()]
                loss_tgt, loss_ctr = [loss_fn(x, y) for x, y in zip(logits, labels)]
                """Gradient ascent on the target dataset,
                gradient descent on the control dataset."""
                loss = -loss_tgt + args.gamma * loss_ctr

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(ddp_tgt_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{len(ddp_tgt_loader)}]\t"   # noqa: E501
                    f"Loss (tgt.): {loss_tgt.item():.6f}\tLoss (ctr.): {loss_ctr.item():.6f}\t"         # noqa: E501
                    f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",                            # noqa: E501
                    flush=True,
                )

        if is_main_process():
            # We only need to evaluate the model on the first GPU.
            image_encoder = ddp_model.module.image_encoder
            tgt_acc = eval_single_dataset(image_encoder, tgt_dataset, args)["top1"]
            ctr_acc = eval_single_dataset(image_encoder, ctr_dataset, args)["top1"]

            # Save the best coefficient
            if tgt_acc < best_acc and ctr_acc >= ctr_zs_acc * args.control_threshold:
                best_acc = tgt_acc
                torch.save(coef, head_path)
                neg_acc[args.partition][tgt_dataset] = best_acc
                neg_acc[args.partition][ctr_dataset] = ctr_acc
                with open(log_path, 'w') as f:
                    json.dump(neg_acc, f, indent=4)

    cleanup_ddp()

if __name__ == "__main__":

    datasets = {
        "Cars": 10,
        "DTD": 10,
        "EuroSAT": 10,
        "GTSRB": 10,
        "MNIST": 10,
        "RESISC45": 10,
        "SUN397": 10,
        "SVHN": 10,
    }

    args = parse_arguments()
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    args.ctr_dataset = "ImageNet"
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"
    with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
        args.zs_acc = json.load(f)

    for dataset in datasets:
        args.tgt_dataset = dataset
        args.epoch = datasets[dataset]
        print("=" * 100)
        print(f"Learn task vector coefficients of {args.model} on {dataset}")
        print("=" * 100)
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)