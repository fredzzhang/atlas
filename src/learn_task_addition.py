"""Learn coefficients on multiple task vectors to produce
a single multi-purpose model, using a supervised objective
on multiple target datasets.

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

    datasets = args.datasets

    ckpdir = os.path.join(args.save, f"combined_{len(datasets)}")
    os.makedirs(ckpdir, exist_ok=True)

    assert args.finetuning_mode in [
        "linear", "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    task_vectors = []
    for dataset in datasets:
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors.append(LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint))
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))

    if args.finetuning_mode == "linear":
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = WeightedLinearizedModel(
            image_encoder.model, task_vectors,
            device=rank, blockwise=args.blockwise_coef
        )
    else:
        image_encoder = ImageEncoder(args)
        image_encoder = WeightedImageEncoder(
            image_encoder, task_vectors,
            device=rank, blockwise=args.blockwise_coef
        )

    preprocess_fn = image_encoder.train_preprocess
    # Prepare all dataloaders.
    dataloaders = [get_dataloader(
        get_dataset(
            dataset, preprocess_fn,
            location=args.data_location,
            batch_size=int(args.batch_size / len(datasets)),
            num_workers=2),
        is_train=True, args=args, image_encoder=None
    ) for dataset in datasets]
    num_batches = [len(dataloader) for dataloader in dataloaders]
    # Select the dataset with smallest size as the primary iterator.
    prim = min(enumerate(num_batches), key=lambda x: x[1])[0]
    num_batches = num_batches[prim]
    prim_loader = dataloaders.pop(prim)

    classification_heads = [get_classification_head(args, dataset) for dataset in datasets]
    # Move the classification head for the primary dataset to the front.
    prim_head = classification_heads.pop(prim)
    classification_heads = [prim_head,] + classification_heads
    model = MultiHeadImageClassifier(image_encoder, classification_heads)

    model.freeze_head()
    model = model.cuda()

    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = int(num_batches / 4)
    else:
        print_every = args.print_every

    # Distribute the data and model across the GPUs.
    ddp_prim_loader = distribute_loader(prim_loader)
    ddp_rest_loader = [distribute_loader(dataloader) for dataloader in dataloaders]
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
        head_path = os.path.join(ckpdir, "linear_learned_additions.pt")
        log_path = os.path.join(args.save, "linear_learned_additions.json")
        coef = ddp_model.module.image_encoder.model.coef
    else:
        head_path = os.path.join(ckpdir, "learned_additions.pt")
        log_path = os.path.join(args.save, "learned_additions.json")
        coef = ddp_model.module.image_encoder.coef

    scaler = GradScaler()
    zs_acc = args.zs_acc
    zs_acc_norm = [args.zs_acc[dataset] / args.ft_acc[dataset] for dataset in datasets]
    best_acc = sum(zs_acc_norm) / len(datasets)
    if is_main_process():
        for dataset in datasets:
            print(
                f"=> Zero-shot accuracy on {dataset}: {100*zs_acc[dataset]:.2f}%,"
                f"normalised: {100*zs_acc_norm[dataset]:.2f}%.")
        if os.path.exists(log_path):
            with open(log_path) as f:
                add_acc = json.load(f)
        else:
            add_acc = {}

    best_coef = None
    neg_acc[tgt_dataset] = {}
    val_acc = []
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

            # Save the best coefficients.
            if tgt_acc < best_acc and ctr_acc >= ctr_zs_acc * args.control_threshold:
                best_acc = tgt_acc
                best_coef = coef.data.clone()
                torch.save(best_coef, head_path)
            val_acc.append({
                f"{tgt_dataset}:top1": tgt_acc,
                f"{ctr_dataset}:top1": ctr_acc,
                f"{tgt_dataset}:normalised_top1": tgt_acc / args.ft_acc[tgt_dataset],
                f"{ctr_dataset}:normalised_top1": ctr_acc / args.ft_acc[ctr_dataset],
            })

    # Log stats and test the model with the optimal coefficients.
    if is_main_process():
        neg_acc[tgt_dataset]["val"] = val_acc
        image_encoder = ddp_model.module.image_encoder
        if linearized_finetuning:
            image_encoder.model.coef = torch.nn.Parameter(best_coef)
        else:
            image_encoder.coef = torch.nn.Parameter(best_coef)
        neg_acc[tgt_dataset]["test"] = eval_single_dataset(
            image_encoder, tgt_dataset.split("Val")[0], args
        )["top1"]
        neg_acc[tgt_dataset]["test_control"] = eval_single_dataset(
            image_encoder, ctr_dataset.split("Val")[0], args
        )["top1"]
        with open(log_path, 'w') as f:
            json.dump(neg_acc, f, indent=4)

    cleanup_ddp()

if __name__ == "__main__":

    datasets = {
        "Cars", "DTD", "EuroSAT", "GTSRB",
        "MNIST", "RESISC45", "SUN397", "SVHN",
    }

    args = parse_arguments()
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    args.datasets = datasets
    args.epoch = 10
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"
    with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
        args.zs_acc = json.load(f)
    with open(os.path.join(args.save, "ft_accuracies.json"), 'r') as f:
        args.ft_acc = json.load(f)

    print("=" * 100)
    print(f"Learn task vector coefficients of {args.model} on {len(datasets)} datasets:")
    for i, x in enumerate(datasets):
        print(f"{i + 1}. {x}")
    print("=" * 100)
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)