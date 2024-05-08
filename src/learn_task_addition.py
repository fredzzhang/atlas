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

def avg(x):
    return sum(x) / len(x)

def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()

def main(rank, args):

    setup_ddp(rank, args.world_size, port=args.port)

    datasets = [f"{dataset}Val" for dataset in args.datasets]
    n_datasets = len(datasets)
    ckpdir = os.path.join(args.save, f"combined_{n_datasets}")
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
            pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned.pt"
            task_vectors.append(LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint))
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
            task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))

    if args.finetuning_mode == "linear":
        with open(os.path.join(args.save, "linear_ft_accuracies.json"), 'r') as f:
            args.ft_acc = json.load(f)
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = WeightedLinearizedModel(
            image_encoder.model, task_vectors, blockwise=args.blockwise_coef
        )
    else:
        with open(os.path.join(args.save, "ft_accuracies.json"), 'r') as f:
            args.ft_acc = json.load(f)
        image_encoder = ImageEncoder(args)
        image_encoder = WeightedImageEncoder(
            image_encoder, task_vectors, blockwise=args.blockwise_coef
        )

    preprocess_fn = image_encoder.train_preprocess
    # Prepare all dataloaders.
    dataloaders = [get_dataloader(
        get_dataset(
            dataset, preprocess_fn,
            location=args.data_location,
            batch_size=int(args.batch_size / n_datasets),
            num_workers=2),
        # Use the validation set to learn the coefficients.
        is_train=False, args=args, image_encoder=None
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
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    # Distribute the data and model across the GPUs.
    ddp_prim_loader = distribute_loader(prim_loader)
    ddp_rmng_loader = [distribute_loader(dataloader) for dataloader in dataloaders]
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
        head_path = os.path.join(ckpdir, "learned_linear_additions.pt")
        log_path = os.path.join(args.save, "learned_linear_additions.json")
        coef = ddp_model.module.image_encoder.model.coef
    else:
        head_path = os.path.join(ckpdir, "learned_additions.pt")
        log_path = os.path.join(args.save, "learned_additions.json")
        coef = ddp_model.module.image_encoder.coef
    if args.subsample is not None:
        head_path = head_path[:-3] + f"_{args.subsample*100:.0f}perc.pt"
        log_path = log_path[:-5] + f"_{args.subsample*100:.0f}perc.json"

    scaler = GradScaler()
    zs_acc = args.zs_acc
    zs_acc_norm = {
        dataset: args.zs_acc[dataset] / args.ft_acc[dataset]
        for dataset in datasets
    }
    best_acc = avg(zs_acc_norm.values())
    if is_main_process():
        for dataset in datasets:
            print(
                f"=> Zero-shot accuracy on {dataset:<12}:\t{100*zs_acc[dataset]:.2f}%, "
                f"normalised by f.t. acc.: {100*zs_acc_norm[dataset]:.2f}%.")
    best_coef = None
    val_acc = []
    for epoch in range(args.epoch):
    
        ddp_prim_loader.sampler.set_epoch(epoch)
        for loader in ddp_rmng_loader:
            loader.sampler.set_epoch(epoch)
        rmng_iter = [iter(loader) for loader in ddp_rmng_loader]
        for i, batch in enumerate(ddp_prim_loader):
            rmng_batch = [next(r_iter) for r_iter in rmng_iter]
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            rmng_batch = [maybe_dictionarize(r_batch) for r_batch in rmng_batch]
            inputs = torch.cat(
                [batch["images"].cuda(),] +
                [r_batch["images"].cuda() for r_batch in rmng_batch]
            )
            data_time = time.time() - start_time
            
            split = [len(batch["images"]),] + [len(r_batch["images"]) for r_batch in rmng_batch]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(inputs, split)
                labels = [batch["labels"].cuda(),] + [r_batch["labels"].cuda() for r_batch in rmng_batch]
                all_losses = [loss_fn(x, y) for x, y in zip(logits, labels)]
                loss = sum(all_losses)
                # Apply regularisation if needed.
                reg = lp_reg(coef, args.lp_reg)
                loss = loss + reg
                # Scale the loss.
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if i % args.num_grad_accumulation == 0:

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and (i % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_prim_loader)
                print_losses = str([round(x.item(), 4) for x in all_losses])
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(ddp_prim_loader)}]\t"          # noqa: E501
                    f"Losses: {print_losses:<72}\t"                                                         # noqa: E501
                    f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",                                # noqa: E501
                    flush=True,
                )

        if is_main_process():
            # We only need to evaluate the model on the first GPU.
            image_encoder = ddp_model.module.image_encoder
            all_acc = [eval_single_dataset(
                image_encoder, dataset, args
                )["top1"] for dataset in datasets
            ]
            all_acc_norm = [
                acc / args.ft_acc[dataset]
                for acc, dataset in zip(all_acc, datasets)
            ]

            # Save the best coefficients.
            if avg(all_acc_norm) > best_acc:
                best_acc = avg(all_acc_norm)
                best_coef = coef.data.clone()
                torch.save(best_coef, head_path)
            cur_acc = {f"{dataset}:top1": acc for dataset, acc in zip(datasets, all_acc)}
            cur_acc.update({
                f"{dataset}:normalised_top1": acc
                for dataset, acc in zip(datasets, all_acc_norm)
            })
            cur_acc.update({
                "avg:top1": avg(all_acc),
                "avg_normalised_top1": avg(all_acc_norm)
            })
            val_acc.append(cur_acc)

    # Log stats and test the model with the optimal coefficients.
    datasets = [dataset.replace("Val", "") for dataset in datasets]
    if is_main_process():
        addition_acc = {"val": val_acc}
        image_encoder = ddp_model.module.image_encoder
        if linearized_finetuning:
            image_encoder.model.coef = torch.nn.Parameter(best_coef)
        else:
            image_encoder.coef = torch.nn.Parameter(best_coef)

        test_acc = {f"{dataset}:top1":
            eval_single_dataset(image_encoder, dataset, args)["top1"]
            for dataset in datasets
        }
        test_acc["avg_top1"] = avg(test_acc.values())

        test_acc_norm = {
            f"{dataset}:normalised_top1": acc / args.ft_acc[dataset]
            for dataset, acc in zip(datasets, test_acc.values())
        }

        test_acc.update(test_acc_norm)
        test_acc["avg_normalised_top1"] = avg(test_acc_norm.values())

        addition_acc["test"] = test_acc
        with open(log_path, 'w') as f:
            json.dump(addition_acc, f, indent=4)

    cleanup_ddp()

if __name__ == "__main__":

    datasets = [
        "Cars", "DTD", "EuroSAT", "GTSRB",
        "MNIST", "RESISC45", "SUN397", "SVHN",
    ]

    args = parse_arguments()
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    args.datasets = datasets
    args.epoch = 20
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"
    with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
        args.zs_acc = json.load(f)

    print("=" * 100)
    print(f"Learn task vector coefficients of {args.model} on {len(datasets)} datasets:")
    for i, x in enumerate(datasets):
        print(f"{i + 1}. {x}")
    print("=" * 100)
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
