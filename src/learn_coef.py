"""Given an objective,
learn the coefficients on task vectors for a dataset
and find the optimal combination.

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import os
import time
import json
import torch
import torchvision

from torch.cuda.amp import GradScaler
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.composition import WeightedImageEncoder, WeightedLinearizedModel

from src.utils import cosine_lr
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()


def main(rank, args):

    # Load the individual task vectors.
    pool = [
        "sciq",
        "quail",
        "commonsense_qa",
    ]
    task_vectors = {}
    for dataset in pool:

        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned.pt"
            task_vectors[dataset] = LinearizedTaskVector(
                pretrained_checkpoint, finetuned_checkpoint
            )
        else:
            pretrained_checkpoint = f"{args.save}/"  # {args.checkpoint_fname}"
            finetuned_checkpoint = f"{args.save}/{dataset}/{args.num_shots}_shot"  # /{args.checkpoint_fname}"
            task_vectors[dataset] = NonLinearTaskVector(
                pretrained_checkpoint, finetuned_checkpoint, lora=args.is_lora
            )

    args.rank = rank
    for dataset in args.datasets:
        args.target_dataset = dataset + "Val"
        print("=" * 100)
        print(f"Learning task vector coefficients on {dataset} with {args.model} ")
        print("=" * 100)

        train(task_vectors, args)


def train(task_vectors, args):

    setup_ddp(args.rank, args.world_size, port=args.port)
    target_dataset = args.target_dataset
    ckpdir = os.path.join(args.save, target_dataset)
    os.makedirs(ckpdir, exist_ok=True)

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    orig_dataset = target_dataset.replace("Val", "")
    # Remove the task vector for the target task
    task_vectors = [v for k, v in task_vectors.items() if orig_dataset != k]

    if args.finetuning_mode == "linear":
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = WeightedLinearizedModel(
            image_encoder.model, task_vectors, blockwise=args.blockwise_coef
        )
    else:
        image_encoder = ImageEncoder(args)
        image_encoder = WeightedImageEncoder(
            image_encoder, task_vectors, blockwise=args.blockwise_coef
        )

    classification_head = get_classification_head(args, target_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    # Use more aggressive random crop with horizontal flip
    preprocess_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ]
        + model.train_preprocess.transforms[-3:]
    )
    dataset = get_dataset(
        target_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(data_loader)

    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.rank],
        find_unused_parameters=False,
        output_device=args.rank,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # Do not use warm up
    scheduler = cosine_lr(
        optimizer,
        args.lr,
        0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    if linearized_finetuning:
        head_path = os.path.join(ckpdir, "learned_linear_composition.pt")
        log_path = os.path.join(args.save, "learned_linear_composition.json")
        coef = ddp_model.module.image_encoder.model.coef
    else:
        head_path = os.path.join(ckpdir, "learned_composition.pt")
        log_path = os.path.join(args.save, "learned_composition.json")
        coef = ddp_model.module.image_encoder.coef

    scaler = GradScaler()
    if is_main_process():
        print(
            f"=> Zero-shot accuracy on {target_dataset}:\t{100*args.zs_acc[target_dataset]:.2f}%."
        )
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                comp_acc = json.load(f)
        else:
            comp_acc = {}

    best_coef = None
    best_acc = args.zs_acc[target_dataset]
    for epoch in range(args.epochs):
        ddp_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            data_time = time.time() - start_time

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = ddp_model(inputs)
                labels = batch["labels"].cuda()
                loss = loss_fn(logits, labels)
                # Apply regularisation if needed.
                reg = lp_reg(coef, args.lp_reg)
                loss = loss + reg
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

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
                percent_complete = 100 * (i + 1) / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )

        # Evaluate after each epoch
        if is_main_process():
            image_encoder = ddp_model.module.image_encoder
            acc = eval_single_dataset(image_encoder, target_dataset, args)["top1"]
            if acc > best_acc:
                best_acc = acc
                best_coef = coef.data.clone()
                torch.save(best_coef, head_path)

    if is_main_process():
        comp_acc[target_dataset] = best_acc
        target_dataset = target_dataset.replace("Val", "")
        image_encoder = ddp_model.module.image_encoder
        if linearized_finetuning:
            image_encoder.model.coef = torch.nn.Parameter(best_coef)
        else:
            image_encoder.coef = torch.nn.Parameter(best_coef)
        comp_acc[target_dataset] = eval_single_dataset(
            image_encoder, target_dataset, args
        )["top1"]
        with open(log_path, "w") as f:
            json.dump(comp_acc, f, indent=4)

    cleanup_ddp()


if __name__ == "__main__":

    target_datasets = [
        "arc-challenge",
    ]

    args = parse_arguments()
    args.datasets = target_datasets
    # HACK: Some command line arguments are overwritten by defaults here.
    args.lr = 1e-1
    args.epochs = 10
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 64 if args.model == "flan-t5-large" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"
    if args.subsample is not None:
        args.save += f"_{args.subsample*100:.0f}perc"

    # with open(os.path.join(args.save, "zeroshot_accuracies.json"), "r") as f:
    #     args.zs_acc = json.load(f)
    main(4, args)
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
