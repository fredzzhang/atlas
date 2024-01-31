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
from src.modeling import ImageEncoder, ImageClassifierWithOrthogReg
from src.utils import LabelSmoothing, cosine_lr
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector


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
    zs_path = (
        os.path.join(args.save, train_dataset, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "zeroshot.pt")
    )
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

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

    # Load the previous task vectors
    task_vectors = []
    for prev in args.previous_datasets:
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{prev}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{prev}Val/linear_finetuned.pt"
            task_vectors.append(
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        else:
            pretrained_checkpoint = f"{args.save}/{prev}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{prev}Val/finetuned.pt"
            task_vectors.append(
                NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
    if len(task_vectors):
        task_vector = sum(task_vectors)
    else:
        task_vector = None
    # Relocate the task vector to the designated device
    task_vector.to(f"cuda:{rank}")

    # Build the classification head with all classes, when the dataset only has one.
    if '_' in train_dataset:
        train_dataset_ = train_dataset.split('_')[-1]
    else:
        train_dataset_ = train_dataset
    classification_head = get_classification_head(args, train_dataset_)

    zs_model_path = (
        os.path.join(ckpdir, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(ckpdir, "zeroshot.pt")
    )
    model = ImageClassifierWithOrthogReg(zs_model_path, image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

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
        ddp_model.module.image_encoder.save(zs_model_path)

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

            logits, reg = ddp_model(task_vector, inputs)

            cls_loss = loss_fn(logits, labels)
            # Add the orthognality regularisation term
            reg = reg * args.orthog_coef
            loss = cls_loss + reg
            loss.backward()

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
                ddp_model.module.image_encoder.save(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                reg_val = reg if isinstance(reg, float) else reg.item()
                percent_complete = 100 * i / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"                # noqa: E501
                    f"Loss: {loss.item():.6f}\tCls loss: {cls_loss.item():.6f}\tOrthog. Reg.: {reg_val:.6f}\t"         # noqa: E501
                    f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",                                           # noqa: E501
                    flush=True,
                )

    # FIXME: Make this work with DDP.
    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = ddp_model.module.image_encoder
        eval_single_dataset(image_encoder, train_dataset, args)

    if args.save is not None and is_main_process():
        zs_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        ft_path = (
            os.path.join(ckpdir, "linear_finetuned.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "finetuned.pt")
        )
        image_encoder.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == "__main__":
    train_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
        # "01_MNIST",
        # "23_MNIST",
        # "45_MNIST",
        # "67_MNIST",
        # "89_MNIST",
    ]
    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
        # "01_MNIST": 1,
        # "23_MNIST": 1,
        # "45_MNIST": 1,
        # "67_MNIST": 1,
        # "89_MNIST": 1,
    }

    previous_datasets = []
    for dataset in train_datasets:
        args = parse_arguments()
        args.previous_datasets = previous_datasets

        # HACK: Some command line arguments are overwritten by defaults here.
        args.lr = 1e-5
        args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.batch_size = 64 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

        if args.seed is not None:
            args.save = f"checkpoints_{args.seed}/{args.model}"
        else:
            args.save = f"checkpoints/{args.model}"
        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)

        # Add the current dataset for orthognality constraint.
        previous_datasets.append(dataset)