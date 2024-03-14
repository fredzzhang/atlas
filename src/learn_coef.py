"""
Fine-tune the coefficients on different task vectors

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import os
import time

import torch
import torch.nn as nn
from functorch import jvp

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.utils import LabelSmoothing, cosine_lr

class LinearizedModel_(nn.Module):
    def __init__(self, linear_model, task_vectors) -> None:
        """Initializes the linearized model."""
        super().__init__()

        self.params0 = linear_model.params0
        self.func0 = linear_model.func0
        self.buffers0 = linear_model.buffers0
        self._model_name = linear_model._model_name

        dparams = []
        for tv in task_vectors:
            # HACK: Pre-moving the tensor to cuda, which will not work for multi-gpu training.
            dp = [tv.vector[k].cuda() for k in tv.vector if k.startswith('model.params.')]
            dparams.append(dp)

        self.dparams = dparams
        self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors),))

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp

def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, f"{train_dataset}_coef")

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    assert train_dataset is not None, "Please provide a training dataset."

    # Load the individual task vectors.
    if train_dataset.startswith("MNIST"):
        class_datasets = [
            "0_MNIST",
            "1_MNIST",
            "2_MNIST",
            "3_MNIST",
            "4_MNIST",
            "5_MNIST",
            "6_MNIST",
            "7_MNIST",
            "8_MNIST",
            "9_MNIST",
        ]
    else:
        raise ValueError(f"The dataset {train_dataset} is not supported in this script.")
    
    task_vectors = []
    for dataset in class_datasets:
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors.append(
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        else:
            raise NotImplementedError("Only the linear models are supported.")
            # pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            # finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            # task_vectors.append(
            #     NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            # )

    image_encoder = task_vectors[0].apply_to(pretrained_checkpoint, scaling_coef=0.0)
    # HACK: Override the liearised model
    image_encoder.model = LinearizedModel_(image_encoder.model, task_vectors)

    classification_head = get_classification_head(args, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    print_every = 100

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

    for epoch in range(args.epochs):
        # Evaluate before each epoch
        if is_main_process():
            # We only need to evaluate the model on the first GPU.
            image_encoder = ddp_model.module.image_encoder
            eval_single_dataset(image_encoder, train_dataset, args)
            coef = ddp_model.module.image_encoder.model.coef.data
            string = 'Coefficients:\t|'
            for c in coef:
                string += f"`{c:.4f}`|"
            print(string)

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

            logits = ddp_model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

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

    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = ddp_model.module.image_encoder
        eval_single_dataset(image_encoder, train_dataset, args)
        coef = ddp_model.module.image_encoder.model.coef.data
        string = 'Coefficients:\t|'
        for c in coef:
            string += f"`{c:.4f}`|"
        print(string)

    cleanup_ddp()


if __name__ == "__main__":
    train_datasets = [
        "MNIST",
    ]
    epochs = {
        "MNIST": 5,
    }

    for dataset in train_datasets:
        args = parse_arguments()

        # HACK: Some command line arguments are overwritten by defaults here.
        args.lr = 1e-2
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
        print(f"Finetuning task vector coefficients of {args.model} on {dataset}")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
