"""Given an objective and a target dataset,
learn layerwise coefficients on an external task vector.

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import os
import time

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.modeling import ImageEncoder, ImageClassifier
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.utils import cosine_lr

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class LinearizedModel_(nn.Module):
    def __init__(self, model, task_vectors, device) -> None:
        """A wrapper class to enable compositions of task vectors"""
        super().__init__()

        self.params0 = model.params0
        self.func0 = model.func0
        self.buffers0 = model.buffers0
        self._model_name = model._model_name

        self.device = device

        dparams = []
        for tv in task_vectors:
            dp = [tv.vector[k].to(device) for k in tv.vector if k.startswith('model.params.')]
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
    
class ImageEncoder_(nn.Module):
    def __init__(self, model, task_vectors, device, layerwise=True) -> None:
        """A wrapper class to enable compositions of task vectors"""
        super().__init__()

        func, params, self.buffer = make_functional_with_buffers(model)
        # NOTE This is important to avoid the following error
        # NotImplementedError: Cannot copy out of meta tensor; no data!
        self.func = lambda p, x: func(p, self.buffer, x)
        self.params = torch.nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False

        self.device = device
        self.layerwise = layerwise
        # Copy the attributes from the image encoder.
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = model.cache_dir

        dparams = []
        for tv in task_vectors:
            dp = [tv.vector[k].to(device) for k in tv.vector]
            dparams.append(dp)

        self.dparams = dparams
        if layerwise:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
        else:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors),))

    def __call__(self, x) -> torch.Tensor:
        if self.layerwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
        new_params = [dp + p for dp, p in zip(dparams, self.params)]
        return self.func(new_params, x)
    
def main(rank, args):

    # Load the individual task vectors.
    task_vectors = {}
    for dataset in args.datasets:
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors[dataset] = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    args.rank = rank
    for tgt_dataset in ["Cars"]:
        for src_dataset in args.datasets:
            if src_dataset == tgt_dataset:
                continue
            args.epochs = args.datasets[tgt_dataset]
            args.tgt_dataset = tgt_dataset
            args.src_dataset = src_dataset
            print("=" * 100)
            print(f"Learning coefficients for {tgt_dataset} using task vector from {src_dataset}")
            print("=" * 100)

            train([task_vectors[src_dataset],], args)

def train(task_vectors, args):

    setup_ddp(args.rank, args.world_size, port=args.port)
    tgt_dataset = args.tgt_dataset
    ckpdir = os.path.join(args.save, "pairwise_exp")
    os.makedirs(ckpdir, exist_ok=True)

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    if args.finetuning_mode == "linear":
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = LinearizedModel_(image_encoder.model, task_vectors, device=args.rank)
    else:
        image_encoder = ImageEncoder(args)
        image_encoder = ImageEncoder_(image_encoder, task_vectors, device=args.rank)

    classification_head = get_classification_head(args, tgt_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.val_preprocess

    dataset = get_dataset(
        tgt_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    # Override shuffle to True
    data_loader.shuffle = True
    num_batches = len(dataset.train_loader)

    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.rank],
        find_unused_parameters=False,
        output_device=args.rank,
    )

    loss_fn = {
        'entropy': softmax_entropy,
        'cross_entropy': torch.nn.CrossEntropyLoss()
    }[args.loss_fn]

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    if args.finetuning_mode == "linear":
        coef = ddp_model.module.image_encoder.model.coef
    else:
        coef = ddp_model.module.image_encoder.coef

    ddp_model.eval()
    for epoch in range(args.epochs):
        # Evaluate before each epoch
        if is_main_process():
            # We only need to evaluate the model on the first GPU.
            image_encoder = ddp_model.module.image_encoder
            eval_single_dataset(image_encoder, tgt_dataset, args)

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

            logits = ddp_model(inputs)

            if args.loss_fn == 'cross_entropy':
                labels = batch["labels"].cuda()
                loss = loss_fn(logits, labels)
            else:
                loss = loss_fn(logits).mean(0)

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % args.print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{len(dataset.test_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )

    percent_complete = 100 * (i + 1) / len(ddp_loader)
    print(
        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{len(dataset.test_loader)}]\t"  # noqa: E501
        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
        flush=True,
    )
    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = ddp_model.module.image_encoder
        eval_single_dataset(image_encoder, tgt_dataset, args)
        if linearized_finetuning:
            head_path = os.path.join(ckpdir, f"linear_{args.src_dataset}_for_{tgt_dataset}_.pt")
            coef = ddp_model.module.image_encoder.model.coef
        else:
            head_path = os.path.join(ckpdir, f"{args.src_dataset}_for_{tgt_dataset}_.pt")
            coef = ddp_model.module.image_encoder.coef
        torch.save(coef, head_path)

    cleanup_ddp()


if __name__ == "__main__":

    # Epochs w/ lr=1e-2 for entropy objective
    datasets = {
        "Cars": 5,
        "DTD": 3,
        "EuroSAT": 6,
        "GTSRB": 3,
        "MNIST": 10,
        "RESISC45": 2,
        "SUN397": 1,
        "SVHN": 1,
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
    }

    args = parse_arguments()
    args.datasets = datasets
    # HACK: Some command line arguments are overwritten by defaults here.
    args.lr = 1e-2
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
