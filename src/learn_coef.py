"""Given an objective,
learn the coefficients on task vectors for a dataset
and find the optimal combination.

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset, eval_single_train_dataset, get_val_features
from src.heads import get_classification_head
from src.modeling import ImageEncoder, ImageClassifier, ImageEncoder_, LinearizedModel_
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.utils import (
    cosine_lr,
    cosine_annealing_lr,
    adjust_lr,
    TwoTransform,
    TwoAsymetricTransform,
    adjust_lr_lp,
    extract_datasets,
    IndexWrapper,
    update_transforms,
    get_n_shots,
)
from src.sampler import TwoStreamBatchSampler, SubsetSampler
from src.sar import SAR, SAM

import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from lightning.fabric import Fabric, seed_everything
import minlora
import gc


import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # cond = (x.softmax(1).max(dim=1)[0] < .9)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
    # return (-(x.softmax(1) * x.log_softmax(1)).sum(1) * cond).sum() / cond.sum()


@torch.jit.script
def cb_softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    entro = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    prior = torch.ones(x.shape[1]) / x.shape[1]
    prior = prior.to(x.device)
    pred_mean = x.softmax(1).mean(0)
    cb_penalty = torch.sum(prior * torch.log(prior / pred_mean))

    return entro + cb_penalty


def l1_reg(coefs: torch.Tensor, lambda1: float = 0.5) -> torch.Tensor:
    """Simple L1 regularization."""
    l1_regularization = lambda1 * torch.norm(coefs, p=1, dim=0)
    return l1_regularization.mean()


def ce_loss_trusted(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    preds: torch.FloatTensor = None,
    trusted: torch.BoolTensor = None,
    lam: float = None,
    index: torch.Tensor = None,
) -> torch.Tensor:
    trusted = trusted.to(logits1)
    return (
        F.cross_entropy(logits2, preds.to(logits2), reduction="none") * trusted
    ).sum() / trusted.sum()


def ssl_loss_trusted(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    targets: torch.FloatTensor = None,
    trusted: torch.BoolTensor = None,
    lam: float = None,
    index: torch.Tensor = None,
    thresh: float = 0.99,
) -> torch.Tensor:
    """
    Computes a Fixmatch style semi supervised loss given trusted samples.

    Args:
    logits1: network logits using an unaugmented view of the image
    logits2: network logits using a strongly augemented view of the image (e.g. RandAugment or SimCLR augs)
    targets: the ground-truth labels for the batch, only used for the trusted part of the batch
    trusted: a BoolTensor with the size of the batch indicating the trusted samples (True)
    lam: the mixing ratio if mixup is used for logits2 (logits1 should not be mixed). If left as None, loss mixup will not be applied.
    index: the mixing index where index[i] indicates the index of the sample i is mixed with in logits2.
    thresh: the threshold on the confidence value for selecting pseudo-labels. Using an adaptive threshold is recommended.

    Returns:
    The semi-supervised cross-entropy loss with a trusted set.
    """
    one_hot = logits1.softmax(1).detach()

    guessed_targets = one_hot ** (0.5)  # temp sharp
    guessed_targets = guessed_targets / guessed_targets.sum(
        dim=1, keepdim=True
    )  # normalization

    if trusted is not None:
        guessed_targets[trusted] = targets[trusted].to(guessed_targets)

    one_hot = guessed_targets.detach()
    one_hot = F.one_hot(
        torch.argmax(guessed_targets, dim=1), num_classes=one_hot.shape[1]
    ).float()

    w, _ = guessed_targets.max(1)
    w = (w > thresh).to(logits2)
    if lam is not None:
        return lam * F.cross_entropy(logits2, guessed_targets) + (
            1 - lam
        ) * F.cross_entropy(logits2, guessed_targets[index])
    trusted = trusted.to(logits1)
    # return (F.cross_entropy(logits2, guessed_targets, reduction='none')*trusted).sum() / trusted.sum()
    return (
        F.cross_entropy(logits2, guessed_targets, reduction="none") * w
    ).sum() / w.sum()


def simclr_loss(
    logits1: torch.Tensor, logits2: torch.Tensor, mu: float = 0.2
) -> torch.Tensor:
    """Simclr contrastive loss (Npairs for now)"""
    logits1, logits2 = F.normalize(logits1, 2), F.normalize(logits2, 2)
    mask = torch.eye(logits1.shape[0]).to(logits1)

    sims = torch.div(logits1 @ logits2.T, mu)

    npairs_loss = -(mask * sims.log_softmax(1)).sum(1)
    return npairs_loss


def simclr_mixup_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    lam: float,
    index: torch.tensor,
    mu: float = 0.2,
) -> torch.Tensor:
    """iMix contrastive loss"""
    logits1, logits2 = F.normalize(logits1, 2), F.normalize(logits2, 2)
    mask = torch.eye(logits1.shape[0]).to(logits1)

    sims = torch.div(logits1 @ logits2.T, mu)

    npairs_loss = -lam * (mask * sims.log_softmax(1)).sum(1) - (1 - lam) * (
        mask[index] * sims.log_softmax(1)
    ).sum(1)
    return npairs_loss


def main(rank, args):

    # Load the individual task vectors.
    pool = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
        "CIFAR10",
        "CIFAR100",
        "ImageNet",
        "STL10",
        "Food101",
        "Caltech256",
        "FGVCAircraft",
        "Flowers102",
        "OxfordIIITPet",
        "CUB200",
        "PascalVOC",
        "Country211",
        "Caltech101",
        "UCF101",
    ]

    VIT_B_32_hugg = [
        "laion400m_e31",
        "laion400m_e32",
        "laion2b_e16",
        "laion2b_s34b_b79k",
        "datacomp_xl_s13b_b90k",
        "datacomp_m_s128m_b4k",
        "commonpool_m_clip_s128m_b4k",
        "commonpool_m_laion_s128m_b4k",
        "commonpool_m_image_s128m_b4k",
        "commonpool_m_text_s128m_b4k",
        "commonpool_m_basic_s128m_b4k",
        "commonpool_m_s128m_b4k",
        "datacomp_s_s13m_b4k",
        "commonpool_s_clip_s13m_b4k",
        "commonpool_s_laion_s13m_b4k",
        "commonpool_s_image_s13m_b4k",
        "commonpool_s_text_s13m_b4k",
        "commonpool_s_basic_s13m_b4k",
        "commonpool_s_s13m_b4k",
    ]  #'openai' (zero-shot base)

    if args.add_random_tv is not None:
        pool = []
        for i in range(args.add_random_tv):
            pool.append(f"randomtv_{i}")
    if args.hugg:
        pool = VIT_B_32_hugg
    task_vectors = {}
    l = 0
    for i, dataset in enumerate(pool):
        if args.lora:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/lora_{args.rank}{'_attn_only' if args.attn_only else ''}{'_mlp_only' if args.mlp_only else ''}.pt"
            task_vectors[dataset] = NonLinearTaskVector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                scale=args.scale,
                lora=args.lora,
            )
        elif args.hugg:
            pretrained_checkpoint = f"{args.save}/CarsVal/zeroshot.pt"
            task_vectors[dataset] = NonLinearTaskVector(
                pretrained_checkpoint, scale=args.scale, hugg_checkpoint=dataset
            )
        elif "randomtv" in dataset:
            pretrained_checkpoint = f"{args.save}/CarsVal/zeroshot.pt"
            task_vectors[dataset] = NonLinearTaskVector(
                pretrained_checkpoint, scale=args.scale
            )
        elif args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors[dataset] = LinearizedTaskVector(
                pretrained_checkpoint, finetuned_checkpoint
            )
        elif args.optimally_random is not None:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            opt_tv = torch.load(os.path.join(args.optimally_random, f"tv_{i}.pth"))
            task_vectors[dataset] = NonLinearTaskVector(
                pretrained_checkpoint, vector=opt_tv["state_dict"], scale=args.scale
            )
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors[dataset] = NonLinearTaskVector(
                pretrained_checkpoint, finetuned_checkpoint, scale=args.scale
            )

    fname = f"results/{args.model}/"
    if args.fname is not None:
        fname = os.path.join(args.fname, str(args.seed))
    elif args.datasets[list(args.datasets.keys())[0]] == 0:
        fname = os.path.join(fname, "zero-shot/")
    else:
        if args.add_random_tv:
            fname = os.path.join(fname, f"{args.add_random_tv}_random_tv")
        else:
            fname = os.path.join(fname, f"{len(args.datasets)}_ft_tv")

        if args.semi:
            fname = os.path.join(fname, f"{args.semi}-shots")
        elif args.loss_fn == "cross_entropy":
            fname = os.path.join(fname, f"fully-supervised")
        elif args.loss_fn == "ssl_simclr_trusted":
            fname = os.path.join(fname, f"self-supervised")
        elif args.loss_fn == "entropy":
            fname = os.path.join(fname, f"entropy")
        elif args.loss_fn == "simclr":
            fname = os.path.join(fname, f"simclr")
        elif args.loss_fn == "entropy_sar":
            fname = os.path.join(fname, f"entropy-sar")
        else:
            raise NotImplementedError("The current setting is not implemented")

        if args.tip_ft:
            if args.tip_cot:
                fname = os.path.join(fname, "tv-lp-cot" if args.lp else "tv-tip-cot")
            elif args.tip_only:
                fname = os.path.join(fname, "lp" if args.lp else "tip")
            else:
                fname = os.path.join(fname, "tv-lp" if args.lp else "tv-tip")
        else:
            fname = os.path.join(fname, "tv")

    if args.fname is not None:
        fname = os.path.join(fname, f"results.txt")
        if not args.merge:
            try:
                os.remove(fname)
                os.remove(fname.replace(".txt", ".pth"))
            except OSError:
                pass

    else:
        fname = os.path.join(
            fname, f"{datetime.today().strftime('%Y-%m-%d-%H-%M')}.txt"
        )

    splt = fname.split("/")
    for i in range(len(splt) - 1):
        if not os.path.exists("/".join(splt[: i + 1])):
            os.mkdir("/".join(splt[: i + 1]))
    if not args.no_log and not args.merge:
        with open(fname, "a") as f:
            f.writelines(
                [
                    f"Task vector results for loss {args.loss_fn} on datasets {list(args.datasets.keys())}. {len(task_vectors.keys())} task vectors\n"
                ]
            )

    if args.merge:
        if not os.path.isfile(fname.replace(".txt", ".pth")):
            print(f"Could not find results file to merge with, restarting from the top")
            try:
                os.remove(fname)
                os.remove(fname.replace(".txt", ".pth"))
            except OSError:
                pass
            with open(fname, "a") as f:
                f.writelines(
                    [
                        f"Task vector results for loss {args.loss_fn} on datasets {list(args.datasets.keys())}. {len(task_vectors.keys())} task vectors\n"
                    ]
                )

            coef_dict = {}
        else:
            coef_dict = torch.load(fname.replace(".txt", ".pth"))
    else:
        coef_dict = {}

    for dataset in args.datasets:
        args.epochs = args.datasets[dataset]
        args.test_dataset = dataset
        print("=" * 100)
        print(f"Finetuning task vector coefficients of {args.model} on {dataset}")
        print("=" * 100)

        if args.pool_one:
            pool_size = len(task_vectors)
            keys = task_vectors.keys()
            keys = [k for k in keys if k != dataset.replace("Val", "")]
            for p in range(len(keys)):
                base_acc, best_acc, best_epoch, best_coef, acc = train(
                    {keys[p]: task_vectors[keys[p]]}, args
                )

                with open(fname, "a") as f:
                    f.writelines(
                        [
                            f"{dataset} - TV {keys[p]}\n",
                            f"Base accuracy {base_acc}, best accuracy {best_acc} at epoch {best_epoch}\n, tip beta {acc['beta_alpha'][0]:.3f} alpha {acc['beta_alpha'][1]:.3f}. Accuracy on OOD datasets {[acc[k] for k in acc.keys() if 'top1' in k]}",
                            f"{best_coef}\n",
                        ]
                    )

        else:
            base_acc, best_acc, best_epoch, best_coef, acc = train(task_vectors, args)

            coef_dict[dataset] = {
                "coefs": acc["coefs"],
                "preds": acc["softmax"],
                "targets": acc["targets"],
            }
            print(acc.keys())
            if args.tip_ft:
                coef_dict[dataset]["adapter"] = acc["adapter"]
                if args.lp:
                    coef_dict[dataset]["alpha_vec"] = acc["alpha_vec"]
                else:
                    coef_dict[dataset]["beta_alpha"] = acc["beta_alpha"]

            if not args.no_log:
                torch.save(coef_dict, fname.replace(".txt", ".pth"))
                with open(fname, "a") as f:
                    f.writelines(
                        [
                            f"{dataset}\n",
                            f"Base accuracy {base_acc}, best accuracy {best_acc} at epoch {best_epoch}\n, tip beta {acc['beta_alpha'][0]:.3f} alpha {acc['beta_alpha'][1]:.3f}. Accuracy on OOD datasets {[acc[k] for k in acc.keys() if 'top1' in k]}",
                            f"{best_coef}\n",
                        ]
                    )

    if not args.no_log and len(args.datasets) > 0:
        with open(fname, "a") as f:
            f.writelines([f"\nArguments {args}"])


def train(task_vectors, args):
    if args.loss_fn in [
        "simclr",
        "ssl_simclr_trusted",
        "entropy",
        "entropy_sar",
    ]:  # Test-time adaptations
        test_dataset = args.test_dataset
    else:
        test_dataset = args.test_dataset + "Val"

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    orig_dataset = test_dataset.split("Val")[0]
    if orig_dataset == "Webvision":
        orig_dataset = "ImageNet"

    pool = [k for k, v in task_vectors.items() if orig_dataset != k]
    task_vectors = [v for k, v in task_vectors.items() if orig_dataset != k]

    if "entropy_sar" in args.loss_fn:
        fabric = Fabric(accelerator="cuda", devices=1, strategy="dp", precision="32")
    else:
        fabric = Fabric(
            accelerator="cuda", devices=1, strategy="ddp", precision="16-mixed"
        )
    fabric.launch()
    args.fabric = fabric

    if args.finetuning_mode == "linear":
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = LinearizedModel_(image_encoder.model, task_vectors, args)
    else:
        image_encoder = ImageEncoder(args)
        image_encoder = ImageEncoder_(image_encoder, task_vectors, args)

    classification_head = get_classification_head(args, test_dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()

    # Using the TIP augmentations to learn the coeficients. The larger scale=(0.5, 1) of the RandomResizedCrop is important to get good results on some datasets.
    preprocess_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ]
        + model.val_preprocess.transforms[-3:]
    )

    preprocess_fn = model.val_preprocess

    if "simclr" in args.loss_fn or "ssl" in args.loss_fn:
        size = 224
        preprocess_fn = TwoAsymetricTransform(model.val_preprocess, preprocess_fn)

    dataset = get_dataset(
        test_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    if args.loss_fn not in ["simclr", "ssl_simclr_trusted", "entropy"]:
        # Changing the transforms in the val/test dataset
        dataloader = get_dataloader(
            dataset, is_train=False, args=args, image_encoder=None
        )

    # Wrapping to get index of the samples
    if args.subsample is not None:
        da = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
        index_dataset = IndexWrapper(
            da.dataset
        )  # Subset dataset created in get_dataloader
    elif args.loss_fn in ["simclr", "ssl_simclr_trusted", "entropy", "entropy_sar"]:
        index_dataset = IndexWrapper(
            dataset.test_dataset
        )  # Test time adapatation (no labels used)
    else:
        index_dataset = IndexWrapper(dataset.train_dataset)

    data_loader = torch.utils.data.DataLoader(
        index_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.persistant_workers,
    )

    # Override shuffle to True
    data_loader.shuffle = True
    num_batches = len(data_loader)

    # TO DO: implement lightning Fabric support for DDP and multi-gpu
    loss_fn = {
        "entropy": softmax_entropy,
        "entropy_sar": torch.nn.CrossEntropyLoss(),  # Ununsed
        "cb_entropy": cb_softmax_entropy,
        "cross_entropy": torch.nn.CrossEntropyLoss(),
        "trusted": torch.nn.CrossEntropyLoss(),
        "trusted_mixup": torch.nn.CrossEntropyLoss(),
        "entro_trusted": softmax_entropy,
        "simclr": simclr_loss,
        "ssl_simclr_trusted": ssl_loss_trusted,
        "ssl_simclr_trusted_entro": ssl_loss_trusted,
        "simclr_mixup": simclr_mixup_loss,
    }[args.loss_fn]

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params {sum([p.numel() for p in params])}")

    if args.tune_clip:
        optimizer = torch.optim.AdamW(
            [
                {"params": [model.image_encoder.coef] + [model.image_encoder.coef1]},
                {"params": model.image_encoder.params},
            ],
            lr=args.lr,
            weight_decay=args.wd,
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-5)

    if args.lr_scheduler == "annealing":
        lrs = cosine_annealing_lr(args.lr, 0, args.epochs)
    else:
        scheduler = cosiqne_lr(
            optimizer,
            args.lr,
            args.warmup_length,
            args.epochs * num_batches // args.num_grad_accumulation,
        )

    best_acc = 0
    max_ep = args.epochs
    if args.tip_only:
        max_ep = 1

    if "ssl" in args.loss_fn:
        threshs = torch.arange(args.epochs) / args.epochs / 10 + 0.9
    i = 0
    loss = torch.tensor(0)
    data_time = 0
    batch_time = 0
    epoch = 0

    # Calling fabric.setup() causes a repetitive memory increase with each new call for new datasets. Once the GPU reaches max memory, a flush happens somewhere to free up everything. Then the cycle starts again. gc.collect();torch.cuda.empty_cache() prevents the increase in case you wish to run multiple processes on the same GPU.
    gc.collect()
    torch.cuda.empty_cache()
    if "entropy_sar" in args.loss_fn:
        optimizer = SAM(params, torch.optim.SGD, lr=args.lr, momentum=0.9)
        model = SAR(model, optimizer, fabric)

    model, optimizer = fabric.setup(model, optimizer)

    data_loader = fabric.setup_dataloaders(data_loader, use_distributed_sampler=False)

    if args.finetuning_mode == "linear":
        coef = model.module.image_encoder.model.coef
        if args.attn:
            coef1 = model.module.image_encoder.model.coef1
    else:
        coef = model.module.image_encoder.coef
        if args.attn:
            coef1 = model.module.image_encoder.coef1

    model.eval()
    gc.collect()
    torch.cuda.empty_cache()  # Same as above, this memory problem appears to be related to moving TVs to cuda.
    if args.finetuning_mode == "linear":
        model.image_encoder.model = model.image_encoder.model.tv_to_device()
    else:
        model.image_encoder = model.image_encoder.tv_to_device()

    for epoch in range(max_ep):

        # Evaluate before each epoch
        if args.lr_scheduler == "annealing":
            adjust_lr(
                optimizer, lrs[epoch], [1, 0.0001]
            )  # Lower lr for clip backbone in case of args.tune_clip

        if True:  # is_main_process():
            # We only need to evaluate the model on the first GPU.
            image_encoder = model.module.image_encoder

            if epoch == 0:
                # Accuracy of the Zero-shot on the test set
                acc = eval_single_dataset(
                    image_encoder,
                    test_dataset,
                    dataset,
                    args,
                    test=True and not args.imagenet_ood,
                )
            elif (
                args.loss_fn not in ["simclr", "ssl_simclr_trusted", "entropy"]
                and epoch > max_ep - 5
            ):  # Eval during last 5 epochs only to save time
                acc = eval_single_dataset(
                    image_encoder, test_dataset, dataset, args, test=False
                )

            string = "Coefficients:\t|"
            if isinstance(coef, list):
                for i in range(len(coef[0])):
                    string += f"`{sum([c[i].mean() for c in coef]):.4f}`|"
            else:
                for c in coef.data:
                    if args.layerwise:
                        string += f"`{c.mean():.4f}`|"
                    else:
                        string += f"`{c:.4f}`|"
            print(string)
            if acc["top1"] > best_acc and epoch > 0:
                best_acc = acc["top1"]
                best_epoch = epoch
                best_coefs = string
                if isinstance(coef, list):
                    coefs = [c.data.detach().cpu() for c in coef]
                else:
                    coefs = coef.data.detach().cpu()
                    if args.attn:
                        coefs1 = coef1.data.detach().cpu()

            if epoch == 0:
                base_acc = acc["top1"]

            if (
                "trusted" in args.loss_fn
                and ((epoch >= 0 and "entro" not in args.loss_fn) or epoch >= 1)
            ) and not (args.semi and epoch >= 1):
                preprocess = data_loader.dataset.update_transforms(
                    image_encoder.val_preprocess
                )
                if args.semi:
                    to_keep, preds = get_n_shots(
                        dataset.train_dataset,
                        args.semi,
                        classification_head.out_features,
                        args,
                    )
                else:
                    acc = eval_single_train_dataset(
                        image_encoder, test_dataset, data_loader, args
                    )

                    confs = acc["conf"]
                    preds = acc["preds"]
                    targets = acc["targets"]
                    full_preds = acc["full_preds"]

                    k = min(
                        int(
                            (len(index_dataset) / classification_head.out_features) / 10
                        ),
                        100,
                    )
                    print(k)
                    to_keep = torch.tensor([], dtype=torch.long)
                    unlabeled = torch.tensor([], dtype=torch.long)
                    for c in range(classification_head.out_features):
                        if False:
                            a = torch.argsort(full_preds[:, c])
                            to_keep = torch.cat((to_keep, a[-k:]))
                            unlabeled = torch.cat((unlabeled, a[:-k]))

                            preds[a[-k:]] = (
                                c  # There are probably overlapping examples here
                            )

                            # print((preds[a[-k:]] == targets[a[-k:]]).sum() / len(a[-k:]))
                            # print((preds[a[:-k]] == targets[a[:-k]]).sum() / len(a[:-k]))
                        else:
                            ids_c = torch.arange(len(preds))[preds == c]
                            a = torch.argsort(confs[ids_c])
                            to_keep = torch.cat((to_keep, ids_c[a[-k:]]))
                            unlabeled = torch.cat((unlabeled, ids_c[a[:-k]]))

                            # print((preds[ids_c[a[-k:]]] == targets[ids_c[a[-k:]]]).sum() / len(ids_c[a[-k:]]))
                            # print((preds[ids_c[a[:-k]]] == targets[ids_c[a[:-k]]]).sum() / len(ids_c[a[:-k]]))

                    unlabeled = torch.tensor(
                        [u for u in unlabeled if confs[u] > 0], dtype=torch.long
                    )
                    unlabeled = torch.unique(unlabeled)

                    print("Correctness")
                    print((preds[to_keep] == targets[to_keep]).sum() / len(to_keep))
                    print(
                        (preds[unlabeled] == targets[unlabeled]).sum() / len(unlabeled)
                    )

                data_loader.dataset.update_transforms(preprocess)

                if "ssl" in args.loss_fn:
                    low_confs = torch.arange(len(data_loader.dataset))[unlabeled]
                    print(
                        f"Got {len(to_keep)} trusted and {len(unlabeled)} untrusted samples"
                    )
                    sampler = TwoStreamBatchSampler(low_confs, to_keep, args.batch_size)
                    data_loader = torch.utils.data.DataLoader(
                        index_dataset,
                        batch_sampler=sampler,
                        num_workers=args.workers,
                        persistent_workers=args.persistant_workers,
                    )
                else:
                    print(f"Got {len(to_keep)} trusted samples")
                    r = len(to_keep) / args.batch_size
                    if r < 10:
                        over_sampling = 10 / r
                        over_sampling = int(over_sampling) + 1
                        print(f"Oversampling {over_sampling} times")
                        to_keep = torch.cat([to_keep] * over_sampling)
                    sampler = torch.utils.data.SubsetRandomSampler(to_keep)
                    data_loader = torch.utils.data.DataLoader(
                        index_dataset,
                        batch_size=args.batch_size,
                        sampler=sampler,
                        num_workers=args.workers,
                        persistent_workers=args.persistant_workers,
                    )
                num_batches = len(data_loader)
                data_loader = fabric.setup_dataloaders(
                    data_loader, use_distributed_sampler=False
                )

        if args.tip_only:
            i = 0
            loss = torch.tensor(0)
            data_time = 0
            batch_time = 0
            continue

        if args.select_tvs and epoch == 0:
            if args.features:
                print("Selecting best Tvs based on feature similarity")
                centroids = torch.cat(
                    [
                        torch.load(
                            os.path.join(f"{args.save}/{dataset}Val/zeroshot_feats.pt")
                        ).unsqueeze(0)
                        for dataset in pool
                        if orig_dataset != dataset
                    ],
                    dim=0,
                )

                for i, batch in enumerate(data_loader):
                    batch = maybe_dictionarize(batch, index=True)
                    inputs = batch["images"]
                    labels = batch["labels"]
                    features = []
                    with torch.no_grad():
                        _, feats = model(inputs, return_features=True)
                        feats /= feats.norm(1, keepdim=True)
                        features.append(feats.cpu())

                features = torch.cat(features, dim=0)
                features = features.mean(0, keepdim=True)
                sims = torch.mm(
                    F.normalize(features, p=2), F.normalize(centroids, p=2).T
                ).mean(
                    dim=0
                )  # Cosine sim
                # sims = torch.mm(features, centroids.T).mean(dim=0)

                model = model.module  # Fabric stuff
                if args.worse:
                    model.image_encoder.update_tvs(
                        [
                            task_vectors[i]
                            for i in torch.argsort(sims)[: args.select_tvs]
                        ]
                    )
                elif args.random:
                    model.image_encoder.update_tvs(
                        [
                            task_vectors[i]
                            for i in torch.randperm(len(task_vectors))[
                                : args.select_tvs
                            ]
                        ]
                    )
                else:
                    model.image_encoder.update_tvs(
                        [
                            task_vectors[i]
                            for i in torch.argsort(sims)[-args.select_tvs :]
                        ]
                    )

                print(
                    f"Highest {args.select_tvs} sims for dataset {test_dataset}: {[pool[i] for i in torch.argsort(sims)[-args.select_tvs:]]}, lowest {[pool[i] for i in torch.argsort(sims)[:args.select_tvs]]}"
                )
            else:
                print("Selecting best Tvs based on gradients")
                if args.mem_eff:
                    print("Memory efficient selection")
                    n_tvs = len(task_vectors)
                    j = 0
                    grad = torch.zeros(n_tvs, len(coef[0]))
                    while j * args.select_tvs < n_tvs:
                        model = model.module  # Fabric stuff
                        model.image_encoder.update_tvs(
                            task_vectors[
                                j * args.select_tvs : (j + 1) * args.select_tvs
                            ]
                        )

                        if args.finetuning_mode == "linear":
                            coef = model.image_encoder.model.coef
                        else:
                            coef = model.image_encoder.coef
                            if args.attn:
                                coef1 = model.image_encoder.coef1

                        params = [p for p in model.parameters() if p.requires_grad]
                        optimizer = torch.optim.AdamW(
                            params, lr=args.lr, weight_decay=args.wd
                        )
                        model, optimizer = fabric.setup(model, optimizer)
                        for i, batch in enumerate(data_loader):
                            batch = maybe_dictionarize(batch, index=True)
                            inputs = batch["images"]
                            labels = batch["labels"]
                            logits = model(inputs)

                            loss = F.cross_entropy(logits, labels)
                            fabric.backward(loss)

                        print(
                            f"Processed {(j+1)*args.select_tvs} out of {n_tvs} task vectors"
                        )
                        grad[j * args.select_tvs : (j + 1) * args.select_tvs] = (
                            torch.abs(coef.grad.cpu())
                        )
                        j += 1
                        optimizer.zero_grad()
                else:
                    for i, batch in enumerate(data_loader):
                        batch = maybe_dictionarize(batch, index=True)
                        inputs = batch["images"]
                        labels = batch["labels"]
                        logits = model(inputs)

                        loss = F.cross_entropy(logits, labels)
                        fabric.backward(loss)

                    grad = torch.abs(coef.grad.cpu())
                    optimizer.zero_grad()

                if args.blockwise_select:
                    print("Blockwise selection of the task vectors")
                    print(grad.shape)
                    selection = torch.argsort(-grad, dim=0)[: args.select_tvs]
                    new_tvs = [{} for _ in range(args.select_tvs)]
                    names = [
                        [pool[s] for s in selection[:, i]]
                        for i in range(len(selection[0]))
                    ]
                    for j, k in enumerate(model.image_encoder.names):
                        for i in range(args.select_tvs):
                            new_tvs[i][k] = task_vectors[selection[i, j]].vector[k]

                    updated_tvs = copy.deepcopy(task_vectors)
                    for i in range(args.select_tvs):
                        updated_tvs[i].vector = new_tvs[i]

                    model = model.module  # Fabric stuff
                    model.image_encoder.update_tvs(updated_tvs[: args.select_tvs])
                else:
                    grad = grad.mean(1)

                    # plt.bar(torch.arange(grad.shape[0]), torch.abs(grad).mean(1))
                    # print(grad)
                    print(
                        f"Highest {args.select_tvs} grads for dataset {test_dataset}: {[pool[i] for i in torch.argsort(grad)[-args.select_tvs:]]}, lowest {[pool[i] for i in torch.argsort(grad)[:args.select_tvs]]}"
                    )
                    # plt.show()

                    model = model.module  # Fabric stuff
                    if args.worse:
                        model.image_encoder.update_tvs(
                            [
                                task_vectors[i]
                                for i in torch.argsort(grad)[: args.select_tvs]
                            ]
                        )
                    elif args.random:
                        model.image_encoder.update_tvs(
                            [
                                task_vectors[i]
                                for i in torch.randperm(len(task_vectors))[
                                    : args.select_tvs
                                ]
                            ]
                        )
                    else:
                        model.image_encoder.update_tvs(
                            [
                                task_vectors[i]
                                for i in torch.argsort(grad)[-args.select_tvs :]
                            ]
                        )

            if args.finetuning_mode == "linear":
                coef = model.image_encoder.model.coef
            else:
                coef = model.image_encoder.coef
                if args.attn:
                    coef1 = model.image_encoder.coef1

            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
            # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-5)
            model, optimizer = fabric.setup(model, optimizer)

        if args.init and epoch == 0:
            print("Initializing based on feature similarity")
            centroids = torch.cat(
                [
                    torch.load(
                        os.path.join(f"{args.save}/{dataset}Val/zeroshot_feats.pt")
                    ).unsqueeze(0)
                    for dataset in pool
                    if orig_dataset != dataset
                ],
                dim=0,
            )

            for i, batch in enumerate(data_loader):
                batch = maybe_dictionarize(batch, index=True)
                inputs = batch["images"]
                labels = batch["labels"]
                features = []
                with torch.no_grad():
                    _, feats = model(inputs, return_features=True)
                    feats /= feats.norm(1, keepdim=True)
                    features.append(feats.cpu())

            features = torch.cat(features, dim=0)
            features = features.mean(0, keepdim=True)
            sims = torch.mm(
                F.normalize(features, p=2), F.normalize(centroids, p=2).T
            ).mean(
                dim=0
            )  # Cosine sim
            sims = F.softmax(sims, dim=0)  # sims / sims.sum()
            # sims = torch.tensor([0.0496, -0.1261, -0.0051, -0.0229, 0.0772, 0.1165, 0.0335, 0.0534, 0.1092, 0.0280, -0.1288, -0.0202, 0.0923, 0.0035, 0.1026, 0.0733, -0.0370, 0.0630, -0.0292, 0.0645, 0.0314])
            print(f"Initialized coefs to {sims}")
            # model.image_encoder.coef.data = (torch.ones(model.module.image_encoder.coef.data.shape) * sims[:, None]).to(model.module.image_encoder.coef.data)
            model.image_encoder.coef.data = (
                (torch.rand(model.module.image_encoder.coef.data.shape) - 0.5) * 2
            ).to(model.module.image_encoder.coef.data)
            if args.attn:
                # model.image_encoder.coef1.data = (torch.ones(model.module.image_encoder.coef1.data.shape) * sims[:, None, None]).to(model.module.image_encoder.coef1.data)
                model.image_encoder.coef1.data = (
                    (torch.rand(model.module.image_encoder.coef1.data.shape) - 0.5) * 2
                ).to(model.module.image_encoder.coef1.data)

        if args.tune_clip:  # or 'RN' in args.model:
            model.train()  # Compute batch norm stats

        for i, batch in enumerate(data_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            if args.softmax_coef:
                model.module.image_encoder.coef.data = F.softmax(
                    model.module.image_encoder.coef.data, dim=0
                )
                if args.attn:
                    model.module.image_encoder.coef1.data = F.softmax(
                        model.module.image_encoder.coef1.data, dim=0
                    )

            batch = maybe_dictionarize(batch, index=True)
            inputs = batch["images"]

            if "simclr" in args.loss_fn:
                inputs2 = batch["images_aug"]

            if "trusted" in args.loss_fn:
                ids = batch["index"]

            data_time = time.time() - start_time

            if "mixup" in args.loss_fn:
                lam = np.random.beta(1, 1)
                lam = max(lam, 1 - lam)
                index = torch.randperm(len(inputs))
                inputs = lam * inputs + (1 - lam) * inputs[index]

            if "entropy_sar" in args.loss_fn:
                _ = model(inputs)

                if i % 10 == 0:
                    print(
                        f"Train Epoch: {epoch} [{i + 1}/{len(data_loader)}]\t",
                        flush=True,
                    )

                continue

            elif "ssl" in args.loss_fn and "entro" not in args.loss_fn:
                with torch.no_grad():
                    logits = model(inputs)
            elif "simclr" in args.loss_fn:
                logits, feats = model(inputs, return_features=True)
            else:
                logits = model(inputs)

            if args.loss_fn == "cross_entropy":
                labels = batch["labels"]
                loss = loss_fn(logits, labels)
            elif "entro_conf" in args.loss_fn:
                loss = loss_fn(logits).mean()
            elif "ssl_simclr_trusted" in args.loss_fn:
                lam = np.random.beta(1, 1)
                lam = max(lam, 1 - lam)
                index = torch.randperm(len(inputs2))
                # inputs2 = lam * inputs2 + (1-lam) * inputs2[index]
                logits2 = model(inputs2)
                trusted = torch.ones(args.batch_size).bool()
                trusted[: 3 * args.batch_size // 4] = (
                    0  # First half of the batch are untrusted labels
                )
                loss = loss_fn(
                    logits,
                    logits2,
                    F.one_hot(
                        preds.to(logits.device)[ids],
                        num_classes=classification_head.out_features,
                    ).float(),
                    trusted,
                    thresh=threshs[epoch],
                )  # , lam, index)

            elif "trusted" in args.loss_fn:
                if "mixup" in args.loss_fn:
                    loss = lam * loss_fn(
                        logits,
                        F.one_hot(
                            preds.to(logits.device)[ids],
                            num_classes=classification_head.out_features,
                        ).float(),
                    ) + (1 - lam) * loss_fn(
                        logits,
                        F.one_hot(
                            preds.to(logits.device)[ids][index],
                            num_classes=classification_head.out_features,
                        ).float(),
                    )
                else:
                    loss = loss_fn(
                        logits,
                        F.one_hot(
                            preds.to(logits.device)[ids],
                            num_classes=classification_head.out_features,
                        ).float(),
                    )
            elif "ssl" in args.loss_fn:
                lam = np.random.beta(1, 1)
                lam = max(lam, 1 - lam)
                index = torch.randperm(len(inputs2))
                # inputs2 = lam * inputs2 + (1-lam) * inputs2[index]
                logits2 = model(inputs2)
                loss = loss_fn(logits, logits2, preds)  # , lam, index)
            elif "simclr" in args.loss_fn:
                logits2, feats2 = model(inputs2, return_features=True)
                if "entro" in args.loss_fn:
                    loss = loss_fn(logits, logits2, feats, feats2).mean()
                elif "mixup" in args.loss_fn:
                    loss = loss_fn(logits, logits2, lam, index).mean()
                else:
                    loss = loss_fn(feats, feats2).mean()
            else:
                loss = loss_fn(logits).mean(0)

            if args.l1:
                loss += l1_reg(model.image_encoder.coef)

            fabric.backward(loss)

            if (i + 1) % args.num_grad_accumulation == 0:
                if not args.lr_scheduler == "annealing":
                    scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % args.print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and True  # is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{len(data_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\tLr {optimizer.param_groups[0]['lr']:.3f}",  # noqa: E501
                    flush=True,
                )

    percent_complete = 100 * (i + 1) / len(data_loader)
    print(
        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{len(dataset.test_loader)}]\t"  # noqa: E501
        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
        flush=True,
    )

    if True:  # is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = model.module.image_encoder
        # with torch.autocast(device_type="cuda"):
        if args.loss_fn not in ["simclr", "ssl_simclr_trusted", "entropy"]:
            # No early stopping for test-time adaptation
            acc = eval_single_dataset(image_encoder, test_dataset, dataset, args)

        string = "Coefficients:\t|"
        if isinstance(coef, list):
            for i in range(len(coef[0])):
                string += f"`{sum([c[i].mean() for c in coef]):.4f}`|"
        else:
            for c in coef.data:
                if args.layerwise:
                    string += f"`{c.mean():.4f}`|"
                else:
                    string += f"`{c:.4f}`|"
        print(string)

        if acc["top1"] > best_acc:
            best_acc = acc["top1"]
            best_epoch = epoch
            best_coefs = string
            if isinstance(coef, list):
                coefs = [c.data.detach().cpu() for c in coef]
            else:
                coefs = coef.data.detach().cpu()
                if args.attn:
                    coefs1 = coef1.data.detach().cpu()

        if epoch == 0:
            best_acc = 0
        if args.tip_ft:
            fabric = Fabric(
                accelerator="cuda", devices=1, strategy="ddp", precision="32"
            )
            fabric.launch()
            args.fabric = fabric
            model = fabric.setup_module(model.module)

            image_encoder = model.module.image_encoder
            with torch.no_grad():
                if args.loss_fn == "cross_entropy":
                    to_keep_u = torch.arange(len(index_dataset))
                    args.semi = len(to_keep_u) / classification_head.weight.shape[0]
                    sampler = torch.utils.data.SubsetRandomSampler(to_keep_u)
                else:
                    to_keep_u = torch.unique(to_keep)
                    to_keep_u.sort()

                subset_t = SubsetSampler(to_keep_u)
                data_loader_t = torch.utils.data.DataLoader(
                    index_dataset,
                    batch_size=args.batch_size,
                    sampler=subset_t,
                    num_workers=args.workers,
                    persistent_workers=args.persistant_workers,
                )

                aug_n = 1
                tbar = tqdm(range(aug_n), disable=args.no_tqdm)
                tbar.set_description("Generating cache")
                labels = []
                all_features, features = [], []
                for j in tbar:
                    for i, batch in enumerate(data_loader_t):
                        batch = maybe_dictionarize(batch, index=True)
                        inputs = batch["images"]
                        logits, feats = model(inputs, return_features=True)
                        features.append(feats.cpu())
                        if j == 0:
                            labels.append(batch["labels"])
                    all_features.append(torch.cat(features, dim=0).unsqueeze(0))
                    features = []

            features_cache = torch.cat(all_features, dim=0).mean(0)
            features_cache /= features_cache.norm(dim=-1, keepdim=True)
            labels = torch.cat(labels, dim=0)
            labels_cache = F.one_hot(labels.long())

            if args.lp:
                from lpplusplus import init_lp

                adapter, alpha_vec, lr_alpha, lr_temp = init_lp(
                    features_cache,
                    labels.long(),
                    classification_head.weight.T / 100.0,
                    args.semi,
                )
            else:
                features_cache = features_cache.permute(1, 0)
                adapter = nn.Linear(
                    features_cache.shape[0], features_cache.shape[1], bias=False
                )
                adapter.weight = nn.Parameter(features_cache.t())
                adapter.register_parameter(
                    "beta_alpha", nn.Parameter(torch.tensor([1.0, 2.0]))
                )  # Only way to update the parameter on the GPU, otherwise update on the CPU. In any case fp16 is non trivial
                print(sum(p.numel() for p in adapter.parameters() if p.requires_grad))

            adapter = adapter.to(features_cache)

            best_adapter = copy.deepcopy(
                adapter
            ).cpu()  # Not sure if .clone() is necessary here
            if args.lp:
                best_alpha = alpha_vec.cpu().clone()

            if "ssl" in args.loss_fn:
                data_loader = torch.utils.data.DataLoader(
                    index_dataset,
                    batch_sampler=sampler,
                    num_workers=args.workers,
                    persistent_workers=args.persistant_workers,
                )
            else:
                data_loader = torch.utils.data.DataLoader(
                    index_dataset,
                    sampler=sampler,
                    num_workers=args.workers,
                    batch_size=args.batch_size,
                    persistent_workers=args.persistant_workers,
                )

            if args.lp:
                if args.tip_cot:
                    optimizer = torch.optim.SGD(
                        [{"params": params}, {"params": adapter.parameters()}],
                        lr_temp,
                        momentum=0.9,
                    )
                else:
                    optimizer = torch.optim.SGD(
                        adapter.parameters(), lr_temp, momentum=0.9
                    )
            elif args.tip_cot:
                optimizer = torch.optim.AdamW(
                    [
                        {"params": params},
                        {"params": adapter.weight},
                        {"params": adapter.beta_alpha},
                    ],
                    lr=args.lr,
                    eps=1e-4,
                )  # Optimize tip + coefs
            else:
                optimizer = torch.optim.AdamW(
                    [{"params": adapter.weight}, {"params": adapter.beta_alpha}],
                    lr=0.001,
                    eps=1e-4,
                )
                # optimizer = torch.optim.AdamW([adapter.beta_alpha], lr=0.001, eps=1e-4)

            adapter, optimizer = fabric.setup(adapter, optimizer)
            data_loader = fabric.setup_dataloaders(
                data_loader, use_distributed_sampler=False
            )

            if not args.lp:
                adapter.eval()
                acc = eval_single_dataset(
                    image_encoder,
                    test_dataset,
                    dataset,
                    args,
                    adapter=adapter,
                    beta_alpha=adapter.beta_alpha,
                    labels_cache=labels_cache,
                )

                if acc["top1"] > best_acc:
                    best_acc = acc["top1"]
                    best_coefs = string
                    best_epoch = epoch
                    if isinstance(coef, list):
                        coefs = [c.data.detach().cpu() for c in coef]
                    else:
                        coefs = coef.data.detach().cpu()

                    beta_alpha = adapter.beta_alpha
                    if args.attn:
                        coefs1 = coef1.data.detach().cpu()

            model.eval()
            if not args.tip_cot and not args.ours:
                args.epochs = 1
            elif args.tip_cot and not args.loss_fn == "cross_entropy":
                args.epochs *= 2

            if args.lp:
                lrs = cosine_annealing_lr(lr_temp, 0, args.epochs)
            else:
                lrs = cosine_annealing_lr(0.001, 0, args.epochs)
            feats_, logits_, labels_ = [], [], []
            for epoch in range(args.epochs):
                adapter.train()
                if not args.lp:
                    if args.tip_cot:
                        if args.add_random_tv:
                            adjust_lr(optimizer, lrs[epoch], [0.1, 1.0, 100.0])
                        else:
                            adjust_lr(optimizer, lrs[epoch], [1.0, 0.01, 1.0])
                    else:
                        adjust_lr(optimizer, lrs[epoch], [1.0, 100.0])
                else:
                    if args.tip_cot:
                        adjust_lr_lp(optimizer, lrs[epoch], 0.1)

                tbar = tqdm(data_loader, disable=args.no_tqdm)
                tbar.set_description(
                    f"Learning adaptor, epoch {epoch}/{args.epochs}, lr {optimizer.param_groups[0]['lr']:.3f}"
                )
                for i, batch in enumerate(tbar):
                    batch = maybe_dictionarize(batch, index=True)
                    inputs = batch["images"]

                    logits, feats = model(inputs, return_features=True)
                    feats = feats / feats.norm(dim=-1, keepdim=True)

                    if not args.tip_cot:
                        feats_.append(feats.detach())
                        logits_.append(logits.detach())
                        labels_.append(batch["labels"])

                    if args.lp:
                        vision_logits = adapter(feats)
                        text_logits = logits / 100.0
                        logits = (
                            vision_logits
                            + torch.ones(feats.shape[0], 1).to(feats)
                            @ alpha_vec.to(feats)
                            * text_logits
                        )
                    else:
                        affinity = adapter(feats)
                        cache_logits = (
                            (-1)
                            * (adapter.beta_alpha[0] - adapter.beta_alpha[0] * affinity)
                        ).exp() @ labels_cache.to(affinity)
                        tv_logits = logits
                        logits = cache_logits * adapter.beta_alpha[1] + tv_logits

                    loss = F.cross_entropy(logits, batch["labels"])

                    if not args.lp:
                        tbar.set_description(
                            f"Learning adaptor, epoch {epoch}/{args.epochs}, lr {optimizer.param_groups[0]['lr']:.3f}, loss {loss.item():.3f}, beta_alpha {adapter.beta_alpha[0]:.2f} {adapter.beta_alpha[1]:.2f}"
                        )
                    else:
                        tbar.set_description(
                            f"Learning adaptor, epoch {epoch}/{args.epochs}, lr {optimizer.param_groups[0]['lr']:.3f}, loss {loss.item():.3f}"
                        )

                    if args.softmax_coef and args.tip_cot:
                        model.module.image_encoder.coef = F.softmax(
                            model.module.image_encoder.coef, dim=0
                        )
                        if args.attn:
                            model.module.image_encoder.coef1 = F.softmax(
                                model.module.image_encoder.coef, dim=0
                            )

                    fabric.backward(loss)
                    # loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if (epoch + 1) % 10 == 0:
                        if args.lp:
                            alpha_vec.data -= lr_alpha * alpha_vec.grad.data

                if not args.tip_cot and not args.ours:
                    features = torch.cat(feats_, dim=0)
                    logits = torch.cat(logits_, dim=0)
                    labels = torch.cat(labels_, dim=0)
                    del feats_, logits_, labels_
                    tbar = tqdm(range(1, 300), disable=args.no_tqdm)
                    tbar.set_description(f"Tuning {'LP++' if args.lp else 'TIP'}")
                    lrs = cosine_annealing_lr(0.001, 0, 300)
                    for e in tbar:
                        adapter.train()

                        if args.lp:
                            vision_logits = adapter(features)
                            text_logits = logits / 100.0
                            f_logits = (
                                vision_logits
                                + torch.ones(features.shape[0], 1).to(features)
                                @ alpha_vec.to(features)
                                * text_logits
                            )
                        elif args.tip_ft:
                            affinity = adapter(features)
                            cache_logits = (
                                (-1)
                                * (
                                    adapter.beta_alpha[0]
                                    - adapter.beta_alpha[0] * affinity
                                )
                            ).exp() @ labels_cache.to(affinity)
                            tv_logits = logits
                            f_logits = cache_logits * adapter.beta_alpha[1] + tv_logits

                        loss = F.cross_entropy(f_logits, labels)

                        fabric.backward(loss)
                        # loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        if (e + 1) % 10 == 0:
                            tbar.set_description(
                                f"Tuning {'LP++' if args.lp else 'TIP'}, epoch {e}, lr {optimizer.param_groups[0]['lr']:.3f}, best val acc {best_acc:.2f}, loss {loss.item():.2f}"
                            )
                            if args.lp:
                                alpha_vec.data -= lr_alpha * alpha_vec.grad.data

                        if args.tip_ft and not args.lp:
                            adjust_lr(optimizer, lrs[e], [1.0, 100.0])

                        adapter.eval()
                        with torch.no_grad():
                            if e == 1:
                                val_features, val_logits, val_labels = get_val_features(
                                    image_encoder, test_dataset, dataset, args
                                )
                                val_features /= val_features.norm(dim=-1, keepdim=True)

                            if args.lp:
                                vision_logits_val = adapter(val_features)
                                text_logits_val = val_logits / 100.0
                                logits_val = (
                                    vision_logits_val
                                    + torch.ones(val_features.shape[0], 1).to(
                                        val_features
                                    )
                                    @ alpha_vec.to(val_features)
                                    * text_logits_val
                                )
                            elif args.tip_ft:
                                affinity = adapter(val_features)
                                cache_logits = (
                                    (-1)
                                    * (
                                        adapter.beta_alpha[0]
                                        - adapter.beta_alpha[0] * affinity
                                    )
                                ).exp() @ labels_cache.to(affinity)
                                tv_logits = val_logits
                                logits_val = (
                                    cache_logits * adapter.beta_alpha[1] + tv_logits
                                )

                            acc_val = np.mean(
                                logits_val.argmax(dim=1).cpu().numpy()
                                == val_labels.cpu().numpy()
                            )
                            if acc_val > best_acc:
                                best_acc = acc_val
                                best_epoch = epoch
                                best_coefs = string
                                best_adapter = copy.deepcopy(adapter).cpu()
                                if args.lp:
                                    best_alpha = alpha_vec.cpu().clone()
                                if isinstance(coef, list):
                                    coefs = [c.data.detach().cpu() for c in coef]
                                else:
                                    coefs = coef.data.detach().cpu()
                                if args.attn:
                                    coefs1 = coef1.data.detach().cpu()

                else:
                    adapter.eval()

                    if (
                        epoch > args.epochs - 5
                    ):  # Eval during last 5 epochs only to save time
                        if args.lp:
                            acc = eval_single_dataset(
                                image_encoder,
                                test_dataset,
                                dataset,
                                args,
                                adapter=adapter,
                                alpha_vec=alpha_vec,
                            )
                        else:
                            acc = eval_single_dataset(
                                image_encoder,
                                test_dataset,
                                dataset,
                                args,
                                adapter=adapter,
                                beta_alpha=adapter.beta_alpha,
                                labels_cache=labels_cache,
                            )

                    if acc["top1"] > best_acc:
                        best_acc = acc["top1"]
                        best_coefs = string
                        best_epoch = epoch
                        if isinstance(coef, list):
                            coefs = [c.data.detach().cpu() for c in coef]
                        else:
                            coefs = coef.data.detach().cpu()
                        if not args.lp:
                            beta_alpha = adapter.beta_alpha
                        if args.attn:
                            coefs1 = coef1.data.detach().cpu()

                        best_adapter = copy.deepcopy(
                            adapter
                        ).cpu()  # Not sure if .clone() is necessary here
                        if args.lp:
                            best_alpha = alpha_vec.cpu().clone()

    # Load best coefs
    if isinstance(model.image_encoder.coef, list):
        for i in range(len(model.image_encoder.coef)):
            model.image_encoder.coef[i].data = coefs[i].to(
                model.image_encoder.coef[i].data
            )
    elif args.finetuning_mode == "linear":
        model.image_encoder.model.coef.data = coefs.to(
            model.image_encoder.model.coef.data
        )
        if args.attn:
            model.image_encoder.model.coef1.data = coefs1.to(
                model.image_encoder.model.coef.data
            )

    else:
        model.image_encoder.coef.data = coefs.to(model.image_encoder.coef.data)
        if args.attn:
            model.image_encoder.coef1.data = coefs1.to(model.image_encoder.coef.data)

    if args.tip_ft:
        if args.lp:
            adapter = best_adapter
            alpha_vec = best_alpha
            acc = eval_single_dataset(
                image_encoder,
                test_dataset.split("Val")[0],
                dataset,
                args,
                adapter=adapter,
                alpha_vec=alpha_vec,
                test=True,
            )
            acc["alpha_vec"] = best_alpha
        else:
            adapter = best_adapter
            acc = eval_single_dataset(
                image_encoder,
                test_dataset.split("Val")[0],
                dataset,
                args,
                adapter=adapter,
                beta_alpha=adapter.beta_alpha,
                labels_cache=labels_cache,
                test=True,
            )
            acc["beta_alpha"] = adapter.beta_alpha
        acc["adapter"] = best_adapter.state_dict()
    else:
        acc = eval_single_dataset(
            image_encoder, test_dataset.split("Val")[0], dataset, args, test=True
        )

    acc["coefs"] = coefs
    if args.attn:
        acc["coefs"] = (coefs, coefs1)

    best_acc = acc["top1"]
    if args.epochs == 0:
        base_acc = acc["top1"]

    if not args.tip_ft and not args.tip_cot or args.lp:
        beta_alpha = (0, 0)
    else:
        beta_alpha = adapter.beta_alpha.detach().cpu()

    acc["coefs"] = coefs
    if args.attn:
        acc["coefs"] = (coefs, coefs1)

    acc["beta_alpha"] = beta_alpha

    return base_acc, best_acc, best_epoch, best_coefs, acc


if __name__ == "__main__":
    import cProfile
    import sys

    # if check avoids hackery when not profiling
    # Optional; hackery *seems* to work fine even when not profiling, it's just wasteful
    if sys.modules["__main__"].__file__ == cProfile.__file__:
        import learn_coef  # Imports you again (does *not* use cache or execute as __main__)

        globals().update(
            vars(learn_coef)
        )  # Replaces current contents with newly imported stuff
        sys.modules["__main__"] = (
            learn_coef  # Ensures pickle lookups on __main__ find matching version
        )

    # Epochs w/ lr=1e-2 for entropy objective

    args = parse_arguments()

    if args.loss_fn == "cross_entropy":
        datasets = {
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
            "Caltech101": 10,
        }
    else:
        epochs = (
            args.epochs
        )  # Go up to 30 for ResNets, row-wise, col-wise and random-wise
        datasets = {
            "Cars": epochs,
            "DTD": epochs,
            "EuroSAT": epochs,
            "GTSRB": epochs,
            "MNIST": epochs,
            "RESISC45": epochs,
            "SUN397": epochs,
            "SVHN": epochs,
            "CIFAR10": epochs,
            "CIFAR100": epochs,
            "ImageNet": epochs,
            "STL10": epochs,
            "Food101": epochs,
            "Caltech256": epochs,
            "FGVCAircraft": epochs,
            "Flowers102": epochs,
            "OxfordIIITPet": epochs,
            "CUB200": epochs,
            "PascalVOC": epochs,
            "Country211": epochs,
            "Caltech101": epochs,
            "UCF101": epochs,
        }

    if args.datasets is not None:
        datasets = {k: datasets[k] for k in args.datasets}

    # HACK: Some command line arguments are overwritten by defaults here.
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 32 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 4 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    if (
        "RN" in args.model
    ):  # RN models do not have attention layers. Leaving attn as True will cause a problem with coef1 receiving no gradients.
        args.attn = False

    if args.lora:
        args.attn = False

    if not args.layerwise:
        args.attn = False

    if args.seed is not None and False:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

    if args.merge:
        assert args.fname is not None, "--fname needs to be specified"
        if os.path.isfile(os.path.join(args.fname, str(args.seed), "results.txt")):
            print(
                f"Merging with {os.path.join(args.fname, str(args.seed), 'results.txt')}"
            )
            datasets_n = extract_datasets(
                os.path.join(args.fname, str(args.seed), "results.txt")
            )
            datasets = {k: v for k, v in datasets.items() if k not in datasets_n}

    args.datasets = datasets
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
    seed = int(torch.exp(torch.tensor(args.seed)) * 3.1415 * 1000)
    seed_everything(seed)
    main(0, args)
