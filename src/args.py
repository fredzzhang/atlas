"""
Argument list

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al. and Guillermo Ortiz-Jimenez et al.,
at https://github.com/mlfoundations/task_vectors and
https://github.com/gortizji/tangent_task_arithmetic
"""

import argparse
import os

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/Documents/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--eval-on-full",
        default=False,
        action="store_true",
        help="Evaluate on the full dataset, when the model is trained on one class."
    )
    parser.add_argument(
        "--loss-fn",
        default='entropy',
        type=str,
        help="Loss function to use.",
        choices=["entropy", "cross_entropy", "simclr", "simclr_entro", "barlow_simclr", 'simclr_mixup', "cb_entropy", "ssl_simclr", "trusted", "trusted_mixup", "entro_trusted", "ssl_simclr_trusted", "ssl_simclr_trusted_entro", "entropy_sar"]
    )
    parser.add_argument(
        "--lr-scheduler",
        default='',
        type=str,
        help="Loss function to use.",
        choices=["", "annealing"]
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank."
    )
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",  # noqa: E501
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=-1,
        help="How often to checkpoint the model.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12355,
        help="Port for distributed training.",
    )
    parser.add_argument(
        "--finetuning-mode",
        choices=["standard", "linear", "posthoc", "none"],
        help="Whether to use linearized models or not.",
    )
    parser.add_argument(
        "--n-eval-points",
        type=int,
        default=21,
        help="Number of evaluation points used to find optimal coefficient in task arithmetic.",
    )
    parser.add_argument(
        "--layerwise",
        default=False,
        action='store_true',
        help="Compute layerwise parameters."
    )
    parser.add_argument(
        "--trainset",
        default=False,
        action='store_true',
        help="Compute coefs on a subset of the trainset."
    )
    parser.add_argument(
        "--l1",
        default=False,
        action='store_true',
        help="L1 reg"
    )
    parser.add_argument(
        "--lp",
        default=False,
        action='store_true',
        help="Learn LP++ classifier"
    )
    parser.add_argument(
        "--scale",
        default=False,
        action='store_true',
        help="Scale the task vectors"
    )
    parser.add_argument(
        "--attn",
        default=False,
        action='store_true',
        help="Batch norm or LayerNorm tuning"
    )

    parser.add_argument(
        "--datasets",
        default=None,
        nargs='+',
        help="Subset of datasets to train on."
    )
    parser.add_argument(
        "--tip-ft",
        default=False,
        action='store_true',
        help="Finetune the task vector using TIP adaptors"
    )
    parser.add_argument(
        "--ours",
        default=False,
        action="store_true",
        help="TIP/LP++ with multiple forward passes."
    )
    parser.add_argument(
        "--tip-cot",
        default=False,
        action='store_true',
        help="Co-train the task vector with TIP adaptors"
    )
    parser.add_argument(
        "--tip-only",
        default=False,
        action='store_true',
        help="TIP adaptors ours with learned alpha/beta, must be used together with --tip-ft"
    )
    parser.add_argument(
        "--no-log",
        default=False,
        action='store_true',
        help="No logging in the results/ folder"
    )
    parser.add_argument(
        "--select-tvs",
        default=None,
        type=int,
        help="Select the best n Tvs from the pool"
    )
    parser.add_argument(
        "--features",
        default=False,
        action="store_true",
        help="Select TVs based on centroid similarity."
    )
    parser.add_argument(
        "--worse",
        default=False,
        action="store_true",
        help="Select the worse n Tvs from the pool. Needs to be coupled with --select-tvs"
    )
    parser.add_argument(
        "--random",
        default=False,
        action="store_true",
        help="Select random n Tvs from the pool. Needs to be coupled with --select-tvs"
    )
    parser.add_argument(
        "--mem-eff",
        default=False,
        action="store_true",
        help="Select random n Tvs from the pool in a memory efficient way. Needs to be coupled with --select-tvs"
    )
    parser.add_argument(
        "--tv-cpu",
        default=False,
        action='store_true',
        help="Host tvs to cpu (slower but much less VRAM)"
    )
    parser.add_argument(
        "--hugg",
        default=False,
        action='store_true',
        help="Pull 19 weights from hugginface to be used as task vectors (implemented for VIT-B-32 only)"
    )
    parser.add_argument(
        "--add-random-tv",
        type=int,
        default=None,
        help="Add the specified amount random task vectors on top of the existing ones."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed."
    )
    parser.add_argument(
        "--semi",
        type=int,
        default=None,
        help="N of few-shot samples per class"
    )
    parser.add_argument(
        "--optimally-random",
        type=str,
        default=None,
        help="Path to optiized random task vectors.",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=None,
        help="Save file name",
    )
    parser.add_argument(
        "--fabric",
        type=str,
        default=None,
        help="Placeholder for eval functions.",
    )
    parser.add_argument(
        "--lora",
        default=False,
        action="store_true",
        help="Use LoRA task vectors."
    )
    parser.add_argument(
        "--no-tqdm",
        default=False,
        action="store_true",
        help="Decativate tqdm logging on stderr."
    )
    parser.add_argument(
        "--blockwise-select",
        default=False,
        action="store_true",
        help="Select the task vector budget at the layer level."
    )
    parser.add_argument(
        "--tune-clip",
        default=False,
        action="store_true",
        help="Tune the clip backbone together with the TVs."
    )
    parser.add_argument(
        "--tune-lora",
        default=False,
        action="store_true",
        help="Tune a lora together with the TVs."
    )
    parser.add_argument(
        "--merge",
        default=False,
        action='store_true',
        help="Will run on the missing datasets from the results.txt specified through fname and complete the specified results.txt file. Needs fname to be specified."
    )
    parser.add_argument(
        "--imagenet-ood",
        default=False,
        action='store_true',
        help="Evaluates ImageNet weights on OOD versions as well (test only)."
    )
    parser.add_argument(
        "--softmax-coef",
        default=False,
        action='store_true',
        help="Softmaxes the learned coefs."
    )
    parser.add_argument(
        "--init",
        default=False,
        action='store_true',
        help="Init coefs with feature sims."
    )
    parser.add_argument(
        "--mlp-only",
        default=False,
        action='store_true',
        help="LoRA on mlp layers only."
    )
    parser.add_argument(
        "--attn-only",
        default=False,
        action='store_true',
        help="LoRA on attn layers only."
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Train on a percentage of the data."
    )
    
    parser.add_argument(
        "--pool-one",
        default=False,
        action='store_true',
        help="Sequentially learn the coefs for each available TV."
    )
    
    parser.add_argument(
        "--row-wise",
        default=False,
        action='store_true',
        help="1 Coef per row"
    )
    parser.add_argument(
        "--col-wise",
        default=False,
        action='store_true',
        help="1 Coef per col"
    )
    parser.add_argument(
        "--persistant-workers",
        default=False,
        action='store_true',
        help="Persistant dataloader workers for concurrent running"
    )
    parser.add_argument(
        "--addition",
        default=False,
        action='store_true',
        help="Task addition"
    )
    parser.add_argument(
        "--full-addition",
        default=False,
        action='store_true',
        help="Task addition on 22 datasets."
    )
    parser.add_argument(
        "--negation",
        default=False,
        action='store_true',
        help="Task negation"
    )
    parser.add_argument(
        "--full-negation",
        default=False,
        action='store_true',
        help="Task negation on 22 datasets."
    )
    parser.add_argument(
        "--random-wise",
        default=None,
        type=int,
        help="Number of parameter per layer (randomly assigned)."
    )
    parser.add_argument(
        "--clip-ft",
        default=False,
        action='store_true',
        help="FT clip (test time adapt)"
    )
    parser.add_argument("--neg-coef", type=float, default=2., help="Task negation coefficient")
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
