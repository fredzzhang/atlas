"""Evaluation on a task

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al. and Guillermo Ortiz-Jimenez et al.,
at https://github.com/mlfoundations/task_vectors and
https://github.com/gortizji/tangent_task_arithmetic
"""

import json

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

accuracies = {}


print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")

for dataset in [
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
    "Caltech101",
    "Caltech256",
    "FGVCAircraft",
    "Flowers102",
    "OxfordIIITPet",
    "CUB200",
    "PascalVOC",
    "Country211",
    "UCF101",
]:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    # NOTE The original code might have a bug here. This part has been changed because
    # the linearised model only has the state_dict saved and thus has a different data type.

    # pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    pretrained_checkpoint = (
        f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        if args.finetuning_mode == "linear"
        else f"{args.save}/{dataset}Val/zeroshot.pt"
    )

    finetuned_checkpoint = (
        f"{args.save}/{dataset}Val/linear_finetuned.pt"
        if args.finetuning_mode == "linear"
        else f"{args.save}/{dataset}Val/finetuned.pt"
    )

    try:
        task_vector = (
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            if args.finetuning_mode == "linear"
            else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue

    if args.finetuning_mode == "none":
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    elif args.finetuning_mode == "standard" or args.finetuning_mode == "linear":
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    elif args.finetuning_mode == "posthoc":
        zs_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        ft_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        image_encoder = LinearizedImageEncoder(
            init_encoder=zs_encoder, image_encoder=ft_encoder, args=args
        )

    # Evaluate on the entire dataset
    if args.eval_on_full and '_' in dataset:
        _, dataset = dataset.split('_')
    for split in ["test", "val"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            image_encoder, eval_dataset, args
        )["top1"]

# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_accuracies.json"
elif args.finetuning_mode == "standard":
    save_path = f"{args.save}/ft_accuracies.json"
elif args.finetuning_mode == "linear":
    save_path = f"{args.save}/linear_ft_accuracies.json"
elif args.finetuning_mode == "posthoc":
    save_path = f"{args.save}/posthoc_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
