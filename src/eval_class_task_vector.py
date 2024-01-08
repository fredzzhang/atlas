"""
Evaluation of task vectors trained on a single class

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import json
import numpy as np

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

def main(args, eval_datasets):
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

    for dataset in eval_datasets:
        print("*" * 100)
        print(f"Evaluating on {dataset}")

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

        if args.finetuning_mode not in ["standard", "linear"]:
            raise ValueError("Only standard and linear finetuning modes are supported.")

        info = {}
        full_dataset = dataset.split('_')[-1]
        for s_c in np.linspace(0.0, 1.0, args.n_eval_points):
            # Evaluate
            print("=" * 100)
            print(f"Evaluating on val split with coefficient {s_c:.2f}")
            eval_dataset = f"{dataset}Val"
            full_eval_dataset = f"{full_dataset}Val"
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=s_c)

            acc = eval_single_dataset(image_encoder, eval_dataset, args)["top1"]
            full_acc = eval_single_dataset(image_encoder, full_eval_dataset, args)["top1"]
            info[s_c] = {'acc': acc, 'full_acc': full_acc}

        best = 0
        best_coef = 1
        for k in info:
            if info[k]['full_acc'] > best:
                best = info[k]['full_acc']
                best_coef = k

        print("=" * 100)
        print(f"The best coefficient for dataset {dataset} is {best_coef}.")
        print("Evaluating on test split.")
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=best_coef)
        test_acc = eval_single_dataset(image_encoder, dataset, args)["top1"]
        full_test_acc = eval_single_dataset(image_encoder, full_dataset, args)["top1"]
        info['test'] = {'acc': test_acc, 'full_acc': full_test_acc, 'coef': best_coef}
        accuracies[dataset] = info

    # Save results
    if args.finetuning_mode == "standard":
        save_path = f"{args.save}/class_ft_accuracies.json"
    elif args.finetuning_mode == "linear":
        save_path = f"{args.save}/class_linear_ft_accuracies.json"

    with open(save_path, "w") as f:
        json.dump(accuracies, f)

if __name__ == '__main__':

    args = parse_arguments()
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

    eval_datasets = [
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

    main(args, eval_datasets)