"""
Evaluation on the orthogonality of task vectors and datasets

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from src.args import parse_arguments
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

def main(args, eval_datasets):

    print("*" * 100)
    if args.finetuning_mode == "standard":
        print("Evaluating non-linear FT models.")
    elif args.finetuning_mode == "linear":
        print("Evaluating linear FT models.")
    elif args.finetuning_mode == "posthoc":
        print("Evaluating post-hoc linearized models.")
    else:
        raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
    print("*" * 100)

    task_vectors = []

    for dataset in eval_datasets:
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors.append(
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors.append(
                NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )

    n = len(task_vectors)
    cos_sim = np.zeros((n, n))
    norm = [x.dot(x).sqrt().item() for x in task_vectors]
    # Compute the cosine similarity between task vectors
    for i in range(n):
        for j in range(i, n):
            s = task_vectors[i].dot(task_vectors[j]) / (norm[i] * norm[j])
            s = abs(s)
            cos_sim[i, j] = s
            if i != j:
                cos_sim[j, i] = s
    # Plot the similarity
    _, ax = plt.subplots()
    ax.imshow(cos_sim)
    plt.title("Task vector cosine similarity")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cos_sim[i, j]:.2f}', color='w', ha='center', va='center')

    if args.finetuning_mode == "standard":
        save_file = f"{args.save}/task_vector_sim"
    elif args.finetuning_mode == "linear":
        save_file = f"{args.save}/linear_task_vector_sim"
    elif args.finetuning_mode == "posthoc":
        save_file = f"{args.save}/posthoc_task_vector_sim"
    plt.savefig(f"{save_file}.png")
    with open(f"{save_file}.json", 'w') as f:
        json.dump({"cosine_similarity": cos_sim.tolist()}, f)

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