"""Learn coefficients on a task vector for task negation,
with supervised objective on a combination of the target
dataset and the control dataset.

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""
import os
import json
import torch

from utils import find_optimal_coef

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.heads import get_classification_head
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.modeling import ImageEncoder, MultiHeadImageClassifier
from src.linearize import LinearizedImageEncoder
from src.composition import WeightedImageEncoder, WeightedLinearizedModel

args = parse_arguments()


if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

eval_datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies.json")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
    ft_accuracies_path = os.path.join(args.save, "posthoc_ft_accuracies.json")
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

control_dataset = "ImageNet"
negation_accuracies = {}

for dataset in eval_datasets:
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
        task_vector = -LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vector = -NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    # We use the validation set to choose the optimal coefficient.
    args.eval_datasets = [dataset + "Val"]
    args.control_dataset = control_dataset + "Val"
    val_metrics = evaluate_task_vector(
        task_vector,
        pretrained_checkpoint,
        args,
        posthoc_linearization=args.finetuning_mode == "posthoc",
    )

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric=f"{dataset}Val:top1",
        minimize=True,
        control_metric=f"{control_dataset}Val:top1",
        control_metric_threshold=args.control_threshold
        * pretrained_accuracies[control_dataset + "Val"],
    )

    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        optimal_coef,
        posthoc_linearization=args.finetuning_mode == "posthoc",
    )

    print("=" * 100)
    print(f"Test accuracy: {test_metrics[f'{dataset}:top1']}")

    negation_accuracies[dataset] = {
        "test": test_metrics[f"{dataset}:top1"],
        "test_control": test_metrics[f"{control_dataset}:top1"],
        "val": val_metrics,
    }

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/negations.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_negations.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_negations.json"

with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)

def main(rank, args):

    setup_ddp(args.rank, args.world_size, port=args.port)

    tgt_dataset = args.tgt_dataset
    if args.partition == "trainval":
        tgt_dataset += "Val"
    ctr_dataset = args.ctr_dataset

    ckpdir = os.path.join(args.save, tgt_dataset)
    os.makedirs(ckpdir, exist_ok=True)

    assert args.finetuning_mode in [
        "linear", "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    ft_path = (
        os.path.join(args.save, tgt_dataset, "linear_finetuned.pt")
        if linearized_finetuning
        else os.path.join(args.save, tgt_dataset, "finetuned.pt")
    )
    zs_path = (
        os.path.join(args.save, tgt_dataset, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, tgt_dataset, "zeroshot.pt")
    )
    if not os.path.exists(zs_path):
        raise ValueError(f"The checkpoint for the zero-shot model does not exist at {zs_path}.")
    if not os.path.exists(ft_path):
        raise ValueError(f"The checkpoint for the fine-tuned model does not exist at {ft_path}.")

    if args.finetuning_mode == "linear":
        task_vectors = [LinearizedTaskVector(zs_path, ft_path),]
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = WeightedLinearizedModel(
            image_encoder.model, task_vectors,
            device=rank, blockwise=args.blockwise_coef
        )
    else:
        task_vectors = [NonLinearTaskVector(zs_path, ft_path),]
        image_encoder = ImageEncoder(args)
        image_encoder = WeightedImageEncoder(
            image_encoder, task_vectors,
            device=rank, blockwise=args.blockwise_coef
        )

    tgt_classification_head = get_classification_head(args, tgt_dataset)
    ctr_classification_head = get_classification_head(args, ctr_dataset)
    model = MultiHeadImageClassifier(image_encoder,
        [tgt_classification_head, ctr_classification_head]
    )

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    target_dataset = get_dataset(
        tgt_dataset, preprocess_fn,
        locaiton=args.data_location,
        batch_size=args.batch_size
    )
    control_dataset = get_dataset(
        ctr_dataset, preprocess_fn,
        locaiton=args.data_location,
        batch_size=args.batch_size
    )
    
    # TODO: prepare dataset and dataloader

    # TODO: prepare loss function and optimiser

    # TODO: add training loop, stats printing and parameter saving

    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = ddp_model.module.image_encoder
        eval_single_dataset(image_encoder, test_dataset, args)
        string = 'Coefficients:\t|'
        for c in coef.data:
            string += f"`{c.mean().item():.4f}`|"
        print(string)
        if linearized_finetuning:
            head_path = os.path.join(ckpdir, "linear_layer_coef_.pt")
            coef = ddp_model.module.image_encoder.model.coef
        else:
            head_path = os.path.join(ckpdir, "layer_coef.pt")
            coef = ddp_model.module.image_encoder.coef
        torch.save(coef, head_path)

    cleanup_ddp()

if __name__ == "__main__":

    datasets = {
        "Cars": 1,
        "DTD": 3,
        "EuroSAT": 6,
        "GTSRB": 3,
        "MNIST": 10,
        "RESISC45": 2,
        "SUN397": 1,
        "SVHN": 1,
    }

    args = parse_arguments()
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    args.ctr_dataset = "ImageNet"
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

    for dataset in datasets:
        args.tgt_dataset = dataset
        args.epoch = datasets[dataset]
        print("=" * 100)
        print(f"Learn task vector coefficients of {args.model} on {dataset}")
        print("=" * 100)
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)