"""
Evaluation on a dataset

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al. and Guillermo Ortiz-Jimenez et al.,
at https://github.com/mlfoundations/task_vectors and
https://github.com/gortizji/tangent_task_arithmetic
"""

import numpy as np
import torch
import torchvision
import tqdm

from src import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier

def get_val_features(image_encoder, dataset_name, dataset, args):
    # Build the classification head with all classes, when the dataset only has one.
    if '_' in dataset_name:
        dataset_name_ = dataset_name.split('_')[-1]
    else:
        dataset_name_ = dataset_name
    classification_head = get_classification_head(args, dataset_name_)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()
    model.to(args.device)
    
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    dataloader.shuffle=False
    device = args.device

    feats = []
    labels = []
    logits_ = []
    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits, f = utils.get_logits(x, model, return_features=True)
            
            feats.append(f)
            labels.append(y)
            logits_.append(logits)

    return torch.cat(feats, dim=0), torch.cat(logits_, dim=0), torch.cat(labels, dim=0)

def eval_single_dataset(image_encoder, dataset_name, dataset, args, adapter=None, alpha_vec=None, beta_alpha=None, labels_cache=None, test=False):
    # Build the classification head with all classes, when the dataset only has one.
    if '_' in dataset_name:
        dataset_name_ = dataset_name.split('_')[-1]
    else:
        dataset_name_ = dataset_name
        
    classification_head = get_classification_head(args, dataset_name_)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()
    model.to(args.device)

    if test:
        print("Loading test set")
        dataset = get_dataset(
            dataset_name,
            model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    
    dataloader.shuffle=False
    device = args.device

    confs = torch.tensor([])
    preds = torch.tensor([])
    softmaxs = torch.tensor([])
    targets = torch.tensor([])

    if args.lora:
        import minlora
        minlora.merge_lora(model)

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits, feats = utils.get_logits(x, model, return_features=True)
            feats /= feats.norm(dim=-1, keepdim=True)
            
            if alpha_vec is not None:
                #LP evaluation
                adapter = adapter.to(feats.device)
                vision_logits = adapter(feats)
                text_logits = logits / 100.
                logits = vision_logits + torch.ones(feats.shape[0], 1).to(feats) @ alpha_vec.to(feats) * text_logits
            elif adapter is not None:
                #TIP evaluation
                adapter = adapter.to(feats.device)
                affinity = adapter(feats)
            
                cache_logits = ((-1) * (beta_alpha[0] - beta_alpha[0] * affinity)).exp() @ labels_cache.to(affinity)
                tv_logits = logits            
                logits = tv_logits + cache_logits * beta_alpha[1]
            softmaxs = torch.cat((softmaxs, logits.softmax(1).cpu()), dim=0)
            
            conf, _ = logits.softmax(1).max(1)            
            confs = torch.cat((confs, conf.cpu()), dim=0)
            targets = torch.cat((targets, y.cpu()), dim=0)

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            preds = torch.cat((preds, pred.cpu()), dim=0)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {"top1": top1, "conf":confs, "preds":preds, "softmax":softmaxs, "targets":targets.long()}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%")

    if args.lora:
        minlora.remove_lora(model)

    return metrics

def eval_single_train_dataset(image_encoder, dataset_name, dataloader, args):
    # Build the classification head with all classes, when the dataset only has one.
    if '_' in dataset_name:
        dataset_name_ = dataset_name.split('_')[-1]
    else:
        dataset_name_ = dataset_name
        
    classification_head = get_classification_head(args, dataset_name_)
    model = ImageClassifier(image_encoder, classification_head)
    
    model.to(args.device)
    model.eval()

    device = args.device

    confs = torch.zeros(len(dataloader.dataset))
    preds = torch.zeros(len(dataloader.dataset)).long()
    targets = torch.zeros(len(dataloader.dataset)).long()
    full_preds = torch.zeros((len(dataloader.dataset), classification_head.out_features))

    preprocess = dataloader.dataset.update_transforms(image_encoder.val_preprocess)

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data, index=True)
            x = data["images"].to(device)
            y = data["labels"].to(device)
            ids = data["index"]

            logits = utils.get_logits(x, model)
            full_preds[ids] = logits.softmax(1).cpu()
            conf, _ = logits.softmax(1).max(1)            
            confs[ids] = conf.cpu()

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            preds[ids] = pred.cpu().flatten()
            targets[ids] = y.cpu()

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

            top1 = correct / n

    metrics = {"top1": top1, "conf":confs, "preds":preds, "targets":targets, "full_preds":full_preds}

    dataloader.dataset.update_transforms(preprocess)
    
    return metrics


def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )
    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    task_vector, pretrained_checkpoint, args, scaling_coef, posthoc_linearization=False
):
    image_encoder = task_vector.apply_to(
        pretrained_checkpoint, scaling_coef=scaling_coef
    )
    if posthoc_linearization:
        pretrained_encoder = task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=0.0
        )
        image_encoder = LinearizedImageEncoder(
            init_encoder=pretrained_encoder, image_encoder=image_encoder, args=args
        )
    coef_info = evaluate(image_encoder, args)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info


def evaluate_task_vector(
    task_vector, pretrained_checkpoint, args, posthoc_linearization=False
):
    info = {}
    for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_checkpoint,
            args,
            scaling_coef,
            posthoc_linearization,
        )

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results


def nonlinear_advantage(acc_linear, acc_nonlinear, num_classes):
    err_linear = 1 - acc_linear
    err_nonlinear = 1 - acc_nonlinear
    return (err_linear - err_nonlinear) * num_classes / (num_classes - 1)
