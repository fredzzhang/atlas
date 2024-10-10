"""Given an objective,
learn the coefficients on task vectors for a dataset
and find the optimal combination.

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import os
import time
import json
import torch
import torchvision

from torch.cuda.amp import GradScaler
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import NonLinearTaskVector
from src.composition import WeightedImageEncoder

from src.utils import cosine_lr, get_n_shots, TIPWrapper, LPPWrapper, IndexWrapper, _RepeatSampler
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def main(rank, args):
    # Load the individual task vectors.
    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
        "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101", "Caltech101", "Caltech256",
        "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
    ]
    task_vectors = {}
    for dataset in pool:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    args.rank = rank
    if os.path.exists(args.log_path):
        with open(args.log_path, 'r') as f:
            comp_acc = json.load(f)
    else:
        comp_acc = {}
        
    for dataset, epochs in args.target_datasets.items():
        args.target_dataset = dataset + "Val"
        args.epochs = epochs
        if os.path.isfile(os.path.join(f"{args.save}/{dataset}Val/", "zeroshot_accuracies.json")):
            with open(os.path.join(f"{args.save}/{dataset}Val/", "zeroshot_accuracies.json"), 'r') as f:
                args.zs_acc = json.load(f)
            comp_acc[f"{dataset}Val_zeroshot"] = args.zs_acc[f"{dataset}Val"]
        else:
            if not hasattr(args, 'zs_acc'):
                args.zs_acc = {}

        if type(args.subsample) == float:
            data_amount = f"{args.subsample*100}%"
        else:
            data_amount = f"{args.subsample} shots"
            
        print("=" * 100)
        print(f"Learning task vector coefficients on {dataset} with {args.model} - {data_amount}")
        print("=" * 100)

        comp_acc = train(task_vectors, args, comp_acc)

def train(task_vectors, args, comp_acc={}):

    setup_ddp(args.rank, args.world_size, port=args.port)
    target_dataset = args.target_dataset

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    orig_dataset = target_dataset.replace("Val", "")
    # Remove the task vector for the target task
    task_vectors = [v for k, v in task_vectors.items() if orig_dataset != k]

    image_encoder = ImageEncoder(args)
    image_encoder = WeightedImageEncoder(
        image_encoder, task_vectors, blockwise=args.blockwise_coef, part_wise=args.atlas_n, 
    )

    classification_head = get_classification_head(args, target_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    # TIP's more aggressive random crop with horizontal flip
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ), torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + model.train_preprocess.transforms[-3:])
    
    dataset = get_dataset(
        target_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=8,
    )

    if type(args.subsample) == int:
        if os.path.isfile(f"{args.save}/{target_dataset}/{args.subsample}_shots_{args.seed}.pt") and args.seed == 1:
            to_keep = torch.load(f"{args.save}/{target_dataset}/{args.subsample}_shots_{args.seed}.pt")
        else:
            to_keep = get_n_shots(dataset.train_dataset, args.subsample, classification_head.out_features, args)
            torch.save(to_keep, f"{args.save}/{target_dataset}/{args.subsample}_shots_{args.seed}.pt")
            
        r = len(to_keep) / args.batch_size
        if r < 10:
            over_sampling = 10/r
            over_sampling = int(over_sampling) + 1
            print(f"Oversampling {over_sampling} times")
            to_keep = torch.cat([to_keep] * over_sampling)
            
    else:
        if os.path.isfile(f"{args.save}/{target_dataset}/{args.subsample}_{args.seed}.pt") and args.seed == 1:
            to_keep = torch.load(f"{args.save}/{target_dataset}/{args.subsample}_{args.seed}.pt")
        else:
            to_keep = torch.randperm(len(dataset_index))[:int(len(dataset_index)*args.subsample)]
            torch.save(to_keep, f"{args.save}/{target_dataset}/{args.subsample}_{args.seed}.pt")
        
    index_dataset = IndexWrapper(dataset.train_dataset)
    sampler = torch.utils.data.SubsetRandomSampler(to_keep)        
    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8)
    
    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.rank],
        find_unused_parameters=False,
        output_device=args.rank,
    )

    num_batches = len(ddp_loader)
    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # Do not use warm up
    scheduler = cosine_lr(
        optimizer, args.lr, 0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    scaler = GradScaler()    
    if is_main_process():
        if f"{target_dataset}_zeroshot" not in comp_acc.keys():
            comp_acc[f"{target_dataset}_zeroshot"] = eval_single_dataset(image_encoder, target_dataset.replace('Val',''), args)["top1"]
            with open(os.path.join(f"{args.save}/{target_dataset}/", "zeroshot_accuracies.json"), 'w') as f:
                json.dump({f"{target_dataset}": comp_acc[f"{target_dataset}_zeroshot"]}, f, indent=4)
            args.zs_acc[f"{target_dataset}"] = comp_acc[f"{target_dataset}_zeroshot"]
            
        print(f"=> Zero-shot accuracy on {target_dataset}:\t{100*args.zs_acc[target_dataset]:.2f}%.")
        
    best_coef = ddp_model.module.image_encoder.coef.data.clone()
    best_acc = args.zs_acc[target_dataset]
    for epoch in range(args.epochs):
        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            data_time = time.time() - start_time

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(inputs)
                labels = batch["labels"].cuda()
                loss = loss_fn(logits, labels)
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"           # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",   # noqa: E501
                    flush=True,
                )
        
        # Evaluate after each epoch
        if is_main_process():
            image_encoder = ddp_model.module.image_encoder
            coef = ddp_model.module.image_encoder.coef
            acc = eval_single_dataset(image_encoder, target_dataset, args)["top1"]
            if acc > best_acc:
                best_acc = acc
                best_coef = coef.data.clone()

    if is_main_process():
        comp_acc[target_dataset] = best_acc
        target_dataset = target_dataset.replace("Val", "")
        image_encoder = ddp_model.module.image_encoder
        image_encoder.coef = torch.nn.Parameter(best_coef)
        comp_acc[target_dataset] = eval_single_dataset(image_encoder, target_dataset, args)["top1"]
        with open(args.log_path, 'w') as f:
            json.dump(comp_acc, f, indent=4)
        if os.path.isfile(args.head_path):
            heads = torch.load(args.head_path)
        else:
            heads = {}
            
        heads[target_dataset] = best_coef
        torch.save(heads, args.head_path)

    if args.adapter is not None:
        comp_acc = train_adapter(ddp_model, ddp_loader, args, comp_acc, which=args.adapter)
        
    cleanup_ddp()
    return comp_acc


def train_adapter(ddp_model, ddp_loader, args, comp_acc, which='lpp'):
    #Extracting features:
    all_features, all_labels, all_indexes, all_logits = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(ddp_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()

            logits, features = ddp_model(inputs, return_features=True)
            labels = batch["labels"]
            
            all_features.append(features.detach().cpu())
            all_labels.append(labels)
            all_indexes.append(batch["index"])
            all_logits.append(logits.detach().cpu())
            
    logits_cache = torch.cat(all_logits)
    features_cache = torch.cat(all_features)
    labels = torch.cat(all_labels)
    indexes = torch.cat(all_indexes)
    indexes_to_i = {indexes[i].item():i for i in range(len(indexes))}
    
    model = ddp_model.module
    if which == 'lpp':
        if type(args.subsample) == float:
            shots = 100
        else:
            shots = args.subsample            
        model = LPPWrapper(model, features_cache, labels, shots)
        epochs = 300
        lr = model.lr_temp
    elif which == 'tip':
        model = TIPWrapper(model, features_cache, labels)
        lr = 1e-3
        epochs = 10
    else:
        raise NotImplementedError(f"Adapter {which} unknown")

    model = model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.rank],
        find_unused_parameters=False,
        output_device=args.rank,
    )
    
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.wd)
    num_batches = len(ddp_loader)
    scheduler = cosine_lr(
        optimizer, lr, 0,
        epochs * num_batches // args.num_grad_accumulation,
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    print_every = 100
    ddp_loader._DataLoader__initialized = False
    ddp_loader.batch_sampler = _RepeatSampler(ddp_loader.batch_sampler, epochs)
    ddp_loader._DataLoader__initialized = True

    for i, batch in enumerate(ddp_loader):
        start_time = time.time()
        epoch = i // num_batches
        step = (
            i // args.num_grad_accumulation
            + epoch * num_batches // args.num_grad_accumulation
        )
        
        batch = maybe_dictionarize(batch)
        inputs = batch["images"].cuda()
        data_time = time.time() - start_time

        ids = [indexes_to_i[i.item()] for i in batch['index']]
        l_cache, f_cache = logits_cache[ids].to(inputs), features_cache[ids].to(inputs)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = ddp_model(inputs, l_cache, f_cache)
            labels = batch["labels"].to(logits.device)
            loss = loss_fn(logits, labels)
            loss = loss / args.num_grad_accumulation

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
            percent_complete = 100 * (i + 1) / len(ddp_loader)
            print(
                f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{len(ddp_loader)}]\t"           # noqa: E501
                f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",   # noqa: E501
                flush=True,
            )

    if is_main_process():
        #comp_acc[target_dataset+f"_{which}"] = best_acc
        target_dataset = args.target_dataset
        target_dataset = target_dataset.replace("Val", "")
        image_encoder = ddp_model.module.model.image_encoder        
        comp_acc[target_dataset+f"_{which}"] = eval_single_dataset(image_encoder, target_dataset, args, model=ddp_model)["top1"]
        with open(args.log_path, 'w') as f:
            json.dump(comp_acc, f, indent=4)
            
        if os.path.isfile(args.head_path):
            heads = torch.load(args.head_path)
        else:
            heads = {}

        adapter_coefs = {k:v for k,v in ddp_model.module.state_dict().items() if v.requires_grad}
        heads[target_dataset] = adapter_coefs
        torch.save(heads, args.head_path)
        
    return comp_acc

if __name__ == "__main__":

    target_datasets = {
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
        "Caltech101":10,
    }

    args = parse_arguments()
    args.target_datasets = target_datasets
    # HACK: Some command line arguments are overwritten by defaults here.
    args.lr = 1e-1
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    args.logdir += f"{args.model}"
    if type(args.subsample) == float:
        args.logdir += f"/{args.subsample*100:.0f}perc"
    else:
        args.logdir += f"/{args.subsample}shots"
        args.target_datasets = {k:10 for k,v in args.target_datasets.items()}#10 epochs for few-shots using ViTs. 
        
    args.save = os.path.join(args.save, f'{args.model}')
    if args.seed is not None:
        args.logdir += f"/{args.seed}"
        
    args.head_path = os.path.join(args.logdir, "learned_composition.pt")
    args.log_path = os.path.join(args.logdir, "learned_composition.json")

    os.makedirs(args.logdir, exist_ok=True)      
    
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
