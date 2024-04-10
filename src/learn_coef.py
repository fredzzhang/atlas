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
from functorch import jvp, make_functional_with_buffers

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset, eval_single_train_dataset, get_val_features
from src.heads import get_classification_head
from src.modeling import ImageEncoder, ImageClassifier
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.utils import cosine_lr, cosine_annealing_lr, adjust_lr, TwoTransform, TwoAsymetricTransform, adjust_lr_lp
from src.sampler import TwoStreamBatchSampler, SubsetSampler

import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from lightning.fabric import Fabric

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def cb_softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    entro =  -(x.softmax(1) * x.log_softmax(1)).sum(1)
    prior = torch.ones(x.shape[1])/x.shape[1]
    prior = prior.to(x.device)
    pred_mean =  x.softmax(1).mean(0)
    cb_penalty = torch.sum(prior*torch.log(prior/pred_mean))
    
    return entro + cb_penalty

def l1_reg(coefs: torch.Tensor, lambda1: float=0.5) -> torch.Tensor:
    """ Simple L1 regularization. """
    l1_regularization = lambda1 * torch.norm(coefs, p=1, dim=0)
    return l1_regularization.mean()
    
def ce_loss_trusted(logits1: torch.Tensor, logits2: torch.Tensor, preds: torch.FloatTensor=None, trusted: torch.BoolTensor=None, lam: float=None, index: torch.Tensor=None) -> torch.Tensor:
    trusted = trusted.to(logits1)
    return (F.cross_entropy(logits2, preds.to(logits2), reduction='none')*trusted).sum() / trusted.sum()

def ssl_loss_trusted(logits1: torch.Tensor, logits2: torch.Tensor, targets: torch.FloatTensor=None, trusted: torch.BoolTensor=None, lam: float=None, index: torch.Tensor=None, thresh: float=0.99) -> torch.Tensor:
    """
    Computes a Fixmatch type semi supervised loss given trusted samples.

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
    
    guessed_targets = one_hot ** (.5) #temp sharp    
    guessed_targets = guessed_targets / guessed_targets.sum(dim=1, keepdim=True) #normalization
    
    if trusted is not None:
        guessed_targets[trusted] = targets[trusted].to(guessed_targets)

    one_hot = guessed_targets.detach()
    one_hot = F.one_hot(torch.argmax(guessed_targets, dim=1), num_classes=one_hot.shape[1]).float()
    
    w, _ = guessed_targets.max(1)
    w = (w > thresh).to(logits2)
    if lam is not None:
        return lam * F.cross_entropy(logits2, guessed_targets) + (1-lam) * F.cross_entropy(logits2, guessed_targets[index])
    trusted = trusted.to(logits1)
    #return (F.cross_entropy(logits2, guessed_targets, reduction='none')*trusted).sum() / trusted.sum()
    return (F.cross_entropy(logits2, guessed_targets, reduction='none') * w).sum() / w.sum() 


def simclr_loss(logits1: torch.Tensor, logits2: torch.Tensor, mu:float=0.2) -> torch.Tensor:
    """Simclr contrastive loss (Npairs for now)"""
    logits1, logits2 = F.normalize(logits1, 2), F.normalize(logits2, 2)
    mask = torch.eye(logits1.shape[0]).to(logits1)

    sims = torch.div(logits1 @ logits2.T, mu)

    npairs_loss = - (mask * sims.log_softmax(1)).sum(1)
    return npairs_loss

def simclr_mixup_loss(logits1: torch.Tensor, logits2: torch.Tensor, lam:float, index:torch.tensor, mu:float=0.2) -> torch.Tensor:
    """iMix contrastive loss"""
    logits1, logits2 = F.normalize(logits1, 2), F.normalize(logits2, 2)
    mask = torch.eye(logits1.shape[0]).to(logits1)

    sims = torch.div(logits1 @ logits2.T, mu)
    
    npairs_loss = - lam * (mask * sims.log_softmax(1)).sum(1) - (1-lam) * (mask[index] * sims.log_softmax(1)).sum(1)
    return npairs_loss


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
    def __init__(self, model, task_vectors, device, layerwise=False, attn=False) -> None:
        """A wrapper class to enable compositions of task vectors"""
        super().__init__()

        func, params, self.buffer = make_functional_with_buffers(model)
        # NOTE This is important to avoid the following error
        # NotImplementedError: Cannot copy out of meta tensor; no data!
        self.func = lambda p, x: func(p, self.buffer, x)
        self.params = torch.nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False

        self.attn = attn
        self.attn_l = []
        n = 0
        for i, (name, p) in enumerate(model.named_parameters()):
            if 'attn.in_proj' in name:
                self.attn_l.append(i)
                n+=1
            
        self.device = device
        # Copy the attributes from the image encoder.
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = model.cache_dir

        dparams = []
        for tv in task_vectors:
            dp = [tv.vector[k].to(device) for k in tv.vector]
            dparams.append(dp)
        self.layerwise = layerwise
        self.dparams = dparams


        if self.attn:
            self.coef1 = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params), 3))
        elif self.layerwise:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
        else:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors),))

    def __call__(self, x, zero_shot=False) -> torch.Tensor:
        if zero_shot:
            return self.func(self.params, x)
        if self.attn:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef1)]) if i not in self.attn_l else sum([torch.cat((p[:len(p)//3] * c[i, 0], p[len(p)//3:2*len(p)//3] * c[i, 1], p[2*len(p)//3:len(p)] * c[i, 2]))  for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        elif self.layerwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]

        new_params = [(dp + p).to(x.device) for i, (dp, p) in enumerate(zip(dparams, self.params))]
        
        return self.func(new_params, x)

class IndexWrapper(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        if isinstance(self.dataset, torch.utils.data.dataset.Subset):
            self.dataset = self.dataset.dataset
        
    def __getitem__(self, index):
        return self.dataset[index], index

    def update_transforms(self, new_transform):
        if hasattr(self.dataset, "transform"):
            preprocess = self.dataset.transform
            self.dataset.transform = new_transform
        elif hasattr(self.dataset, "transforms"):
            preprocess = self.dataset.transforms
            self.dataset.transforms = new_transform
        else:
            raise AttributeError(f"Can't find transform attribute of dataset {self.dataset}")
        return preprocess
    
    def __len__(self):
        return len(self.dataset)
    
def main(rank, args):

    # Load the individual task vectors.
    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB",
        "MNIST", "RESISC45", "SUN397", "SVHN", "CIFAR10",
        "CIFAR100", "ImageNet", "STL10", "Food101", "Caltech256",
        "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200",
        "PascalVOC", "Country211"
    ]
    if args.add_random_tv is not None:
        pool = []
        for i in range(args.add_random_tv):
            pool.append(f"randomtv_{i}")
    task_vectors = {}
    l = 0
    for i, dataset in enumerate(pool):
        if "randomtv" in dataset:
            pretrained_checkpoint = f"{args.save}/{pool[0]}Val/zeroshot.pt"
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, scale=args.scale)
        elif args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors[dataset] = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        elif args.optimally_random is not None:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            opt_tv = torch.load(os.path.join(args.optimally_random, f'tv_{i}.pth'))
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, vector=opt_tv["state_dict"], scale=args.scale)
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint, scale=args.scale)
            task_vectors[dataset] = task_vectors[dataset]

    args.rank = rank
    fname = f"results/{args.model}/"
    if args.datasets[list(args.datasets.keys())[0]] == 0:
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
                
    fname = os.path.join(fname, f"{datetime.today().strftime('%Y-%m-%d-%H-%M')}.txt")
    splt = fname.split('/')
    for i in range(len(splt)-1):
        if not os.path.exists('/'.join(splt[:i+1])):
            os.mkdir('/'.join(splt[:i+1]))


    fname2 = f"results/results_{args.loss_fn}{'_layerwise' if args.layerwise else '_global'}{'_optitv' if args.optimally_random else ''}{'_attn' if args.attn else ''}{'_semi'+str(args.semi) if args.semi else ''}{'_tipft' if args.tip_ft else ''}{'_tiponly' if args.tip_only else ''}{'_tipcot' if args.tip_cot else ''}{'_lp' if args.lp else ''}_{datetime.today().strftime('%Y-%m-%d-%H:%M')}.txt"
    with open(fname, 'a') as f: f.writelines([f"Task vector results for loss {args.loss_fn} on datasets {list(args.datasets.keys())}. {len(task_vectors.keys())} task vectors\n"])
    with open(fname2, 'a') as f: f.writelines([f"Task vector results for loss {args.loss_fn} on datasets {list(args.datasets.keys())}. {len(task_vectors.keys())} task vectors\n"])
    coef_dict = {}
    
    for dataset in args.datasets:
        args.epochs = args.datasets[dataset]
        args.test_dataset = dataset
        print("=" * 100)
        print(f"Finetuning task vector coefficients of {args.model} on {dataset}")
        print("=" * 100)

        base_acc, best_acc, best_epoch, best_coef, acc = train(task_vectors, args)

        coef_dict[dataset] = {'coefs': acc["coefs"], 'preds':acc["softmax"], 'targets':acc["targets"]}
        torch.save(coef_dict, fname.replace('.txt', '.pth'))
        torch.save(coef_dict, fname2.replace('.txt', '.pth'))
        
        with open(fname, 'a') as f: f.writelines([f"{dataset}\n", f"Base accuracy {base_acc}, best accuracy {best_acc} at epoch {best_epoch}\n, tip beta {acc['beta_alpha'][0]:.3f} alpha {acc['beta_alpha'][1]:.3f}", f"{best_coef}\n"])
        with open(fname2, 'a') as f: f.writelines([f"{dataset}\n", f"Base accuracy {base_acc}, best accuracy {best_acc} at epoch {best_epoch}\n, tip beta {acc['beta_alpha'][0]:.3f} alpha {acc['beta_alpha'][1]:.3f}", f"{best_coef}\n"])
    with open(fname, 'a') as f: f.writelines([f"\nArguments {args}"])
    with open(fname2, 'a') as f: f.writelines([f"\nArguments {args}"])
    
    
def train(task_vectors, args):
    scaler = torch.cuda.amp.GradScaler()
    #setup_ddp(args.rank, args.world_size, port=args.port)
    test_dataset = args.test_dataset + 'Val'

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    orig_dataset = test_dataset.split('Val')[0]
    task_vectors = [v for k, v in task_vectors.items() if orig_dataset != k]

    if args.finetuning_mode == "linear":
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = LinearizedModel_(image_encoder.model, task_vectors, device=args.rank)
    else:
        image_encoder = ImageEncoder(args)
        image_encoder = ImageEncoder_(image_encoder, task_vectors, device=args.rank, layerwise=args.layerwise, attn=args.attn)
    
    classification_head = get_classification_head(args, test_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    
    if 'simclr' in args.loss_fn:
        size = 224
            
        preprocess_fn = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(256, antialias=True),
             torchvision.transforms.RandomCrop(size)] + \
            model.val_preprocess.transforms[-3:]
        )

        preprocess_fn = TwoAsymetricTransform(model.val_preprocess, preprocess_fn)
    else: #Using the TIP augmentations to learn the coeficients but this does not make a huge difference.
        preprocess_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ] + model.val_preprocess.transforms[-3:])
        
        #preprocess_fn = model.val_preprocess

    dataset = get_dataset(
        test_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    #Changing the transforms in the val/test dataset
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    if isinstance(dataset, torch.utils.data.dataset.Subset):#Resisc
        dataloader.dataset.dataset.transform = model.val_preprocess
        dataloader.dataset.dataset.transforms = model.val_preprocess
    else:
        dataloader.dataset.transform = model.val_preprocess
        dataloader.dataset.transforms = model.val_preprocess
    
    index_dataset = IndexWrapper(dataset.train_dataset) #Wrapping to get index of the samples
    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    # Override shuffle to True
    data_loader.shuffle = True
    num_batches = len(data_loader)
            

    #TO DO: implement lightning Fabric support for DDP and multi-gpu    
    loss_fn = {
        'entropy': softmax_entropy,
        'cb_entropy': cb_softmax_entropy,
        'cross_entropy': torch.nn.CrossEntropyLoss(),
        'trusted': torch.nn.CrossEntropyLoss(),
        'entro_trusted': softmax_entropy,
        'simclr': simclr_loss,
        'ssl_simclr_trusted': ssl_loss_trusted,
        'ssl_simclr_trusted_entro': ssl_loss_trusted,
        'simclr_mixup': simclr_mixup_loss,
    }[args.loss_fn]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    #optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-5)

    if args.lr_scheduler=="annealing":
        lrs = cosine_annealing_lr(args.lr, 0, args.epochs)
    else:
        scheduler = cosine_lr(
            optimizer,
            args.lr,
            args.warmup_length,
            args.epochs * num_batches // args.num_grad_accumulation,
        )

    if args.finetuning_mode == "linear":
        coef = model.image_encoder.model.coef
    else:
        coef = model.image_encoder.coef        
        if args.attn:
            coef1 = model.image_encoder.coef1

    model.eval()
    best_acc = 0
    max_ep = args.epochs
    if args.tip_only:
        max_ep = 1
        
    if "ssl" in args.loss_fn:
        threshs = torch.arange(args.epochs) / args.epochs / 10 + .9
    i = 0
    loss = torch.tensor(0)
    data_time = 0
    batch_time = 0
    epoch = 0

    fabric = Fabric(accelerator="cuda", devices=1, strategy="ddp")
    fabric.launch()

    model, optimizer = fabric.setup(model, optimizer)
    data_loader = fabric.setup_dataloaders(data_loader)
    
    for epoch in range(max_ep):
        # Evaluate before each epoch
        if args.lr_scheduler=="annealing":
            adjust_lr(optimizer, lrs[epoch])

        if is_main_process():
            # We only need to evaluate the model on the first GPU.
            image_encoder = model.image_encoder
            if epoch == 0:
                #Accuracy of the Zero-shot on the test set
                acc = eval_single_dataset(image_encoder, test_dataset, dataset, args, test=True)
            else:
                acc = eval_single_dataset(image_encoder, test_dataset, dataset, args, test=False)
                
            string = 'Coefficients:\t|'
            for c in coef.data:
                if args.layerwise:
                    string += f"`{c.mean():.4f}`|"
                else:                    
                    string += f"`{c:.4f}`|"
            print(string)
            if acc['top1'] > best_acc and epoch > 0:
                best_acc = acc['top1']
                best_epoch = epoch
                best_coefs = string
                coefs = coef.data.detach().cpu()
                if args.attn:
                    coefs1 = coef1.data.detach().cpu()

            if epoch == 0:
                base_acc = acc['top1']
                
                
            if ('trusted' in args.loss_fn and ((epoch >= 0 and 'entro' not in args.loss_fn) or epoch >= 1)) and not (args.semi and epoch >=1):

                acc = eval_single_train_dataset(image_encoder, test_dataset, data_loader, args)
                
                confs = acc['conf']
                preds = acc['preds']
                targets = acc['targets']
                full_preds = acc['full_preds']

                if args.semi:
                    r = torch.randperm(len(confs))
                    k = args.semi                    
                    to_keep = torch.tensor([], dtype=torch.long)
                    unlabeled = torch.tensor([], dtype=torch.long)
                    for c in range(classification_head.out_features):
                        cond = (targets == c) * (confs > 0)                      
                        ids_c = torch.arange(len(targets))[cond]
                        a = torch.randperm(len(ids_c))
                        to_keep = torch.cat((to_keep, ids_c[a[-k:]]))
                        unlabeled = torch.cat((unlabeled, ids_c[a[:-k]]))

                    unlabeled = torch.tensor([u for u in unlabeled if confs[u] > 0], dtype=torch.long)
                    unlabeled = torch.unique(unlabeled)
                    preds[to_keep] = targets[to_keep]
                else:
                    k = min(int((len(subset) / classification_head.out_features) / 10), 100)
                    #k = max(int((len(subset) / classification_head.out_features) / 10), 10)
                    #k = int((len(subset) / classification_head.out_features) / 10)
                    print(k)
                    to_keep = torch.tensor([], dtype=torch.long)
                    unlabeled = torch.tensor([], dtype=torch.long)
                    for c in range(classification_head.out_features):
                        if False: 
                            a = torch.argsort(full_preds[:, c])
                            to_keep = torch.cat((to_keep, a[-k:]))
                            unlabeled = torch.cat((unlabeled, a[:-k]))

                            preds[a[-k:]] = c # There are probably overlapping examples here

                            #print((preds[a[-k:]] == targets[a[-k:]]).sum() / len(a[-k:]))
                            #print((preds[a[:-k]] == targets[a[:-k]]).sum() / len(a[:-k]))
                        else:
                            ids_c = torch.arange(len(preds))[preds == c]
                            a = torch.argsort(confs[ids_c])
                            to_keep = torch.cat((to_keep, ids_c[a[-k:]]))
                            unlabeled = torch.cat((unlabeled, ids_c[a[:-k]]))

                            #print((preds[ids_c[a[-k:]]] == targets[ids_c[a[-k:]]]).sum() / len(ids_c[a[-k:]]))
                            #print((preds[ids_c[a[:-k]]] == targets[ids_c[a[:-k]]]).sum() / len(ids_c[a[:-k]]))

                        
                    unlabeled = torch.tensor([u for u in unlabeled if confs[u] > 0], dtype=torch.long)
                    unlabeled = torch.unique(unlabeled)

                print("Correctness")
                print((preds[to_keep] == targets[to_keep]).sum() / len(to_keep))
                print((preds[unlabeled] == targets[unlabeled]).sum() / len(unlabeled))
                
                preds = F.one_hot(preds, num_classes=classification_head.out_features)                
                if 'ssl' in args.loss_fn:
                    low_confs = torch.arange(len(data_loader.dataset))[unlabeled]
                    print(f"Got {len(to_keep)} trusted and {len(unlabeled)} untrusted samples")
                    sampler = TwoStreamBatchSampler(low_confs, to_keep, args.batch_size)
                    data_loader = torch.utils.data.DataLoader(index_dataset, batch_sampler=sampler, num_workers=16)
                else:
                    print(f"Got {len(to_keep)} trusted samples")
                    r = len(to_keep) / args.batch_size
                    if r < 10:
                        over_sampling = 10/r
                        over_sampling = int(over_sampling) + 1
                        print(f"Oversampling {over_sampling} times")
                        to_keep = torch.cat([to_keep] * over_sampling)
                    sampler = torch.utils.data.SubsetRandomSampler(to_keep) 
                    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=16)                    
                num_batches = len(data_loader)
                data_loader = fabric.setup_dataloaders(data_loader)
                
                if args.tip_only:
                    i = 0
                    loss = torch.tensor(0)
                    data_time = 0
                    batch_time = 0
                    continue
        
        for i, batch in enumerate(data_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )
            batch = maybe_dictionarize(batch, index=True)
            inputs = batch["images"]
            
            if 'simclr' in args.loss_fn:
                inputs2 = batch["images_aug"]
            if 'trusted' in args.loss_fn:
                ids = batch['index']

            data_time = time.time() - start_time

            if 'mixup' in args.loss_fn:
                lam = np.random.beta(1, 1)
                lam = max(lam, 1-lam)
                index = torch.randperm(len(inputs))
                inputs = lam * inputs + (1-lam) * inputs[index]
                
            elif 'ssl' in args.loss_fn and 'entro' not in args.loss_fn:
                with torch.no_grad():
                    logits = model(inputs)
            elif 'simclr' in args.loss_fn:
                logits, feats = model(inputs, return_features=True)
            else:
                logits = model(inputs)

            if args.loss_fn == 'cross_entropy':
                labels = batch["labels"]
                loss = loss_fn(logits, labels)
            elif 'entro_conf' in args.loss_fn:
                loss = loss_fn(logits).mean()
            elif 'ssl_simclr_trusted' in args.loss_fn:
                lam = np.random.beta(1, 1)
                lam = max(lam, 1-lam)
                index = torch.randperm(len(inputs2))
                #inputs2 = lam * inputs2 + (1-lam) * inputs2[index]
                if epoch == 0 and 'entro' in args.loss_fn:
                    #logits2, feats2 = model(inputs2, return_features=True)
                    #loss = simclr_loss(feats, feats2).mean()
                    loss = softmax_entropy(logits).mean()
                    #loss = F.cross_entropy(logits, labels.float())
                    #labels = batch["labels"]
                    #labels = F.one_hot(labels, num_classes=classification_head.out_features)
                elif epoch < 2 and False:
                    logits2 = model(inputs2)
                    trusted = torch.ones(args.batch_size).bool()
                    trusted[:args.batch_size//2] = 0 #First half of the batch are untrusted labels
                    loss = ce_loss_trusted(logits, logits2, preds[ids], trusted)#, lam, index)
                else:
                    logits2 = model(inputs2)
                    trusted = torch.ones(args.batch_size).bool()
                    trusted[:3*args.batch_size//4] = 0 #First half of the batch are untrusted labels
                    loss = loss_fn(logits, logits2, preds[ids], trusted, thresh=threshs[epoch])#, lam, index)
                    #loss = softmax_entropy(logits).mean()

            elif 'trusted' in args.loss_fn:
                loss = loss_fn(logits, preds[ids].to(logits))
            elif 'ssl' in args.loss_fn:
                lam = np.random.beta(1, 1)
                lam = max(lam, 1-lam)
                index = torch.randperm(len(inputs2))
                #inputs2 = lam * inputs2 + (1-lam) * inputs2[index]
                logits2 = model(inputs2)
                loss = loss_fn(logits, logits2, preds)#, lam, index)
            elif 'simclr' in args.loss_fn:
                logits2, feats2 = model(inputs2, return_features=True)
                if 'entro' in args.loss_fn:
                    loss = loss_fn(logits, logits2, feats, feats2).mean()
                elif 'mixup' in args.loss_fn:
                    loss = loss_fn(logits, logits2, lam, index).mean()
                else:
                    loss = loss_fn(feats, feats2).mean()                    
            else:
                loss = loss_fn(logits).mean(0)

            if args.l1:
                loss += l1_reg(model.image_encoder.coef)

                    
            fabric.backward(loss)

            if (i + 1) % args.num_grad_accumulation == 0:
                if not args.lr_scheduler=="annealing":
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
    
    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = model.image_encoder
        #with torch.autocast(device_type="cuda"):
        acc = eval_single_dataset(image_encoder, test_dataset, dataset, args)
        string = 'Coefficients:\t|'
        for c in coef.data:
            if args.layerwise:
                string += f"`{c.mean():.4f}`|"
            else:                    
                string += f"`{c:.4f}`|"
                
        if acc['top1'] > best_acc:
            best_acc = acc['top1']
            best_coefs = string
            best_epoch = epoch
            coefs = coef.data.detach().cpu()
            if args.attn:
                coefs1 = coef1.data.detach().cpu()
            
        print(string)

        if args.tip_ft:
            #Load best coefs            
            #model.image_encoder.coef.requires_grad = False
            #if args.attn:
            #    model.image_encoder.coef1.requires_grad = False
            image_encoder = model.image_encoder
            with torch.no_grad():
                to_keep_u = torch.unique(to_keep)
                to_keep_u.sort()
                subset_t = SubsetSampler(to_keep_u)
                data_loader_t = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, sampler=subset_t, num_workers=16)

                aug_n = 1
                tbar = tqdm(range(aug_n))
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
                adapter, alpha_vec, lr_alpha, lr_temp = init_lp(features_cache, labels.long(), classification_head.weight.T / 100., args.semi)                
            else:
                features_cache = features_cache.permute(1, 0)
                adapter = nn.Linear(features_cache.shape[0], features_cache.shape[1], bias=False)
                adapter.weight = nn.Parameter(features_cache.t())
                adapter.register_parameter("beta_alpha", nn.Parameter(torch.tensor([1.,1.]))) #Only way to update the parameter on the GPU, otherwise update on the CPU. In any case fp16 is non trivial
                #adapter.beta_alpha = torch.tensor([1.,2.])
            adapter = adapter.to(features_cache)                

            if 'ssl' in args.loss_fn:
                data_loader = torch.utils.data.DataLoader(index_dataset, batch_sampler=sampler, num_workers=16)
            else:
                data_loader = torch.utils.data.DataLoader(index_dataset, sampler=sampler, num_workers=16, batch_size=args.batch_size)
                
            if args.lp:
                optimizer = torch.optim.SGD(adapter.parameters(), lr_temp, momentum=0.9)
                if args.tip_cot:
                    optimizer = torch.optim.SGD([{'params': params} , {'params': adapter.parameters()}], lr_temp, momentum=0.9)
            elif args.tip_cot:
                optimizer = torch.optim.AdamW([{'params': params}, {'params': adapter.weight}, {'params': adapter.beta_alpha}], lr=0.001, eps=1e-4)#Optimize tip + coefs
            else:
                optimizer = torch.optim.AdamW([{'params': adapter.weight}, {'params': adapter.beta_alpha}], lr=0.001, eps=1e-4)

            adapter, optimizer = fabric.setup(adapter, optimizer)
            data_loader = fabric.setup_dataloaders(data_loader)

            if not args.lp:
                adapter.eval()   
                acc = eval_single_dataset(image_encoder, test_dataset, dataset, args, adapter=adapter, beta_alpha=adapter.beta_alpha, labels_cache=labels_cache)

            
            if acc['top1'] > best_acc:
                best_acc = acc['top1']
                best_coefs = string
                best_epoch = epoch
                coefs = coef.data.detach().cpu()
                beta_alpha = adapter.beta_alpha
                if args.attn:
                    coefs1 = coef1.data.detach().cpu()

            model.eval()
            if not args.tip_cot:
                args.epochs = 1
            else:
                args.epochs *= 2

            if args.lp:
                lrs = cosine_annealing_lr(0.1, 0, args.epochs)
            else:
                lrs = cosine_annealing_lr(0.001, 0, args.epochs)
            feats_, logits_, labels_ = [], [], []
            for epoch in range(args.epochs):
                adapter.train()
                if not args.lp:
                    if args.tip_cot:
                        if args.add_random_tv:
                            adjust_lr(optimizer, lrs[epoch], [.1,1.,100.])
                        else:
                            adjust_lr(optimizer, lrs[epoch], [100.,1., 100.])
                    else:
                        adjust_lr(optimizer, lrs[epoch], [1., 100.])
                else:
                    adjust_lr_lp(optimizer, lrs[epoch], lr_temp)

                tbar = tqdm(data_loader)
                tbar.set_description(f"Learning adaptor, epoch {epoch}/{args.epochs}, lr {optimizer.param_groups[0]['lr']:.3f}")
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
                        text_logits = logits / 100.
                        logits = vision_logits + torch.ones(feats.shape[0], 1).to(feats) @ alpha_vec.to(feats) * text_logits
                    else:
                        affinity = adapter(feats)
                        cache_logits = ((-1) * (adapter.beta_alpha[0] - adapter.beta_alpha[0] * affinity)).exp() @ labels_cache.to(affinity)  
                        tv_logits = logits                    
                        logits = cache_logits * adapter.beta_alpha[1] + tv_logits
                    
                    loss = F.cross_entropy(logits, batch["labels"])

                    if not args.lp:
                        tbar.set_description(f"Learning adaptor, epoch {epoch}/{args.epochs}, lr {optimizer.param_groups[0]['lr']:.3f}, loss {loss.item():.3f}, beta_alpha {adapter.beta_alpha[0]:.2f} {adapter.beta_alpha[1]:.2f}")
                    else:
                        tbar.set_description(f"Learning adaptor, epoch {epoch}/{args.epochs}, lr {optimizer.param_groups[0]['lr']:.3f}, loss {loss.item():.3f}")
                    
                    fabric.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if args.lp and (epoch + 1) % 10 == 0:
                        alpha_vec.data -= lr_alpha * alpha_vec.grad.data



                if not args.tip_cot:
                    features = torch.cat(feats_, dim=0)
                    logits = torch.cat(logits_, dim=0)
                    labels = torch.cat(labels_, dim=0)
                    del feats_, logits_, labels_
                    tbar = tqdm(range(1, 300))
                    tbar.set_description(f"Tuning {'LP++' if args.lp else 'TIP'}")
                    lrs = cosine_annealing_lr(0.001, 0, 300)
                    for e in tbar:
                        adapter.train()
                        
                        if args.lp:
                            vision_logits = adapter(features)
                            text_logits = logits / 100.
                            f_logits = vision_logits + torch.ones(features.shape[0], 1).to(features) @ alpha_vec.to(features) * text_logits
                        elif args.tip_ft:
                            affinity = adapter(features)
                            cache_logits = ((-1) * (adapter.beta_alpha[0] - adapter.beta_alpha[0] * affinity)).exp() @ labels_cache.to(affinity)  
                            tv_logits = logits
                            f_logits = cache_logits * adapter.beta_alpha[1] + tv_logits
                            
                        loss = F.cross_entropy(f_logits, labels)
                        
                        fabric.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        if (e + 1) % 10 == 0:
                            tbar.set_description(f"Tuning {'LP++' if args.lp else 'TIP'}, epoch {e}, best val acc {best_acc:.2f}, loss {loss.item():.2f}")
                            if args.lp:
                                alpha_vec.data -= lr_alpha * alpha_vec.grad.data
                                
                        if args.tip_ft and not args.lp:
                            adjust_lr(optimizer, lrs[e], [1., 100.])
                            
                        adapter.eval()
                        if e == 1:
                            val_features, val_logits, val_labels = get_val_features(image_encoder, test_dataset, dataset, args)
                            val_features /= val_features.norm(dim=-1, keepdim=True)
                        if args.lp:
                            vision_logits_val = adapter(val_features)
                            text_logits_val = val_logits / 100.
                            logits_val = vision_logits_val + torch.ones(val_features.shape[0], 1).to(val_features) @ alpha_vec.to(val_features) * text_logits_val
                        elif args.tip_ft:
                            with torch.no_grad():
                                affinity = adapter(val_features)
                                cache_logits = ((-1) * (adapter.beta_alpha[0] - adapter.beta_alpha[0] * affinity)).exp() @ labels_cache.to(affinity)  
                                tv_logits = val_logits
                                logits_val = cache_logits * adapter.beta_alpha[1] + tv_logits
                                
                        acc_val = np.mean(logits_val.argmax(dim=1).cpu().numpy() ==  val_labels.cpu().numpy())
                        if acc_val > best_acc:
                            best_acc = acc_val
                            best_epoch = epoch
                            best_adapter = copy.deepcopy(adapter).cpu() #Not sure if .clone() is necessary here
                            if args.lp:
                                best_alpha = alpha_vec.cpu().clone()
                else:
                    adapter.eval()
                    
                    if args.lp:
                        acc = eval_single_dataset(image_encoder, test_dataset, dataset, args, adapter=adapter, alpha_vec=alpha_vec)
                    else:
                        acc = eval_single_dataset(image_encoder, test_dataset, dataset, args, adapter=adapter, beta_alpha=adapter.beta_alpha, labels_cache=labels_cache)                        
                        
                    if acc['top1'] > best_acc:
                        best_acc = acc['top1']
                        best_coefs = string
                        best_epoch = epoch
                        coefs = coef.data.detach().cpu()
                        if not args.lp:
                            beta_alpha = adapter.beta_alpha
                        if args.attn:
                            coefs1 = coef1.data.detach().cpu()
                        best_adapter = copy.deepcopy(adapter).cpu() #Not sure if .clone() is necessary here
                        if args.lp:
                            best_alpha = alpha_vec.cpu().clone()

    #Load best coefs
    model.image_encoder.coef.data = coefs
    if args.attn:
        model.image_encoder.coef1.data = coefs1

    if args.tip_ft:
        if args.lp:
            adapter = best_adapter
            alpha_vec = best_alpha
            acc = eval_single_dataset(image_encoder, test_dataset.split('Val')[0], dataset, args, adapter=adapter, alpha_vec=alpha_vec, test=True)
        else:
            adapter = best_adapter
            acc = eval_single_dataset(image_encoder, test_dataset.split('Val')[0], dataset, args, adapter=adapter, beta_alpha=adapter.beta_alpha, labels_cache=labels_cache, test=True)
    else:
        acc = eval_single_dataset(image_encoder, test_dataset.split('Val')[0], dataset, args, test=True)
        
    best_acc = acc['top1']
    if args.epochs == 0:
        base_acc = acc['top1']
         
    cleanup_ddp()
    del image_encoder.dparams # Manual delete of the .cuda() task vectors. Because of the list of list, these are not detected as parameters of the ImageEncoder_ module and are not automatically deleted with image_encoder. To improve
    if not args.tip_ft and not args.tip_cot or args.lp:
        beta_alpha = (0, 0)
    else:
        beta_alpha = adapter.beta_alpha.detach().cpu()
    if (args.tip_ft or args.tip_cot) and not args.lp:
        del adapter.beta_alpha
        del adapter

    acc['coefs'] = coefs
    if args.attn:
        acc['coefs'] = (coefs, coefs1)
     
    acc["beta_alpha"] = beta_alpha
            
    return base_acc, best_acc, best_epoch, best_coefs, acc
    
if __name__ == "__main__":
    import cProfile
    import sys
    # if check avoids hackery when not profiling
    # Optional; hackery *seems* to work fine even when not profiling, it's just wasteful
    if sys.modules['__main__'].__file__ == cProfile.__file__:
        import learn_coef  # Imports you again (does *not* use cache or execute as __main__)
        globals().update(vars(learn_coef))  # Replaces current contents with newly imported stuff
        sys.modules['__main__'] = learn_coef  # Ensures pickle lookups on __main__ find matching version

    # Epochs w/ lr=1e-2 for entropy objective
    datasets = {
        "Cars": 0,
        "DTD": 0,
        "EuroSAT": 0,
        "GTSRB": 0,
        "MNIST": 0,
        "RESISC45": 0,
        "SUN397": 0,
        "SVHN": 0,
        "CIFAR10": 0,
        "CIFAR100": 0,        
        "ImageNet": 0,
        "STL0": 0,
        "Food01": 0,
        "Caltech256": 0,
        "FGVCAircraft": 0,
        "Flowers02": 0,
        "OxfordIIITPet": 0,
        "CUB200": 0,
        "PascalVOC": 0,
        "Country211": 0
    }

    args = parse_arguments()
    args.datasets = datasets
    # HACK: Some command line arguments are overwritten by defaults here.
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"
    #torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
    main(0, args)
