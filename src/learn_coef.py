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
from src.eval import eval_single_dataset, eval_single_train_dataset
from src.heads import get_classification_head
from src.modeling import ImageEncoder, ImageClassifier
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.utils import cosine_lr, cosine_annealing_lr, adjust_lr, TwoTransform, TwoAsymetricTransform
from src.sampler import TwoStreamBatchSampler

import matplotlib.pyplot as plt

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

def ssl_loss(logits1: torch.Tensor, logits2: torch.Tensor, lam: float=None, index: torch.Tensor=None) -> torch.Tensor:
    #one_hot = torch.argmax(logits1.softmax(1), dim=1).detach()
    one_hot = logits1.softmax(1).detach()
    
    guessed_targets = one_hot ** (.5) #temp sharp    
    guessed_targets = guessed_targets / guessed_targets.sum(dim=1, keepdim=True) #normalization
   
    w, _ = logits1.softmax(1).max(1)
    w = (w > .8).to(logits2)
    if lam is not None:        
        return lam * F.cross_entropy(logits2, guessed_targets) + (1-lam) * F.cross_entropy(logits2, guessed_targets[index])
    return F.cross_entropy(logits2, guessed_targets) #(F.cross_entropy(logits2, one_hot, reduction='none') * w).sum() / w.sum() #

def l1_reg(coefs: torch.Tensor, lambda1: float=0.5):
    l1_regularization = lambda1 * torch.norm(coefs, p=1, dim=0)
    return l1_regularization.mean()
    
    

def ce_loss_trusted(logits1: torch.Tensor, logits2: torch.Tensor, preds: torch.FloatTensor=None, trusted: torch.BoolTensor=None, lam: float=None, index: torch.Tensor=None) -> torch.Tensor:
    trusted = trusted.to(logits1)
    return (F.cross_entropy(logits2, preds.to(logits2), reduction='none')*trusted).sum() / trusted.sum()
    #return F.cross_entropy(logits2, guessed_targets)#(F.cross_entropy(logits2, one_hot, reduction='none') * w).sum() / w.sum() 


def ssl_loss_trusted(logits1: torch.Tensor, logits2: torch.Tensor, preds: torch.FloatTensor=None, trusted: torch.BoolTensor=None, lam: float=None, index: torch.Tensor=None, thresh: float=0.99) -> torch.Tensor:
    #one_hot = torch.argmax(logits1.softmax(1), dim=1).detach()
    one_hot = logits1.softmax(1).detach()
    
    guessed_targets = one_hot ** (.5) #temp sharp    
    guessed_targets = guessed_targets / guessed_targets.sum(dim=1, keepdim=True) #normalization
    
    if trusted is not None:
        guessed_targets[trusted] = preds[trusted].to(guessed_targets)

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

def off_diagonal(x):
    """Extracts off-diagonal elements.                                                                                                                                                                                                                                       
    Args:
    X (torch.Tensor): batch of images in tensor format.
    Returns:
    torch.Tensor: flattened off-diagonal elements.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_loss(logits1: torch.Tensor, logits2: torch.Tensor, lam:float=10) -> torch.Tensor:
    """Barlow-twins non-contrastive loss """
    logits1, logits2 = F.normalize(logits1, 2), F.normalize(logits2, 2)
    corr_mat = logits1.T @ logits2
    on_diag_feat = (torch.diagonal(corr_mat).add(-1).pow(2).mean() * 0.5).sqrt()
    off_diag_feat = (off_diagonal(corr_mat).pow(2).mean() * 0.5).sqrt()
    feature_loss = (0.5 * on_diag_feat + 0.5 * off_diag_feat) * lam
    return feature_loss

def barlow_clr_loss(logits1: torch.Tensor, logits2: torch.Tensor, lam:float=10, mu:float=0.2) -> torch.Tensor:
    return simclr_loss(logits1, logits2, mu).mean() + barlow_loss(logits1, logits2, lam).mean()

def simclr_entro_loss(logits1: torch.Tensor, logits2: torch.Tensor, feats1: torch.Tensor, feats2: torch.Tensor, mu:float=0.2) -> torch.Tensor:
    """Simclr contrastive loss (Npairs for now) + entropy"""
    npairs_loss = simclr_loss(feats1, feats2, mu=mu)
    npairs_loss += softmax_entropy(logits1) #*.5 + softmax_entropy(logits2)*.5
    
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
    def __init__(self, model, task_vectors, device, layerwise=False) -> None:
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
        
        if self.layerwise:
            #self.coef = torch.nn.Parameter(torch.randn(len(task_vectors), len(self.params))/1000)
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
        else:
            #self.coef = torch.nn.Parameter(torch.rand(len(task_vectors),))
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors),))

    def __call__(self, x) -> torch.Tensor:
        if self.layerwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
        new_params = [dp + p for dp, p in zip(dparams, self.params)]
        return self.func(new_params, x)

class IndexWrapper(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
    def __getitem__(self, index):
        return self.dataset[index], index
    
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
        for i in range(args.add_random_tv):
            pool.append(f"randomtv_{i}")
    task_vectors = {}
    for i, dataset in enumerate(pool):
        if "randomtv" in dataset:
            pretrained_checkpoint = "checkpoints/ViT-B-32/CarsVal/zeroshot.pt"#f"{args.save}/{dataset}Val/zeroshot.pt"
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, scale=args.scale)
        elif args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors[dataset] = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        elif args.random_tv:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, scale=args.scale)
        elif args.optimally_random is not None:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            opt_tv = torch.load(os.path.join(args.optimally_random, f'tv_{i}.pth'))
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, vector=opt_tv["state_dict"], scale=args.scale)
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint, scale=args.scale)

    args.rank = rank
    fname = f"results/results_{args.loss_fn}{'_layerwise' if args.layerwise else '_global'}{'_randomtv' if args.random_tv else ''}{'_optitv' if args.optimally_random else ''}_{datetime.today().strftime('%Y-%m-%d-%H:%M')}.txt"
    with open(fname, 'a') as f: f.writelines([f"Task vector results for loss {args.loss_fn} on datasets {list(args.datasets.keys())}. {len(task_vectors.keys())} task vectors\n"])
    coef_dict = {}
    for dataset in args.datasets:
        args.epochs = args.datasets[dataset]
        args.test_dataset = dataset
        print("=" * 100)
        print(f"Finetuning task vector coefficients of {args.model} on {dataset}")
        print("=" * 100)

        base_acc, best_acc, best_epoch, best_coef, coefs = train(task_vectors, args)
        coef_dict[dataset] = coefs
        torch.save(coef_dict, fname.replace('.txt', '.pth'))
        with open(fname, 'a') as f: f.writelines([f"{dataset}\n", f"Base accuracy {base_acc}, best accuracy {best_acc} at epoch {best_epoch}\n", f"{best_coef}\n"])
    with open(fname, 'a') as f: f.writelines([f"\nArguments {args}"])
    
def train(task_vectors, args):
    scaler = torch.cuda.amp.GradScaler()
    setup_ddp(args.rank, args.world_size, port=args.port)
    test_dataset = args.test_dataset

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
        image_encoder = ImageEncoder_(image_encoder, task_vectors, device=args.rank, layerwise=args.layerwise)
    
    classification_head = get_classification_head(args, test_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()
        
    if 'simclr' in args.loss_fn:
        size = 224
        #[torchvision.transforms.RandomResizedCrop(size, antialias=True)] + \
        #torchvision.transforms.RandomHorizontalFlip(p=0.5)
        #torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #torchvision.transforms.RandomGrayscale(p=0.2)] + \
            
        preprocess_fn = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(256, antialias=True),
             torchvision.transforms.RandomCrop(size)] + \
            model.val_preprocess.transforms[-3:]
        )

        preprocess_fn = TwoAsymetricTransform(model.val_preprocess, preprocess_fn)
    else:
        preprocess_fn = model.val_preprocess

    dataset = get_dataset(
        test_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    if args.trainset:
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
        torch.random.manual_seed(torch.exp(torch.tensor(1)).item())
        subset = torch.randperm(len(data_loader.dataset))[:len(dataset.test_dataset)]
        subset.sort()
        subset_s = torch.utils.data.SubsetRandomSampler(subset)
        index_dataset = IndexWrapper(data_loader.dataset)
        data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, sampler=subset_s, num_workers=16)
    else:
        subset = torch.arange(len(data_loader.dataset))
        data_loader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
        index_dataset = IndexWrapper(data_loader.dataset)
        data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        

    # Override shuffle to True
    data_loader.shuffle = True
    num_batches = len(data_loader)
        
    # Distribute the data and model across the GPUs.
    ddp_loader = data_loader#distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.rank],
        find_unused_parameters=False,
        output_device=args.rank,
    )
    
    loss_fn = {
        'entropy': softmax_entropy,
        'cb_entropy': cb_softmax_entropy,
        'cross_entropy': torch.nn.CrossEntropyLoss(),
        'trusted': torch.nn.CrossEntropyLoss(),
        'entro_trusted': softmax_entropy,
        'simclr': simclr_loss,
        'ssl_simclr': ssl_loss,
        'ssl_simclr_trusted': ssl_loss_trusted,
        'ssl_simclr_trusted_entro': ssl_loss_trusted,
        'simclr_mixup': simclr_mixup_loss,
        'simclr_entro': simclr_entro_loss,
        'barlow_simclr': barlow_loss,
    }[args.loss_fn]

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    #optimizer = torch.optim._multi_tensor.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)

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
        coef = ddp_model.module.image_encoder.model.coef
    else:
        coef = ddp_model.module.image_encoder.coef

    ddp_model.eval()
    best_acc = 0
    for epoch in range(args.epochs):
        # Evaluate before each epoch
        if args.lr_scheduler=="annealing":
            adjust_lr(optimizer, lrs[epoch])

        if is_main_process():
            # We only need to evaluate the model on the first GPU.
            image_encoder = ddp_model.module.image_encoder
            #with torch.autocast(device_type="cuda"):
            acc = eval_single_dataset(image_encoder, test_dataset, args)
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
            if epoch == 0:
                base_acc = acc['top1']
                
            if ('trusted' in args.loss_fn and ((epoch >= 0 and 'entro' not in args.loss_fn) or epoch >= 1)):
                #with torch.autocast(device_type="cuda"):
                acc = eval_single_train_dataset(image_encoder, test_dataset, data_loader, args)
                
                confs = acc['conf']
                preds = acc['preds']
                targets = acc['targets']
                full_preds = acc['full_preds']

                if False:
                    to_keep = torch.arange(len(confs))[confs > .8]
                    unlabeled = torch.arange(len(confs))[(confs < .8) * (confs > 0)]
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
                    sampler = torch.utils.data.SubsetRandomSampler(to_keep)                    
                    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=16)                    
                num_batches = len(data_loader)
                
                # Distribute the data and model across the GPUs.
                #ddp_loader = distribute_loader(data_loader)
                ddp_loader = data_loader

        if "ssl" in args.loss_fn:
            threshs = torch.arange(args.epochs) / args.epochs / 10 + .9
            #threshs = (1-torch.tensor(cosine_annealing_lr(1, 0, args.epochs))) / 10 + .9
            
                
        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )
            batch = maybe_dictionarize(batch, index=True)
            inputs = batch["images"].cuda()
            
            if 'simclr' in args.loss_fn:
                inputs2 = batch["images_aug"].cuda()
            if 'trusted' in args.loss_fn:
                ids = batch['index']

            data_time = time.time() - start_time

            if 'mixup' in args.loss_fn:
                lam = np.random.beta(1, 1)
                lam = max(lam, 1-lam)
                index = torch.randperm(len(inputs))
                inputs = lam * inputs + (1-lam) * inputs[index]
            with torch.autocast(device_type="cuda"):
                if 'ssl' in args.loss_fn and 'entro' not in args.loss_fn:
                    with torch.no_grad():
                        logits = ddp_model(inputs)
                elif 'simclr' in args.loss_fn:
                    logits, feats = ddp_model(inputs, return_features=True)
                else:
                    logits = ddp_model(inputs)

                if args.loss_fn == 'cross_entropy':
                    labels = batch["labels"].cuda()
                    loss = loss_fn(logits, labels)
                elif 'entro_conf' in args.loss_fn:
                    loss = loss_fn(logits).mean()
                elif 'ssl_simclr_trusted' in args.loss_fn:

                    lam = np.random.beta(1, 1)
                    lam = max(lam, 1-lam)
                    index = torch.randperm(len(inputs2))
                    #inputs2 = lam * inputs2 + (1-lam) * inputs2[index]
                    if epoch == 0 and 'entro' in args.loss_fn:
                        #logits2, feats2 = ddp_model(inputs2, return_features=True)
                        #loss = simclr_loss(feats, feats2).mean()
                        loss = softmax_entropy(logits).mean()
                        #loss = F.cross_entropy(logits, labels.float())
                        #labels = batch["labels"].cuda()
                        #labels = F.one_hot(labels, num_classes=classification_head.out_features)
                    elif epoch < 2 and False:
                        logits2 = ddp_model(inputs2)
                        trusted = torch.ones(args.batch_size).bool()
                        trusted[:args.batch_size//2] = 0 #First half of the batch are untrusted labels
                        loss = ce_loss_trusted(logits, logits2, preds[ids], trusted)#, lam, index)
                    else:
                        logits2 = ddp_model(inputs2)
                        trusted = torch.ones(args.batch_size).bool()
                        trusted[:args.batch_size//2] = 0 #First half of the batch are untrusted labels
                        loss = loss_fn(logits, logits2, preds[ids], trusted, thresh=threshs[epoch])#, lam, index)
                        #loss = softmax_entropy(logits).mean()

                elif 'trusted' in args.loss_fn:
                    loss = loss_fn(logits, preds[ids].to(logits))
                elif 'ssl' in args.loss_fn:
                    lam = np.random.beta(1, 1)
                    lam = max(lam, 1-lam)
                    index = torch.randperm(len(inputs2))
                    #inputs2 = lam * inputs2 + (1-lam) * inputs2[index]
                    logits2 = ddp_model(inputs2)
                    loss = loss_fn(logits, logits2, preds)#, lam, index)
                elif 'simclr' in args.loss_fn:
                    logits2, feats2 = ddp_model(inputs2, return_features=True)
                    if 'entro' in args.loss_fn:
                        loss = loss_fn(logits, logits2, feats, feats2).mean()
                    elif 'mixup' in args.loss_fn:
                        loss = loss_fn(logits, logits2, lam, index).mean()
                    else:
                        loss = loss_fn(feats, feats2).mean()                    
                else:
                    loss = loss_fn(logits).mean(0)
                    
                if args.l1:
                    loss += l1_reg(ddp_model.module.image_encoder.coef)

                    
            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                if not args.lr_scheduler=="annealing":
                    scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % args.print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{len(data_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\tLr {optimizer.param_groups[0]['lr']:.3f}",  # noqa: E501
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
        #with torch.autocast(device_type="cuda"):
        acc = eval_single_dataset(image_encoder, test_dataset, args)
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
            
        print(string)

    cleanup_ddp()
    del image_encoder.dparams # Manual delete of the .cuda() task vectors. Because of the list of list, these are not detected as parameters of the ImageEncoder_ module and are not automatically deleted with image_encoder. To improve
    return base_acc, best_acc, best_epoch, best_coefs, coefs
    


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
        "Cars": 10,
        "DTD": 10,
        "EuroSAT": 10,
        "GTSRB": 10,
        "MNIST": 10,
        "RESISC45": 10,
        "SUN397": 10,
        "SVHN": 10,
        "CIFAR10": 10,
        "CIFAR100": 10,
        "ImageNet": 10,
        "STL10": 10,
        "Food101": 10,
        "Caltech256": 10,
        "FGVCAircraft": 10,
        "Flowers102": 10,
        "OxfordIIITPet": 10,
        "CUB200": 10,
        "PascalVOC": 10,
        "Country211": 10
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
