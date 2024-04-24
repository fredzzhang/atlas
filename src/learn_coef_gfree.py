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
from src.modeling import ImageEncoder, ImageClassifier, ImageEncoder_, LinearizedModel_
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.utils import cosine_lr, cosine_annealing_lr, adjust_lr, TwoTransform, TwoAsymetricTransform, adjust_lr_lp, extract_datasets, IndexWrapper
from src.sampler import TwoStreamBatchSampler, SubsetSampler

import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from lightning.fabric import Fabric
import minlora
import nevergrad as ng
from functools import partial

def l1_reg(coefs: torch.Tensor, lambda1: float=0.5) -> torch.Tensor:
    """ Simple L1 regularization. """
    l1_regularization = lambda1 * torch.norm(coefs, p=1, dim=0)
    return l1_regularization.mean()

def get_loss(weights, data_loader, model, loss_fn, preds, args):
    losses = 0
    tbar = tqdm(data_loader, disable=args.no_tqdm)
    for i, batch in enumerate(tbar):
        batch = maybe_dictionarize(batch, index=True)
        inputs = batch["images"]

        if 'simclr' in args.loss_fn:
            inputs2 = batch["images_aug"]

        if 'trusted' in args.loss_fn:
            ids = batch['index']

        new_coef = weights.reshape(model.image_encoder.coef.data.shape)
        model.image_encoder.coef.data = torch.from_numpy(new_coef).to(model.image_encoder.coef.data)

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
        elif 'trusted' in args.loss_fn:
            loss = loss_fn(logits, F.one_hot(preds.to(logits.device)[ids], num_classes=model.classification_head.out_features).float())

        if args.l1:
            loss += l1_reg(model.image_encoder.coef)
            
        tbar.set_description(f"Loss: {loss.item():.6f}")


        losses += loss.cpu().numpy()  * args.batch_size / len(inputs)
    return losses
        
def main(rank, args):

    # Load the individual task vectors.
    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB",
        "MNIST", "RESISC45", "SUN397", "SVHN", "CIFAR10",
        "CIFAR100", "ImageNet", "STL10", "Food101", "Caltech256",
        "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200",
        "PascalVOC", "Country211", "Caltech101", "UCF101"
    ]
    
    VIT_B_32_hugg= ['laion400m_e31', 'laion400m_e32', 'laion2b_e16', 'laion2b_s34b_b79k', 'datacomp_xl_s13b_b90k', 'datacomp_m_s128m_b4k', 'commonpool_m_clip_s128m_b4k', 'commonpool_m_laion_s128m_b4k', 'commonpool_m_image_s128m_b4k', 'commonpool_m_text_s128m_b4k', 'commonpool_m_basic_s128m_b4k', 'commonpool_m_s128m_b4k', 'datacomp_s_s13m_b4k', 'commonpool_s_clip_s13m_b4k', 'commonpool_s_laion_s13m_b4k', 'commonpool_s_image_s13m_b4k', 'commonpool_s_text_s13m_b4k', 'commonpool_s_basic_s13m_b4k',  'commonpool_s_s13m_b4k'] #'openai' (zero-shot base)
    
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
            finetuned_checkpoint = f"{args.save}/{dataset}Val/lora.pt"
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint, scale=args.scale, lora=args.lora)
        elif args.hugg:
            pretrained_checkpoint = f"{args.save}/CarsVal/zeroshot.pt"
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, scale=args.scale, hugg_checkpoint=dataset)            
        elif "randomtv" in dataset:
            pretrained_checkpoint = f"{args.save}/CarsVal/zeroshot.pt"
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

    args.rank = rank
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
        else:
            raise NotImplementedError("The current setting is not implemented")

    if args.fname is not None:
        fname = os.path.join(fname, f"results.txt")        
        if not args.merge:
            try:
                os.remove(fname)
                os.remove(fname.replace(".txt", ".pth"))
            except OSError:
                pass
        
    else:
        fname = os.path.join(fname, f"{datetime.today().strftime('%Y-%m-%d-%H-%M')}.txt")
        
    splt = fname.split('/')
    for i in range(len(splt)-1):
        if not os.path.exists('/'.join(splt[:i+1])):
            os.mkdir('/'.join(splt[:i+1]))

    fname2 = f"results/results_{args.loss_fn}{'_layerwise' if args.layerwise else '_global'}{'_optitv' if args.optimally_random else ''}{'_attn' if args.attn else ''}{'_semi'+str(args.semi) if args.semi else ''}{'_tipft' if args.tip_ft else ''}{'_tiponly' if args.tip_only else ''}{'_tipcot' if args.tip_cot else ''}{'_lp' if args.lp else ''}{'_select'+str(args.select_tvs) if args.select_tvs else ''}{'_memeff' if args.mem_eff else ''}{'_random' if args.random else ''}{'_worse' if args.worse else ''}{'_blockwisesel' if args.blockwise_select else ''}_{datetime.today().strftime('%Y-%m-%d-%H:%M')}.txt"
    
    if not args.no_log and not args.merge:
        with open(fname, 'a') as f: f.writelines([f"Task vector results for loss {args.loss_fn} on datasets {list(args.datasets.keys())}. {len(task_vectors.keys())} task vectors\n"])
        with open(fname2, 'a') as f: f.writelines([f"Task vector results for loss {args.loss_fn} on datasets {list(args.datasets.keys())}. {len(task_vectors.keys())} task vectors\n"])
    if args.merge:
        coef_dict = torch.load(fname.replace('.txt', '.pth'))
    else:
        coef_dict = {}
        
    
    for dataset in args.datasets:
        args.epochs = args.datasets[dataset]
        args.test_dataset = dataset
        print("=" * 100)
        print(f"Finetuning task vector coefficients of {args.model} on {dataset}")
        print("=" * 100)

        base_acc, best_acc, best_epoch, best_coef, acc = train(task_vectors, args)

        coef_dict[dataset] = {'coefs': acc["coefs"], 'preds':acc["softmax"], 'targets':acc["targets"]}
        print(acc.keys())
        if args.tip_ft:
            coef_dict[dataset]["adapter"] = acc["adapter"]
            if args.lp:
                coef_dict[dataset]["alpha_vec"] = acc["alpha_vec"]
            else:
                coef_dict[dataset]["beta_alpha"] = acc["beta_alpha"]
                
        if not args.no_log:
            torch.save(coef_dict, fname.replace('.txt', '.pth'))
            torch.save(coef_dict, fname2.replace('.txt', '.pth'))
            with open(fname, 'a') as f: f.writelines([f"{dataset}\n", f"Base accuracy {base_acc}, best accuracy {best_acc} at epoch {best_epoch}\n, tip beta {acc['beta_alpha'][0]:.3f} alpha {acc['beta_alpha'][1]:.3f}", f"{best_coef}\n"])
            with open(fname2, 'a') as f: f.writelines([f"{dataset}\n", f"Base accuracy {base_acc}, best accuracy {best_acc} at epoch {best_epoch}\n, tip beta {acc['beta_alpha'][0]:.3f} alpha {acc['beta_alpha'][1]:.3f}", f"{best_coef}\n"])
            
    if not args.no_log and len(args.datasets) > 0:
        with open(fname, 'a') as f: f.writelines([f"\nArguments {args}"])
        with open(fname2, 'a') as f: f.writelines([f"\nArguments {args}"])
    
@torch.no_grad()    
def train(task_vectors, args):
    scaler = torch.cuda.amp.GradScaler()
    #setup_ddp(args.rank, args.world_size, port=args.port)
    if args.loss_fn in ["simclr", "ssl_simclr_trusted", "entropy"]:#Test-time adaptations
        test_dataset = args.test_dataset
    else:
        test_dataset = args.test_dataset + 'Val'
    
    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    orig_dataset = test_dataset.split('Val')[0]
    pool = [k for k, v in task_vectors.items() if orig_dataset != k]
    task_vectors = [v for k, v in task_vectors.items() if orig_dataset != k]

    if args.finetuning_mode == "linear":
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = LinearizedModel_(image_encoder.model, task_vectors, device=args.rank)
    else:
        image_encoder = ImageEncoder(args)
        image_encoder = ImageEncoder_(image_encoder, task_vectors, args)
    
    classification_head = get_classification_head(args, test_dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    
    #TIP augmentations
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + model.val_preprocess.transforms[-3:])
        
    if 'simclr' in args.loss_fn or 'ssl' in args.loss_fn:
        preprocess_fn = TwoAsymetricTransform(model.val_preprocess, preprocess_fn)
    else:
        preprocess_fn = model.train_preprocess
    
    dataset = get_dataset(
        test_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    
    index_dataset = IndexWrapper(dataset.train_dataset)
        
    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    num_batches = len(data_loader)
    loss_fn = {
        'cross_entropy': torch.nn.CrossEntropyLoss(),
        'trusted': torch.nn.CrossEntropyLoss(),
    }[args.loss_fn]

    params = [p for p in model.parameters() if p.requires_grad]
    if args.finetuning_mode == "linear":
        coef = model.image_encoder.model.coef
    else:
        coef = model.image_encoder.coef
        
    best_acc = 0
    max_ep = args.epochs
    i = 0
    loss = torch.tensor(0)
    data_time = 0
    batch_time = 0
    epoch = 0

    fabric = Fabric(accelerator="cuda", devices=1, strategy="ddp", precision="16-mixed")
    fabric.launch()

    model = fabric.setup_module(model)
    data_loader = fabric.setup_dataloaders(data_loader, use_distributed_sampler=False)

    model.eval()
    if fabric.global_rank == 0:
        # We only need to evaluate the model on the first GPU.
        image_encoder = model.image_encoder
        if epoch == 0:
            #Accuracy of the Zero-shot on the test set
            acc = eval_single_dataset(image_encoder, test_dataset, dataset, args, test=True)
        elif args.loss_fn not in ["simclr", "ssl_simclr_trusted", "entropy"]:
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
            
        if epoch == 0:
            base_acc = acc['top1']  

        if ('trusted' in args.loss_fn) and epoch == 0:
            preprocess = data_loader.dataset.update_transforms(image_encoder.val_preprocess)

            targets = - torch.ones(len(data_loader.dataset), dtype=torch.long)
            preds = - torch.ones(len(data_loader.dataset), dtype=torch.long)
            with torch.no_grad():
                for i, batch in enumerate(tqdm(data_loader, disable=args.no_tqdm)):
                    batch = maybe_dictionarize(batch, index=True)
                    targets[batch["index"]] = batch["labels"].to(targets.device)
                    if i >= 1000:
                        print("Too much data, breaking ...")
                        break
                    
            k = args.semi                    
            to_keep = torch.tensor([], dtype=torch.long)
            unlabeled = torch.tensor([], dtype=torch.long)                    
            for c in range(classification_head.out_features):
                cond = (targets == c)
                ids_c = torch.arange(len(targets))[cond]
                a = torch.randperm(len(ids_c))
                to_keep = torch.cat((to_keep, ids_c[a[-k:]]))
                unlabeled = torch.cat((unlabeled, ids_c[a[:-k]]))
                
            #unlabeled = torch.tensor([u for u in unlabeled if confs[u] > 0], dtype=torch.long)
            #unlabeled = torch.unique(unlabeled)
            preds[to_keep] = targets[to_keep]

            data_loader.dataset.update_transforms(preprocess)
            print("Correctness")
            print((preds[to_keep] == targets[to_keep]).sum() / len(to_keep))
            print((preds[unlabeled] == targets[unlabeled]).sum() / len(unlabeled))
            
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
            data_loader = fabric.setup_dataloaders(data_loader, use_distributed_sampler=False)

    if args.layerwise:
        instrum = ng.p.Array(
            init=[0] * coef.shape[0]*coef.shape[1],
            upper=[1.5] * coef.shape[0]*coef.shape[1],
            lower=[-1.5] * coef.shape[0]*coef.shape[1],
        )
    else:
        instrum = ng.p.Array(
            init=[0] * len(task_vectors),
            upper=[1.5] * len(task_vectors),
            lower=[-1.5] * len(task_vectors),
        )
        
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=50)
            
    get_loss_partial = partial(get_loss, data_loader=data_loader, model=model, loss_fn=loss_fn, preds=preds, args=args)
    recommendation = optimizer.minimize(get_loss_partial, verbosity=1)

    image_encoder = model.image_encoder

    new_coef = recommendation.value.reshape(image_encoder.coef.data.shape)
    image_encoder.coef.data = torch.from_numpy(new_coef).to(image_encoder.coef.data)

    acc = eval_single_dataset(image_encoder, test_dataset.split('Val')[0], dataset, args, test=True)

    string = 'Coefficients:\t|'
    for c in coef.data:
        if args.layerwise:
            string += f"`{c.mean():.4f}`|"
        else:                    
            string += f"`{c:.4f}`|"
    print(string)
    best_epoch = 0
    coefs = image_encoder.coef.data.cpu()
    best_coefs = coefs
    
    acc["coefs"] = coefs    
    best_acc = acc['top1']         
    del image_encoder.dparams # Manual delete of the .cuda() task vectors. Because of the list of list, these are not detected as parameters of the ImageEncoder_ module and are not automatically deleted with image_encoder. To improve
    if args.blockwise_select:
        del updated_tvs
             
    acc["beta_alpha"] = (0, 0)
    acc["adapter"] = None
            
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

    args = parse_arguments()

    epochs = 10
    
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
        "UCF101": epochs
    }

    datasets = {
        "ImageNet": epochs,
    }
        
    # HACK: Some command line arguments are overwritten by defaults here.
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 32 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 4 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    
    if "RN" in args.model:#RN models do not have attention layers. Leaving attn as True will cause a problem with coef1 receiving no gradients.
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
        assert args.fname is not None, "args.fname needs to be specified"
        print(f"Merging with {os.path.join(args.fname, str(args.seed), 'results.txt')}")
        datasets_n = extract_datasets(os.path.join(args.fname, str(args.seed), "results.txt"))
        datasets = {k:v for k,v in datasets.items() if k not in datasets_n}
        
    args.datasets = datasets
    #torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
    seed = int(torch.exp(torch.tensor(args.seed)) * 3.1415 * 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    main(0, args)
