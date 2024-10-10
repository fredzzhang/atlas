import os
import pickle

import numpy as np
import torch
import tqdm
from src.datasets.common import maybe_dictionarize
from torch.utils.data.sampler import BatchSampler
import itertools

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]

def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier

def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)

def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu")
    if device is not None:
        model = model.to(device)
    return model

def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)

def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)

class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_optimal_coef(
    results,
    metric="avg_normalized_top1",
    minimize=False,
    control_metric=None,
    control_metric_threshold=0.0,
):
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef


def nonlinear_advantage(nonlinear_acc, linear_acc, num_classes):
    """Computes the normalized non-linear advantage of a finetuned model.

    The nonlinear_advantage is defined as:
        error_rate(linear_model) - error_rate(nonlinear_model) / (1 - 1 / num_classes)
    and takes values between [-1, 1]. A value of 0 indicates that the nonlinear
    model is no better than the linear one. Meanwhile, a value of 1 indicates
    that the nonlinear model is perfect and the linear trivial, and a value of
    -1 indicates the opposite.
    """
    return (nonlinear_acc - linear_acc) / (1.0 - 1.0 / num_classes)

class IndexWrapper(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
    def __getitem__(self, index):
        instance = self.dataset[index]
        if isinstance(instance, dict):
            instance["index"] = index
            return instance
        return *instance, index
    
    def __len__(self):
        return len(self.dataset)
    
def get_n_shots(dataset, shots, n_class, args):
    index_dataset = IndexWrapper(dataset)
    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    targets = - torch.ones(len(dataset), dtype=torch.long)
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            batch = maybe_dictionarize(batch)
            targets[batch["index"]] = batch["labels"].to(targets.device)
            if i >= 1000:
                print("Too much data, breaking ...")
                break
            
    to_keep = torch.tensor([], dtype=torch.long)
    for c in range(n_class):
        cond = (targets == c)
        ids_c = torch.arange(len(targets))[cond]
        a = torch.randperm(len(ids_c))
        to_keep = torch.cat((to_keep, ids_c[a[-shots:]]))
        
    return to_keep

def get_preds(dataset, model, args):
    index_dataset = IndexWrapper(dataset)
    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    all_preds = - torch.ones((len(dataset), model.module.classification_head.out_features))
    trusted = torch.zeros(len(dataset), dtype=torch.bool)
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            batch = maybe_dictionarize(batch)
            preds = model(batch["images"].cuda())
            all_preds[batch["index"]] = torch.nn.functional.softmax(preds, dim=-1).to(all_preds)
    return all_preds


class TIPWrapper(torch.nn.Module):
    def __init__(self, model, features_cache, labels):
        super().__init__()
        for p in model.parameters():
            p.requires_grad = False    
        self.model = model
        
        features_cache = features_cache.permute(1, 0).detach() #Just in case
        self.adapter = torch.nn.Linear(features_cache.shape[0], features_cache.shape[1], bias=False)
        self.adapter.weight.data = features_cache.t()
        self.beta_alpha = torch.nn.Parameter(torch.tensor([1.,2.]))
        self.labels = torch.nn.functional.one_hot(labels.long())
        print("Num classes", self.model.classification_head.weight.shape[0])

    def forward(self, x, tv_logits=None, feats=None):
        if tv_logits is None:
            tv_logits, feats = self.model(x, return_features=True)
        
        affinity = self.adapter(feats)
        cache_logits = ((-1) * (self.beta_alpha[0] - self.beta_alpha[0] * affinity)).exp() @ self.labels.to(affinity)
        logits = cache_logits * self.beta_alpha[1] + tv_logits
        return logits
    
class LPPWrapper(torch.nn.Module):
    def __init__(self, model, features_cache, labels, shots):
        super().__init__()
        for p in model.parameters():
            p.requires_grad = False
            
        self.model = model        
        from src.lpplusplus import init_lp
        self.adapter, self.alpha_vec, self.lr_alpha, self.lr_temp = init_lp(features_cache, labels, self.model.classification_head.weight.T / 100., shots)

    def forward(self, x, tv_logits=None, feats=None):
        if tv_logits is None:
            tv_logits, feats = self.model(x, return_features=True)
            
        vision_logits = self.adapter(feats)
        logits = vision_logits + torch.ones(feats.shape[0], 1).to(feats) @ self.alpha_vec.to(feats) * tv_logits / 100
        return logits
    
class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler, epochs):
        self.sampler = sampler
        self.epochs = epochs

    def __iter__(self):
        for _ in range(self.epochs):
            yield from iter(self.sampler)

    def __len__(self):
        return self.epochs * len(self.sampler)

    
def iterate_once(iterable):
   
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class TwoStreamBatchSampler(BatchSampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.inter_batch_size = 3 * batch_size // 4
        self.batch_size = batch_size

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, 3*self.batch_size // 4),
                    grouper(secondary_iter,  self.batch_size // 4))
        )

    def __len__(self):
        return len(self.primary_indices) // self.inter_batch_size
    
class TwoAsymetricTransform:
    """Create two asymetrics transforms of the same image"""

    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2
 
    def __call__(self, x, *args, **kwargs):
        return [self.transform(x, *args, **kwargs), self.transform2(x, *args, **kwargs)]
