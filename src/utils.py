import os
import pickle

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


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


def get_logits(inputs, classifier, **kwargs):
    assert callable(classifier)
    if hasattr(classifier.classification_head, "to"):
        classifier.classification_head = classifier.classification_head.to(inputs.device)
    return classifier(inputs, **kwargs)


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

def cosine_annealing_lr(init_lr, stage1, epochs):
    lrs = [init_lr] * epochs
    init_lr_stage_ldl = init_lr
    for t in range(stage1, epochs):
        lrs[t] = 0.5 * init_lr_stage_ldl * (1 + math.cos((t - stage1 + 1) * math.pi / (epochs - stage1 + 1)))
    return lrs

def adjust_lr(optimizer, lr, param_groups=None):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
        if param_groups is not None:
            param_group['lr'] *= param_groups[i]
            
def adjust_lr_lp(optimizer, lr, lr_lp, param_groups=None):
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 1:
            param_group['lr'] = lr_lp
        else:
            param_group['lr'] = lr
        

        
class TwoTransform:
    """Create two transforms of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x, *args, **kwargs):
        return [self.transform(x, *args, **kwargs), self.transform2(x, *args, **kwargs)]
    
class TwoAsymetricTransform:
    """Create two asymetrics transforms of the same image"""

    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2
 
    def __call__(self, x, *args, **kwargs):
        return [self.transform(x, *args, **kwargs), self.transform2(x, *args, **kwargs)]
   
class MetaAdapter(nn.Module):
    def __init__(self, dim=1024, num_heads=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.alpha_proj = nn.Linear(dim, 1, bias=True)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, 1)

    def forward(self, query, key, value):
        B, K, C = key.shape
        res = query

        query = query.reshape(B, 1, C)
        key = torch.cat([query, key], dim=1)
        value = torch.cat([query, value], dim=1)
        query = self.q_proj(query).reshape(B, self.num_heads, C)
        key = self.k_proj(key)

        query = query.reshape(B, self.num_heads, 1, -1).permute(0, 2, 1, 3)
        key = key.reshape(B, K + 1, 1, -1).permute(0, 2, 1, 3)
        value = value.reshape(B, K + 1, 1, -1).permute(0, 2, 1, 3)

        attn_weight = (query @ key.transpose(-1, -2) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float))).softmax(-1)
        attn = attn_weight @ value

        alpha = torch.nn.functional.sigmoid(self.alpha_proj(res).reshape(B, -1, 1, 1))
        attn = (alpha * attn).squeeze()

        attn = res + attn
        attn = F.normalize(attn, p=2, dim=-1)
        return attn

def extract_datasets(f):
    datasets = []
    with open(f, 'r') as fi: lines=fi.readlines()
    for l in lines[1:]:
        datasets.append(l.split(' ')[0].replace('\n',''))
    return datasets



def _select_lora(lora_layer, index):
    lora_layer.lora_A = lora_layer.lora_As[index]
    lora_layer.lora_B = lora_layer.lora_Bs[index]


def select_lora(model, index):
    model.apply(apply_to_lora(lambda x: _select_lora(x, index)))
    return model
