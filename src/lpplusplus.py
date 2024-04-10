import torch
from scipy.linalg import eigh
import torch.nn as nn

def calculate_lr_alpha(features, clip_weights):
    # lr_alpha
    ftT = features @ clip_weights.to(features)
    temp = torch.sum(torch.pow(ftT, 2),dim = 0)
    max_sum = max(temp)
    lr_alpha = features.shape[0] / (max_sum * 4)
    return lr_alpha

def calculate_init_alpha(features, labels, shots, clip_weights):
    # init_alpha
    alpha_tilde = compute_centroids_alpha((features @ clip_weights.to(features)).unsqueeze(0), labels.unsqueeze(0))[0]
    alpha_tilde = alpha_tilde.double() * shots
    alpha_init = 250 / shots * alpha_tilde
    final_init_alpha_mean = torch.mean(alpha_init)
    return final_init_alpha_mean

def calculate_lr_w(features):
    # lr_w
    ff_t = features.T @ features
    ff_t_np = ff_t.cpu().numpy()
    w, v = eigh(ff_t_np)
    max_eigen = max(w) # check the iters of power iteration
    lr_w =  (4 * features.shape[0]) / max_eigen
    return lr_w

def get_one_hot(y_s: torch.tensor, num_classes: int):
    """
        args:
            y_s : torch.Tensor of shape [n_task, shot]
        returns
            y_s : torch.Tensor of shape [n_task, shot, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    one_hot = torch.zeros(one_hot_size, device=y_s.device)

    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)
    return one_hot

def compute_centroids_alpha(z_s: torch.tensor,
                      y_s: torch.tensor):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0)).to(z_s)
    centroids = (one_hot*z_s/ one_hot.sum(-2, keepdim=True)).sum(1)  # [batch, K, d]
    return centroids


def compute_centroids(z_s: torch.tensor,
                      y_s: torch.tensor):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0)).transpose(1, 2)
    # centroids = one_hot.bmm(z_s) / one_hot.sum(-1, keepdim=True)  # [batch, K, d]
    centroids = one_hot.bmm(z_s)  # [batch, K, d]
    return centroids

def init_lp(features, labels, text_weights, shot):
    centroids = compute_centroids(features.unsqueeze(0), labels.unsqueeze(0))  # [batch, num_class, d]                        

    num_classes = text_weights.shape[1]
    classifier = nn.Linear(features.shape[1], num_classes,bias=True).to(features)
    classifier.weight.data = centroids[0]

    # lr_w
    lr_temp = calculate_lr_w(features)

    # init_alpha
    final_init_alpha_mean = calculate_init_alpha(features, labels, shot, text_weights)

    alpha_vec = torch.autograd.Variable(final_init_alpha_mean * torch.ones(1, num_classes).to(features), requires_grad=True)

    # lr_alpha
    lr_alpha = calculate_lr_alpha(features, text_weights)

    print('final_init_alpha_mean: {}'.format(final_init_alpha_mean))

    print('Calculated lr_temp, lr_alpha:', lr_temp, lr_alpha)

    return classifier, alpha_vec, lr_alpha, lr_temp
