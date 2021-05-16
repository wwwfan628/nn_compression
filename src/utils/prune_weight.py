from typing import List, Optional
import math
import torch
from torch import Tensor
from scipy import stats
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


@torch.no_grad()
def prune_weight_interval(weight):
    mean = weight.mean().item()
    std = weight.std().item()
    min_value, max_value = stats.norm.interval(0.9, loc=mean, scale=std)
    weight[torch.logical_and(weight > min_value, weight < max_value)] = 0



@torch.no_grad()
def prune_weight_abs(weight, amount=0.9):
    thr = (len(weight.view(-1))-1) * amount
    weight.view(-1)[torch.argsort(weight.abs().view(-1))<thr] = 0



@torch.no_grad()
def prune_weight_abs_all_layers(params, amount=0.9):
    params = list(params)
    params_abs_flatten = np.zeros(0)
    params_shape = []
    params_flatten_len = []
    for param in params:
        params_abs_flatten = np.append(params_abs_flatten, param.abs().view(-1).clone().detach().cpu())
        params_shape.append(param.shape)
        params_flatten_len.append(len(param.view(-1)))
    params_abs_flatten = torch.Tensor(params_abs_flatten).to(device)
    k = int(len(params_abs_flatten) * (1-amount))
    idx_topk = torch.topk(params_abs_flatten, k=k)[1]
    mask_flatten = torch.zeros(params_abs_flatten.shape).to(device)
    mask_flatten[idx_topk] = 1
    for i, param in enumerate(params):
        if i == 0:
            mask = mask_flatten[:params_flatten_len[i]].reshape(params_shape[i])
            param.mul_(mask)
        else:
            mask = mask_flatten[params_flatten_len[i-1]:params_flatten_len[i-1]+params_flatten_len[i]].reshape(params_shape[i])
            param.mul_(mask)