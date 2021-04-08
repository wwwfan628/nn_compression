from typing import List, Optional
import math
import torch
from torch import Tensor
from scipy import stats
import numpy as np


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