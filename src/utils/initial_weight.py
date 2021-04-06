from typing import List, Optional
import math
import torch
from torch import Tensor
from scipy import stats
import numpy as np


@torch.no_grad()
def prune_weighht(weight, min_value, max_value):
        weight[weight<=min_value] = 0
        weight[weight>=max_value] = 0


@torch.no_grad()
def compute_conf_interval(weight):
    mean = weight.mean().item()
    std = weight.std().item()
    min_value, max_value = stats.norm.interval(0.5, loc=mean, scale=std)
    return min_value, max_value