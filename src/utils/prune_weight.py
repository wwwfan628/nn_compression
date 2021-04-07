from typing import List, Optional
import math
import torch
from torch import Tensor
from scipy import stats
import numpy as np


@torch.no_grad()
def prune_weight(weight):
    mean = weight.mean().item()
    std = weight.std().item()
    min_value, max_value = stats.norm.interval(0.9, loc=mean, scale=std)
    weight[torch.logical_and(weight > min_value, weight < max_value)] = 0