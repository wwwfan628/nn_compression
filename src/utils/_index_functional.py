from typing import List, Optional
import math
import torch
from torch import Tensor
from .bipartite_matching import bipartite_perfect_matching

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def index_sgd(params: List[Tensor], d_p_list: List[Tensor], momentum_buffer_list: List[Optional[Tensor]],
              weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool, weighted_reconstruction: bool):
    """
    Functional API that performs Index SGD algorithm computation.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        #param.add_(d_p, alpha=-lr)  # sgd, from pytorch original code
        if weighted_reconstruction:
            # create mask
            k = min(int(len(param.view(-1)) * 0.05), 0)
            all_idx = torch.arange(len(param.view(-1))).to(device)
            largest_k_idx = torch.topk(d_p_list[i].view(-1), k=k)[1]
            sort_idx = torch.tensor(list(set(all_idx.clone().detach().cpu().numpy()).difference(
                set(largest_k_idx.clone().detach().cpu().numpy())))).to(device)
            # compute new weight values
            param_new = param.add(d_p, alpha=-lr)
            # find matching for the largest k weights
            largest_k_idx_new = reorder_largest_k(param_new.clone().detach().view(-1)[largest_k_idx], param.clone().detach().view(-1))
            param_copy = param.clone().detach()
            param.view(-1)[largest_k_idx] = param_copy.view(-1)[largest_k_idx_new]
            # create mask, delete those parameters which have been used in last step
            sort_idx_new = torch.tensor(list(set(all_idx.clone().detach().cpu().numpy()).difference(
                set(largest_k_idx_new.clone().detach().cpu().numpy())))).to(device)
            # sort the rest weight
            param_sort_tmp, _ = torch.sort(param.view(-1)[sort_idx_new])
            param.view(-1)[sort_idx[torch.argsort(param_new.view(-1)[sort_idx])]] = param_sort_tmp
        else:
            param_new = param.add(d_p, alpha=-lr)
            param_tmp, _ = torch.sort(param.view(-1))
            param.view(-1)[torch.argsort(param_new.view(-1))] = param_tmp


def index_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor], state_steps: List[int], amsgrad: bool, beta1: float, beta2: float, lr: float,
         weight_decay: float, eps: float, weighted_reconstruction: bool):
    """
    Functional API that performs Adam algorithm computation.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1
        # param.addcdiv_(exp_avg, denom, value=-step_size)   # adam, from pytorch original code
        if weighted_reconstruction:
            # create mask
            k = min(int(len(param.view(-1))*0.05), 50)
            all_idx = torch.arange(len(param.view(-1))).to(device)
            largest_k_idx = torch.topk(grads[i].view(-1), k=k)[1]
            #combined = torch.cat((all_idx, largest_k_idx))
            #uniques, counts = combined.unique(return_counts=True)
            #sort_idx = uniques[counts == 1]
            sort_idx = torch.tensor(list(set(all_idx.clone().detach().cpu().numpy()).difference(
                set(largest_k_idx.clone().detach().cpu().numpy())))).to(device)
            # compute new weight values
            param_new = param.addcdiv(exp_avg, denom, value=-step_size)
            # find matching for the largest k weights
            largest_k_idx_new = reorder_largest_k(param_new.clone().detach().view(-1)[largest_k_idx], param.clone().detach().view(-1))
            param_copy = param.clone().detach()
            param.view(-1)[largest_k_idx] = param_copy.view(-1)[largest_k_idx_new]
            # create mask, delete those parameters which have been used in last step
            #combined = torch.cat((all_idx, largest_k_idx_new))
            #uniques, counts = combined.unique(return_counts=True)
            #sort_idx_new = uniques[counts == 1]
            sort_idx_new = torch.tensor(list(set(all_idx.clone().detach().cpu().numpy()).difference(
                set(largest_k_idx_new.clone().detach().cpu().numpy())))).to(device)
            # sort the rest weight
            param_sort_tmp, _ = torch.sort(param.view(-1)[sort_idx_new])
            param.view(-1)[sort_idx[torch.argsort(param_new.view(-1)[sort_idx])]] = param_sort_tmp
        else:
            param_new = param.addcdiv(exp_avg, denom, value=-step_size)
            param_tmp, _ = torch.sort(param.view(-1))
            param.view(-1)[torch.argsort(param_new.view(-1))] = param_tmp


@torch.no_grad()
def reorder_largest_k(param_new, param_orig):
    largest_k_idx_new = torch.ones(len(param_new), dtype=int).to(device) * len(param_orig)
    for i in range(len(param_new)):
        # create matrix for param_new & param_orig
        param_new_i = torch.ones(len(param_orig)).to(device) * param_new[i]
        diff_abs_i = (param_new_i - param_orig).abs()
        min_idx_i = torch.argmin(diff_abs_i)
        largest_k_idx_new[i] = min_idx_i
        param_orig[min_idx_i] = float('inf')
    return largest_k_idx_new