from typing import List, Optional
import math
import torch
from torch import Tensor
from .bipartite_matching import bipartite_perfect_matching


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
            param_new = param.add(d_p, alpha=-lr)
            sequence = bipartite_perfect_matching(param_new.data.view(-1), param.data.view(-1), d_p_list[i].data.view(-1).abs())
            param.view(-1)[:] = param.view(-1)[sequence]
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
        #param.addcdiv_(exp_avg, denom, value=-step_size)   # adam, from pytorch original code
        if weighted_reconstruction:
            param_new = param.addcdiv(exp_avg, denom, value=-step_size)
            sequence = bipartite_perfect_matching(param_new.data.view(-1), param.data.view(-1), grads[i].data.view(-1).abs())
            param.view(-1)[:] = param.view(-1)[sequence]
        else:
            param_new = param.addcdiv(exp_avg, denom, value=-step_size)
            param_tmp, _ = torch.sort(param.view(-1))
            param.view(-1)[torch.argsort(param_new.view(-1))] = param_tmp
