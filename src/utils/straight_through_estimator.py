import torch
import torch.nn.functional as F

class ste_function(torch.autograd.Function):
    @staticmethod
    #def forward(ctx, input):
    #    return (input > 0).float()
    def forward(ctx, weight, init_weight):
        init_weight_tmp, _ = torch.sort(init_weight.view(-1))
        init_weight.view(-1)[torch.argsort(weight.view(-1))] = init_weight_tmp
        return init_weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, torch.zeros(grad_output.shape)