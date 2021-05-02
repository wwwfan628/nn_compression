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


class ste_function_small(torch.autograd.Function):
    @staticmethod
    #def forward(ctx, input):
    #    return (input > 0).float()
    def forward(ctx, weight, init_weight):
        for idx_dim0, init_weight_dim0 in enumerate(init_weight):
            init_weight_dim0_tmp, _ = torch.sort(init_weight_dim0.view(-1))
            init_weight_dim0.view(-1)[torch.argsort(weight[idx_dim0].view(-1))] = init_weight_dim0_tmp
        return init_weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, torch.zeros(grad_output.shape)


class ste_function_extra_small(torch.autograd.Function):
    @staticmethod
    #def forward(ctx, input):
    #    return (input > 0).float()
    def forward(ctx, weight, init_weight):
        for idx_dim0, init_weight_dim0 in enumerate(init_weight):
            for idx_dim1, init_weight_dim1 in enumerate(init_weight_dim0):
                init_weight_dim1_tmp, _ = torch.sort(init_weight_dim1.view(-1))
                init_weight_dim1.view(-1)[torch.argsort(weight[idx_dim0, idx_dim1].view(-1))] = init_weight_dim1_tmp
        return init_weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, torch.zeros(grad_output.shape)