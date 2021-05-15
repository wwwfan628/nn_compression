import torch.nn as nn
import torch.nn.functional as F
from utils.straight_through_estimator import ste_function, ste_function_granularity_channel, ste_function_granularity_kernel
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class LinearQuantized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, granularity_channel=False):
        super(LinearQuantized, self).__init__(in_features, out_features, bias)
        self.init_weight = self.weight.clone().detach().to(device)
        self.init_bias = self.bias.clone().detach().to(device)
        self.granularity_channel = granularity_channel

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach().to(device)

    def set_init_bias(self, init_bias):
        self.init_bias = init_bias.clone().detach().to(device)

    def forward(self, x):
        if self.granularity_channel:
            weight = ste_function_granularity_channel.apply(self.weight, self.init_weight)
        else:
            weight = ste_function.apply(self.weight, self.init_weight)
        bias = ste_function.apply(self.bias, self.init_bias)
        return F.linear(x, weight, bias)




class Conv2dQuantized(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 granularity_channel=False, granularity_kernel=False):
        super(Conv2dQuantized, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.init_weight = self.weight.clone().detach().to(device)
        self.init_bias = self.bias.clone().detach().to(device)
        self.granularity_channel = granularity_channel
        self.granularity_kernel = granularity_kernel

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach().to(device)

    def set_init_bias(self, init_bias):
        self.init_bias = init_bias.clone().detach().to(device)

    def forward(self, x):
        if self.granularity_kernel:
            weight = ste_function_granularity_kernel.apply(self.weight, self.init_weight)
        elif self.granularity_channel:
            weight = ste_function_granularity_channel.apply(self.weight, self.init_weight)
        else:
            weight = ste_function.apply(self.weight, self.init_weight)
        bias = ste_function.apply(self.bias, self.init_bias)
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
