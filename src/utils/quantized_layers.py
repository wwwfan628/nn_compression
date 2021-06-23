import torch.nn as nn
import torch.nn.functional as F
from utils.straight_through_estimator import ste_function, ste_function_granularity_channel, ste_function_granularity_kernel
import torch


class LinearQuantized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearQuantized, self).__init__(in_features, out_features, bias)
        self.init_weight = self.weight.clone().detach()
        self.init_bias = self.bias.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()
            self.init_bias = self.init_bias.cuda()

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()

    def set_init_bias(self, init_bias):
        self.init_bias = init_bias.clone().detach()
        if torch.cuda.is_available():
            self.init_bias = self.init_bias.cuda()

    def forward(self, x):
        weight = ste_function.apply(self.weight, self.init_weight.clone().detach().cuda())
        bias = ste_function.apply(self.bias, self.init_bias.clone().detach().cuda())
        return F.linear(x, weight, bias)


class LinearQuantized_granularity_channel(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearQuantized_granularity_channel, self).__init__(in_features, out_features, bias)
        self.init_weight = self.weight.clone().detach()
        self.init_bias = self.bias.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()
            self.init_bias = self.init_bias.cuda()

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()

    def set_init_bias(self, init_bias):
        self.init_bias = init_bias.clone().detach()
        if torch.cuda.is_available():
            self.init_bias = self.init_bias.cuda()

    def forward(self, x):
        weight = ste_function_granularity_channel.apply(self.weight, self.init_weight.clone().detach().cuda())
        bias = ste_function.apply(self.bias, self.init_bias.clone().detach().cuda())
        return F.linear(x, weight, bias)


class Conv2dQuantized(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dQuantized, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.init_weight = self.weight.clone().detach()
        self.init_bias = self.bias.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()
            self.init_bias = self.init_bias.cuda()

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()

    def set_init_bias(self, init_bias):
        self.init_bias = init_bias.clone().detach()
        if torch.cuda.is_available():
            self.init_bias = self.init_bias.cuda()

    def forward(self, x):
        weight = ste_function.apply(self.weight, self.init_weight.clone().detach().cuda())
        bias = ste_function.apply(self.bias, self.init_bias.clone().detach().cuda())
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dQuantized_granularity_channel(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dQuantized_granularity_channel, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                                  padding, dilation, groups, bias)
        self.init_weight = self.weight.clone().detach()
        self.init_bias = self.bias.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()
            self.init_bias = self.init_bias.cuda()

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()

    def set_init_bias(self, init_bias):
        self.init_bias = init_bias.clone().detach()
        if torch.cuda.is_available():
            self.init_bias = self.init_bias.cuda()

    def forward(self, x):
        weight = ste_function_granularity_channel.apply(self.weight, self.init_weight.clone().detach().cuda())
        bias = ste_function.apply(self.bias, self.init_bias.clone().detach().cuda())
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dQuantized_granularity_kernel(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dQuantized_granularity_kernel, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                                 padding, dilation, groups, bias)
        self.init_weight = self.weight.clone().detach()
        self.init_bias = self.bias.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()
            self.init_bias = self.init_bias.cuda()

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach()
        if torch.cuda.is_available():
            self.init_weight = self.init_weight.cuda()

    def set_init_bias(self, init_bias):
        self.init_bias = init_bias.clone().detach()
        if torch.cuda.is_available():
            self.init_bias = self.init_bias.cuda()

    def forward(self, x):
        weight = ste_function_granularity_kernel.apply(self.weight, self.init_weight.clone().detach().cuda())
        bias = ste_function.apply(self.bias, self.init_bias.clone().detach().cuda())
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
