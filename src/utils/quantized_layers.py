import torch.nn as nn
import torch.nn.functional as F
from utils.straight_through_estimator import ste_function, ste_function_small, ste_function_extra_small


class LinearQuantized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, small=False):
        super(LinearQuantized, self).__init__(in_features, out_features, bias)
        self.init_weight = self.weight.clone().detach()
        self.init_bias = self.bias.clone().detach()
        self.small = small

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach()

    def set_init_bias(self, init_bias):
        self.init_bias = init_bias.clone().detach()

    def forward(self, x):
        if self.small:
            weight = ste_function_small.apply(self.weight, self.init_weight)
        else:
            weight = ste_function.apply(self.weight, self.init_weight)
        bias = ste_function.apply(self.bias, self.init_bias)
        return F.linear(x, weight, bias)




class Conv2dQuantized(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 small=False, extra_small=False):
        super(Conv2dQuantized, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.init_weight = self.weight.clone().detach()
        self.init_bias = self.bias.clone().detach()
        self.small = small
        self.extra_small = extra_small

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach()

    def set_init_bias(self, init_bias):
        self.init_bias = init_bias.clone().detach()

    def forward(self, x):
        if self.extra_small:
            weight = ste_function_extra_small.apply(self.weight, self.init_weight)
        elif self.small:
            weight = ste_function_small.apply(self.weight, self.init_weight)
        else:
            weight = ste_function.apply(self.weight, self.init_weight)
        bias = ste_function.apply(self.bias, self.init_bias)
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
