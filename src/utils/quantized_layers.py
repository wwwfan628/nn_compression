import torch.nn as nn
import torch.nn.functional as F
from straight_through_estimator import ste_function


class LinearQuantized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearQuantized, self).__init__(in_features, out_features, bias)
        self.init_weight = self.weight.clone().detach()

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach()

    def forward(self, x):
        weight = ste_function.apply(self.weight, self.init_weight)
        return F.linear(x, weight, self.bias)




class Conv2dQuantized(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dQuantized, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.init_weight = self.weight.clone().detach()

    def set_init_weight(self, init_weight):
        self.init_weight = init_weight.clone().detach()

    def forward(self, x):
        weight = ste_function.apply(self.weight, self.init_weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)