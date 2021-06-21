from torch import nn
from utils.masked_layers import Conv2dMasked, LinearMasked
from utils.quantized_layers import Conv2dQuantized, Conv2dQuantized_granularity_channel, Conv2dQuantized_granularity_kernel, LinearQuantized, LinearQuantized_granularity_channel


class VGG_small(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, in_channels=3, num_classes=1000, normal_init=True):
        super(VGG_small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        if normal_init:
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_small_masked(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, in_channels=3, num_classes=1000, normal_init=True):
        super(VGG_small_masked, self).__init__()
        self.features = nn.Sequential(
            Conv2dMasked(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dMasked(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dMasked(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dMasked(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dMasked(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dMasked(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            LinearMasked(512 * 4 * 4, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearMasked(1024, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearMasked(1024, num_classes),
        )
        if normal_init:
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class VGG_small_quantized(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, in_channels=3, num_classes=1000, normal_init=True):
        super(VGG_small_quantized, self).__init__()
        self.features = nn.Sequential(
            Conv2dQuantized(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dQuantized(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dQuantized(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            LinearQuantized(512 * 4 * 4, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearQuantized(1024, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearQuantized(1024, num_classes)
        )
        if normal_init:
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    m.set_init_weight(m.weight)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_small_quantized_granularity_channel(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, in_channels=3, num_classes=1000, normal_init=True):
        super(VGG_small_quantized_granularity_channel, self).__init__()
        self.features = nn.Sequential(
            Conv2dQuantized_granularity_channel(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized_granularity_channel(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dQuantized_granularity_channel(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized_granularity_channel(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dQuantized_granularity_channel(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized_granularity_channel(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            LinearQuantized_granularity_channel(512 * 4 * 4, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearQuantized_granularity_channel(1024, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearQuantized_granularity_channel(1024, num_classes)
        )
        if normal_init:
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    m.set_init_weight(m.weight)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_small_quantized_granularity_kernel(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, in_channels=3, num_classes=1000, normal_init=True):
        super(VGG_small_quantized_granularity_kernel, self).__init__()
        self.features = nn.Sequential(
            Conv2dQuantized_granularity_kernel(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized_granularity_kernel(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dQuantized_granularity_kernel(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized_granularity_kernel(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dQuantized_granularity_kernel(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized_granularity_kernel(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            LinearQuantized_granularity_channel(512 * 4 * 4, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearQuantized_granularity_channel(1024, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearQuantized_granularity_channel(1024, num_classes)
        )
        if normal_init:
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    m.set_init_weight(m.weight)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x