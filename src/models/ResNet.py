from torch import nn
from utils.masked_layers import Conv2dMasked, LinearMasked
from utils.quantized_layers import Conv2dQuantized, LinearQuantized

ResNet_types = {
    'ResNet18': [2, 2, 2, 2],
    'ResNet34': [3, 4, 6, 3],
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
    'ResNet152': [3, 8, 36, 3]
}

class block_small(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block_small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class block_large(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block_large, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels/4, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels/4)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels/4, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels/4)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResNet_type='ResNet18', image_channels=3, num_classes=1000, normal_init=True):
        super(ResNet, self).__init__()
        # initial layers
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet layers
        if ('18' in ResNet_type) or ('34' in ResNet_type):
            self.layer1 = self._make_layer(block_small, num_residual_blocks=ResNet_types[ResNet_type][0], out_channels=64, stride=1)
            self.layer2 = self._make_layer(block_small, num_residual_blocks=ResNet_types[ResNet_type][1], out_channels=128, stride=2)
            self.layer3 = self._make_layer(block_small, num_residual_blocks=ResNet_types[ResNet_type][2], out_channels=256, stride=2)
            self.layer4 = self._make_layer(block_small, num_residual_blocks=ResNet_types[ResNet_type][3], out_channels=512, stride=2)
            # average pool amd full connected layer in the end
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features=512, out_features=num_classes)
        else:
            self.layer1 = self._make_layer(block_large, num_residual_blocks=ResNet_types[ResNet_type][0], out_channels=256, stride=1)
            self.layer2 = self._make_layer(block_large, num_residual_blocks=ResNet_types[ResNet_type][1], out_channels=512, stride=2)
            self.layer3 = self._make_layer(block_large, num_residual_blocks=ResNet_types[ResNet_type][2], out_channels=1024, stride=2)
            self.layer4 = self._make_layer(block_large, num_residual_blocks=ResNet_types[ResNet_type][3], out_channels=2048, stride=2)
            # average pool amd full connected layer in the end
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        if normal_init:
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels
        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)




class block_small_masked(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block_small_masked, self).__init__()
        self.conv1 = Conv2dMasked(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2dMasked(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class block_large_masked(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block_large_masked, self).__init__()
        self.conv1 = Conv2dMasked(in_channels=in_channels, out_channels=out_channels/4, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels/4)
        self.conv2 = Conv2dMasked(in_channels=out_channels, out_channels=out_channels/4, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels/4)
        self.conv3 = Conv2dMasked(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet_masked(nn.Module):
    def __init__(self, ResNet_type='ResNet18', image_channels=3, num_classes=1000, normal_init=True):
        super(ResNet_masked, self).__init__()
        # initial layers
        self.in_channels = 64
        self.conv1 = Conv2dMasked(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet layers
        if ('18' in ResNet_type) or ('34' in ResNet_type):
            self.layer1 = self._make_layer(block_small_masked, num_residual_blocks=ResNet_types[ResNet_type][0], out_channels=64, stride=1)
            self.layer2 = self._make_layer(block_small_masked, num_residual_blocks=ResNet_types[ResNet_type][1], out_channels=128, stride=2)
            self.layer3 = self._make_layer(block_small_masked, num_residual_blocks=ResNet_types[ResNet_type][2], out_channels=256, stride=2)
            self.layer4 = self._make_layer(block_small_masked, num_residual_blocks=ResNet_types[ResNet_type][3], out_channels=512, stride=2)
            # average pool amd full connected layer in the end
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = LinearMasked(in_features=512, out_features=num_classes)
        else:
            self.layer1 = self._make_layer(block_large_masked, num_residual_blocks=ResNet_types[ResNet_type][0], out_channels=256, stride=1)
            self.layer2 = self._make_layer(block_large_masked, num_residual_blocks=ResNet_types[ResNet_type][1], out_channels=512, stride=2)
            self.layer3 = self._make_layer(block_large_masked, num_residual_blocks=ResNet_types[ResNet_type][2], out_channels=1024, stride=2)
            self.layer4 = self._make_layer(block_large_masked, num_residual_blocks=ResNet_types[ResNet_type][3], out_channels=2048, stride=2)
            # average pool amd full connected layer in the end
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = LinearMasked(in_features=2048, out_features=num_classes)
        if normal_init:
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                Conv2dMasked(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels
        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)




class block_small_quantized(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, granularity_channel=False, granularity_kernel=False):
        super(block_small_quantized, self).__init__()
        self.conv1 = Conv2dQuantized(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2dQuantized(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=1, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class block_large_quantized(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, granularity_channel=False, granularity_kernel=False):
        super(block_large_quantized, self).__init__()
        self.conv1 = Conv2dQuantized(in_channels=in_channels, out_channels=out_channels/4, kernel_size=1, stride=1, padding=0, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
        self.bn1 = nn.BatchNorm2d(out_channels/4)
        self.conv2 = Conv2dQuantized(in_channels=out_channels, out_channels=out_channels/4, kernel_size=3, stride=stride, padding=1, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
        self.bn2 = nn.BatchNorm2d(out_channels/4)
        self.conv3 = Conv2dQuantized(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet_quantized(nn.Module):
    def __init__(self, ResNet_type='ResNet18', image_channels=3, num_classes=1000, normal_init=True, granularity_channel=False, granularity_kernel=False):
        super(ResNet_quantized, self).__init__()
        # initial layers
        self.in_channels = 64
        self.conv1 = Conv2dQuantized(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet layers
        if ('18' in ResNet_type) or ('34' in ResNet_type):
            self.layer1 = self._make_layer(block_small_quantized, num_residual_blocks=ResNet_types[ResNet_type][0], out_channels=64, stride=1, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
            self.layer2 = self._make_layer(block_small_quantized, num_residual_blocks=ResNet_types[ResNet_type][1], out_channels=128, stride=2, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
            self.layer3 = self._make_layer(block_small_quantized, num_residual_blocks=ResNet_types[ResNet_type][2], out_channels=256, stride=2, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
            self.layer4 = self._make_layer(block_small_quantized, num_residual_blocks=ResNet_types[ResNet_type][3], out_channels=512, stride=2, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
            # average pool amd full connected layer in the end
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = LinearQuantized(in_features=512, out_features=num_classes, granularity_channel=granularity_channel)
        else:
            self.layer1 = self._make_layer(block_large_quantized, num_residual_blocks=ResNet_types[ResNet_type][0], out_channels=256, stride=1, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
            self.layer2 = self._make_layer(block_large_quantized, num_residual_blocks=ResNet_types[ResNet_type][1], out_channels=512, stride=2, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
            self.layer3 = self._make_layer(block_large_quantized, num_residual_blocks=ResNet_types[ResNet_type][2], out_channels=1024, stride=2, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
            self.layer4 = self._make_layer(block_large_quantized, num_residual_blocks=ResNet_types[ResNet_type][3], out_channels=2048, stride=2, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel)
            # average pool amd full connected layer in the end
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = LinearQuantized(in_features=2048, out_features=num_classes, granularity_channel=granularity_channel)
        if normal_init:
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    m.set_init_weight(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride, granularity_channel=False, granularity_kernel=False):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                Conv2dQuantized(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, stride=stride, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel),
                nn.BatchNorm2d(out_channels))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel))
        self.in_channels = out_channels
        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels, out_channels, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel))
        return nn.Sequential(*layers)
