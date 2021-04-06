from torch import nn
from src.utils.masked_layers import Conv2dMasked, LinearMasked

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
# Then flatten and 4096 -> 4096 -> num_classes Linear Layers

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers(VGG_types['VGG16'])
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=512*1*1, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc_layers(x)
        return x

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for layer in architecture:
            if type(layer) == int:
                out_channels = layer
                layers+= [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU()]
                in_channels = out_channels
            elif layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=2)]
        return nn.Sequential(*layers)




class VGGMasked(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGGMasked, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers(VGG_types['VGG16'])
        self.fc_layers = nn.Sequential(
            LinearMasked(in_features=512*1*1, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            LinearMasked(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            LinearMasked(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc_layers(x)
        return x

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for layer in architecture:
            if type(layer) == int:
                out_channels = layer
                layers+= [Conv2dMasked(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU()]
                in_channels = out_channels
            elif layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=2)]
        return nn.Sequential(*layers)