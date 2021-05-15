from torch import nn
import torch.nn.functional as F
from utils.masked_layers import Conv2dMasked, LinearMasked
from utils.quantized_layers import Conv2dQuantized, LinearQuantized

class LeNet5(nn.Module):
	def __init__(self, in_channels=1, num_classes=10, normal_init=True):
		super(LeNet5, self).__init__()
		self.features = nn.Sequential(nn.Conv2d(in_channels, 20, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=False),
									  nn.Conv2d(20, 50, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=False))
		self.classifier = nn.Sequential(nn.Linear(4 * 4 * 50, 500), nn.ReLU(inplace=False), nn.Linear(500, num_classes))
		if normal_init:
			for m in self.modules():
				if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 4 * 4 * 50)
		x = self.classifier(x)
		return x


class LeNet5_masked(nn.Module):
	def __init__(self, in_channels=1, num_classes=10, normal_init=True):
		super(LeNet5_masked, self).__init__()
		self.features = nn.Sequential(Conv2dMasked(in_channels, 20, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
									  Conv2dMasked(20, 50, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))
		self.classifier = nn.Sequential(LinearMasked(4 * 4 * 50, 500), nn.ReLU(inplace=True), LinearMasked(500, num_classes))
		if normal_init:
			for m in self.modules():
				if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 4 * 4 * 50)
		x = self.classifier(x)
		return x



class LeNet5_quantized(nn.Module):
	def __init__(self, in_channels=1, num_classes=10, normal_init=True, granularity_channel=False, granularity_kernel=False):
		super(LeNet5_quantized, self).__init__()
		self.features = nn.Sequential(Conv2dQuantized(in_channels, 20, 5, 1, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel),
									  nn.MaxPool2d(2, 2),
									  nn.ReLU(inplace=True),
									  Conv2dQuantized(20, 50, 5, 1, granularity_channel=granularity_channel, granularity_kernel=granularity_kernel),
									  nn.MaxPool2d(2, 2),
									  nn.ReLU(inplace=True))
		self.classifier = nn.Sequential(LinearQuantized(4 * 4 * 50, 500, granularity_channel=granularity_channel),
										nn.ReLU(inplace=True),
										LinearQuantized(500, num_classes, granularity_channel=granularity_channel))
		if normal_init:
			for m in self.modules():
				if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
					m.set_init_weight(m.weight)

	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 4 * 4 * 50)
		x = self.classifier(x)
		return x