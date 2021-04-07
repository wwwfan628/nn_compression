from torch import nn
import torch.nn.functional as F
from ..utils.masked_layers import Conv2dMasked, LinearMasked

class LeNet5(nn.Module):
	def __init__(self, in_channels=1, num_classes=10, normal_init=True):
		super(LeNet5, self).__init__()
		self.features = nn.Sequential(nn.Conv2d(in_channels, 20, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
									  nn.Conv2d(20, 50, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))
		self.classifier = nn.Sequential(nn.Linear(4 * 4 * 50, 500), nn.ReLU(inplace=True), nn.Linear(500, num_classes))
		if normal_init:
			for m in self.modules():
				if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 4 * 4 * 50)
		x = self.classifier(x)
		return x


class LeNet5Masked(nn.Module):
	def __init__(self, in_channels=1, num_classes=10, normal_init=True):
		super(LeNet5Masked, self).__init__()
		self.conv1 = Conv2dMasked(in_channels=in_channels, out_channels=6, kernel_size=(5, 5), padding=2)
		self.conv2 = Conv2dMasked(in_channels=6, out_channels=16, kernel_size=(5, 5))
		self.fc1 = LinearMasked(in_features=16 * 5 * 5, out_features=120)
		self.fc2 = LinearMasked(in_features=120, out_features=84)
		self.fc3 = LinearMasked(in_features=84, out_features=num_classes)
		if normal_init:
			nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
			nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
			nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
			nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
			nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
		x = x.reshape(x.shape[0], -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x