from src.models.LeNet5 import LeNet5
from src.models.VGG import VGG
from src.models.ResNet import ResNet
from torch import nn, optim
from torchvision.datasets import MNIST, CIFAR10, ImageNet
import torch
import torchvision
import numpy as np
import argparse
import time


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")


def main(args):
    # load dataset
    in_channels, num_classes, dataloader_train, dataloader_test = load_dataset(args.dataset_name)


    # build neural network
    if args.model_name == 'LeNet5':
        model = LeNet5(in_channels=in_channels,num_classes=num_classes)
    elif 'VGG' in args.model_name:
        model = VGG(in_channels=in_channels, num_classes=num_classes)
    elif 'ResNet' in args.model_name:
        model = ResNet(in_channels=in_channels, num_classes=num_classes)
    else:
        print('Architecture not supported! Please choose from: LeNet5, VGG and ResNet.')

    # train
    if args.train_index:
        pass
    else:
        train(model, dataloader_train, dataloader_test, max_epoch=3)


def load_dataset(dataset_name, batch_size=64):
    # load dataset
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if dataset_name == 'MNIST':
        data_train = MNIST(root='../datasets/MNIST', train=True, download=True, transform=transform)
        data_test = MNIST(root='../datasets/MNIST', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        data_train = CIFAR10(root='../datasets/CIFAR10', train=True, download=True, transform=transform)
        data_test = CIFAR10(root='../datasets/CIFAR10', train=False, download=True, transform=transform)
    elif dataset_name == 'ImageNet':
        data_train = ImageNet(root='../datasets/ImageNet', splite='train', download=True, transform=transform)
        data_test = ImageNet(root='../datasets/ImageNet', splite='val', download=True, transform=transform)
    else:
        print('Dataset not supported! Please choose from: MNIST, CIFAR10 and ImageNet.')
    in_channels = data_train[0][0].shape[0]
    num_classes = len(data_train.classes)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size)
    return in_channels, num_classes, dataloader_train, dataloader_test


def validate(model, dataloader_test):
    # validate
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader_test):
            images = images.to(device)
            x = model(images)
            value, pred = torch.max(x, 1)
            pred = pred.data.cpu()
            total += x.size(0)
            correct += torch.sum(pred == labels)
    return correct*100.0/total


def train(model, dataloader_train, dataloader_test, max_epoch=10000, lr=1e-3, patience=20):
    dur = []  # duration for training epochs
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    max_accuracy = 0
    cur_step = 0
    for epoch in range(max_epoch):
        t0 = time.time()  # start time
        model.train()
        for i, (images, labels) in enumerate(dataloader_train):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()
        # validate
        dur.append(time.time() - t0)
        accuracy = float(validate(model, dataloader_test))
        print("Epoch {:05d} | Test Acc {:.4f}% | Time(s) {:.4f}".format(epoch + 1, accuracy, np.mean(dur)))
        # early stop
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Fixed Point")

    parser.add_argument('--dataset_name', default='MNIST', help='choose dataset from: MNIST, CIFAR10, ImageNet')
    parser.add_argument('--model_name', default='LeNet5', help='choose architecture from: LeNet5, VGG16, ResNet18')
    parser.add_argument('--train_index', action='store_true', help='if true train index, else train in normal way')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")