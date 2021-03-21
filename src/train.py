from src.models.LeNet5 import LeNet5
from src.models.VGG import VGG
from src.models.ResNet import ResNet
from torch import nn, optim
from torchvision.datasets import MNIST, CIFAR10, ImageNet
import torch
import torchvision
import numpy as np
import argparse
import copy
import os
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
    init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '.pth'
    if args.train_index:
        # load previously used parameters
        model.load_state_dict(torch.load(init_param_path))
        train_index(model, dataloader_train, dataloader_test, max_epoch=3)
    else:
        # save initial parameters
        torch.save(model.state_dict(), init_param_path)
        train(model, dataloader_train, dataloader_test, max_epoch=3)

    # save weights
    if args.train_index:
        param_after_training_path = './checkpoints/param_after_training' + args.model_name + '_' + args.dataset_name + '_train_index.pth'
    else:
        param_after_training_path = './checkpoints/param_after_training' + args.model_name + '_' + args.dataset_name + '.pth'
    torch.save(model.state_dict(), param_after_training_path)


def load_dataset(dataset_name, batch_size=64):
    # load dataset
    if dataset_name == 'MNIST':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        data_train = MNIST(root='../datasets/MNIST', train=True, download=True, transform=transform)
        data_test = MNIST(root='../datasets/MNIST', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        data_train = CIFAR10(root='../datasets/CIFAR10', train=True, download=True, transform=transform)
        data_test = CIFAR10(root='../datasets/CIFAR10', train=False, download=True, transform=transform)
    elif dataset_name == 'ImageNet':
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=[256,480]),
                                                    torchvision.transforms.RandomCrop(size=[224,224]),
                                                    torchvision.transforms.ToTensor()])
        data_train = torchvision.datasets.ImageFolder('../datasets/ImageNet/train', transform=transform)
        data_test = torchvision.datasets.ImageFolder('../datasets/ImageNet/val', transform=transform)
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

def reorder_weights(state_dict, state_dict_temp):
    dict_reordered = state_dict
    for (layer_key, layer_weights), (layer_key_temp, layer_weights_temp) in zip(state_dict.items(), state_dict_temp.items()):
        weights_shape = layer_weights.shape
        weights_flattened = layer_weights.flatten()
        index_order = torch.argsort(layer_weights_temp.flatten())
        dict_reordered[layer_key] = sort_1Dtensor_by_index(weights_flattened, index_order).view(weights_shape)
    return dict_reordered


def sort_1Dtensor_by_index(tensor_to_sort, index):
    tensor_sorted = tensor_to_sort
    for i in range(tensor_to_sort.shape[0]):
        tensor_sorted[i] = tensor_to_sort[index[i]]
    return tensor_sorted


def train_index(model, dataloader_train, dataloader_test, max_epoch=10000, lr=1e-3, patience=20):
    model_temp = copy.deepcopy(model)
    optimizer = optim.Adam(model_temp.parameters(), lr=lr)
    dur = []  # duration for training epochs
    loss_func = nn.CrossEntropyLoss()
    max_accuracy = 0
    cur_step = 0
    for epoch in range(max_epoch):
        t0 = time.time()  # start time
        for i, (images, labels) in enumerate(dataloader_train):
            with torch.no_grad():
                model_temp.load_state_dict(model.state_dict())
            images = images.to(device)
            labels = labels.to(device)
            model_temp.train()
            optimizer.zero_grad()
            pred = model_temp(images)
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()
            # reorder weights in each layer
            state_dict_reordered = reorder_weights(model.state_dict(), model_temp.state_dict())
            model.load_state_dict(state_dict_reordered)
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