from models.LeNet5 import LeNet5, LeNet5_masked
from models.VGG import VGG_small, VGG_small_masked
from models.ResNet import ResNet
from torch import nn, optim
from torchvision.datasets import MNIST, CIFAR10, ImageNet
import torch
import torchvision
import numpy as np
import argparse
import copy
import os
import time
from utils.index_optimizer import Index_SGD, Index_Adam
import torch.nn.utils.prune
from torchvision import transforms


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")


def main(args):
    # load dataset
    num_workers, in_channels, num_classes, dataloader_train, dataloader_test = load_dataset(args.dataset_name)
    #torch.set_num_threads(num_workers)

    # build neural network
    if args.model_name == 'LeNet5':
        model = LeNet5_masked(in_channels=in_channels, num_classes=num_classes).to(device)
    elif 'VGG' in args.model_name:
        model = VGG_small_masked(in_channels=in_channels, num_classes=num_classes).to(device)
    elif 'ResNet' in args.model_name:
        pass
    else:
        print('Architecture not supported! Please choose from: LeNet5, VGG and ResNet.')

    if args.train_from_start:
        if args.train_index:
            init_param_path = './checkpoints/init_param_' + args.model_name + '_' + \
                              args.dataset_name + '_train_index_from_start.pth'
            # save initial parameters
            torch.save(model.state_dict(), init_param_path)
            prune_param_path = './checkpoints/prune_param_' + args.model_name + '_' + \
                               args.dataset_name + '_train_index_from_start.pth'
            prune(model, dataloader_train, dataloader_test, max_fine_tuning_epoch=200,
                  save_path=prune_param_path, train_index=True)
        else:
            init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '_from_start.pth'
            # save initial parameters
            torch.save(model.state_dict(), init_param_path)
            prune_param_path = './checkpoints/prune_param_' + args.model_name + '_' + args.dataset_name + '_from_start.pth'
            prune(model, dataloader_train, dataloader_test, max_fine_tuning_epoch=200, save_path=prune_param_path)
    else:
        # train
        if args.train_index:
            init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '_train_index.pth'
            final_param_path = './checkpoints/final_param_' + args.model_name + '_' + args.dataset_name + '_train_index.pth'
            # save initial parameters
            torch.save(model.state_dict(), init_param_path)
            _ = train(model, dataloader_train, dataloader_test, train_index=True, save_path=final_param_path)
        else:
            init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '.pth'
            final_param_path = './checkpoints/final_param_' + args.model_name + '_' + args.dataset_name + '.pth'
            # save initial parameters
            torch.save(model.state_dict(), init_param_path)
            _ = train(model, dataloader_train, dataloader_test, save_path=final_param_path)

        # prune
        model.load_state_dict(torch.load(final_param_path))
        if args.train_index:
            prune_param_path = './checkpoints/prune_param_' + args.model_name + '_' + args.dataset_name + '_train_index.pth'
        else:
            prune_param_path = './checkpoints/prune_param_' + args.model_name + '_' + args.dataset_name + '.pth'
        prune(model, dataloader_train, dataloader_test, save_path=prune_param_path, train_index=args.train_index)


def load_dataset(dataset_name, batch_size=128):
    # load dataset
    if dataset_name == 'MNIST':
        num_workers = 1
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_train = MNIST(root='../datasets', train=True, download=True, transform=transform)
        data_test = MNIST(root='../datasets', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        num_workers = 8
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        data_train = CIFAR10(root='../datasets', train=True, download=True, transform=transform_train)
        data_test = CIFAR10(root='../datasets', train=False, download=True, transform=transform_test)
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
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, pin_memory=True)
    return num_workers, in_channels, num_classes, dataloader_train, dataloader_test



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



def train(model, dataloader_train, dataloader_test, train_index=False, max_epoch=100, lr=1e-3, patience=15,
          save_path=None, args=None):
    dur = []  # duration for training epochs
    loss_func = nn.CrossEntropyLoss()
    if train_index:
        #optimizer = Index_SGD(model.parameters(), lr=1e-2, momentum=0.9)   # for VGG
        optimizer = Index_Adam(model.parameters())    # for LeNet5
    else:
        #optimizer = optim.SGD(model.parameters(), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    best_test_acc = 0
    corresp_train_acc = 0
    best_epoch = 0
    cur_step = 0
    for epoch in range(max_epoch):
        optimizer.param_groups[0]['lr'] *= 0.99
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
        train_accuracy = float(validate(model, dataloader_train))
        test_accuracy = float(validate(model, dataloader_test))
        print("Epoch {:05d} | Training Acc {:.4f}% | Test Acc {:.4f}% | Time(s) {:.4f}".format(epoch + 1, train_accuracy, test_accuracy, np.mean(dur)))
        # early stop
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            corresp_train_acc = train_accuracy
            best_epoch = epoch + 1
            cur_step = 0
            # save checkpoint
            torch.save(model.state_dict(), save_path)
        else:
            cur_step += 1
            if cur_step == patience:
                break
    print("Training finished! Best test accuracy = {:.4f}%, corresponding training accuracy = {:.4f}%, "
          "found at Epoch {:05d}.".format(best_test_acc, corresp_train_acc, best_epoch))
    return best_test_acc


def prune(model, dataloader_train, dataloader_test, max_pruning_epoch=1, max_fine_tuning_epoch=50, amount=0.99,
          save_path=None, train_index=False):
    l = [module for module in model.modules() if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)]
    for pruning_epoch in range(max_pruning_epoch):
        for layer in l:
            mask = torch.nn.utils.prune.l1_unstructured(layer, 'weight', amount=amount)
            layer.set_mask(mask.weight_mask)
        acc_before_fine_tuning = validate(model, dataloader_test)
        acc_after_fine_tuning = train(model, dataloader_train, dataloader_test, train_index=train_index,
                                      max_epoch=max_fine_tuning_epoch, save_path=save_path)
        print("Prune Epoch {:05d} | Acc Before Tuning {:.4f}% | Acc After Tuning {:.4f}% "
              .format(pruning_epoch + 1, acc_before_fine_tuning, acc_after_fine_tuning))


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Fixed Point")

    parser.add_argument('--dataset_name', default='MNIST', help='choose dataset from: MNIST, CIFAR10, ImageNet')
    parser.add_argument('--model_name', default='LeNet5', help='choose architecture from: LeNet5, VGG16, ResNet18')
    parser.add_argument('--train_index', action='store_true', help='if true train index, else train in normal way')
    parser.add_argument('--train_from_start', action='store_true', help='if or not train sparse architecture from start')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")