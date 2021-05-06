from models.LeNet5 import LeNet5
from models.VGG import VGG_small
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
from utils.prune_weight import prune_weight_interval, prune_weight_abs
from torchvision import transforms


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")


def main(args):
    # check whether checkpoints directory exist
    path = os.path.join(os.getcwd(), './checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)

    # set flag & parallel processing
    torch.backends.cudnn.benchmark = True
    if args.dataset_name == 'MNIST':
        num_threads = 1
    elif args.dataset_name == 'CIFAR10':
        num_threads = 8
    else:
        num_threads = 32
    torch.set_num_threads(num_threads)

    # load dataset
    in_channels, num_classes, dataloader_train, dataloader_test = load_dataset(args.dataset_name)

    # build neural network
    if args.model_name == 'LeNet5':
        model = LeNet5(in_channels=in_channels, num_classes=num_classes, normal_init=True).to(device)
    elif 'VGG' in args.model_name:
        model = VGG_small(in_channels=in_channels, num_classes=num_classes, normal_init=True).to(device)
    elif 'ResNet' in args.model_name:
        model = ResNet(ResNet_type=args.model_name, image_channels=in_channels, num_classes=num_classes, normal_init=True).to(device)
    else:
        print('Architecture not supported! Please choose from: LeNet5, VGG and ResNet.')

    # preprocess parameters
    with torch.no_grad():
        l = [module for module in model.modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)]
        for layer in l:
            #prune_weight_interval(layer.weight)
            prune_weight_abs(layer.weight, amount=0.9)

    # train
    if args.train_index:
        init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '_train_index_prune.pth'
    else:
        init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '_prune.pth'
    # save initial parameters
    torch.save(model.state_dict(), init_param_path)
    train(model, dataloader_train, dataloader_test, args)


def load_dataset(dataset_name):
    # load dataset
    if dataset_name == 'MNIST':
        num_workers = 1
        batch_size = 128
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_train = MNIST(root='../datasets', train=True, download=True, transform=transform)
        data_test = MNIST(root='../datasets', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        num_workers = 8
        batch_size = 128
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        data_train = CIFAR10(root='../datasets', train=True, download=True, transform=transform_train)
        data_test = CIFAR10(root='../datasets', train=False, download=True, transform=transform_test)
    elif dataset_name == 'ImageNet':
        num_workers = 32
        batch_size = 1024
        transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        data_train = ImageNet(root='/srv/beegfs01/projects/imagenet/data/', split='train', transform=transform_train)
        data_test = ImageNet(root='/srv/beegfs01/projects/imagenet/data/', split='val', transform=transform_val)
    else:
        print('Dataset is not supported! Please choose from: MNIST, CIFAR10 and ImageNet.')
    in_channels = data_train[0][0].shape[0]
    num_classes = len(data_train.classes)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
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



def train(model, dataloader_train, dataloader_test, train_index=False, max_epoch=200, lr=1e-3, patience=20):
    dur = []  # duration for training epochs
    loss_func = nn.CrossEntropyLoss()
    if args.train_index:
        if 'LeNet' in args.model_name:
            optimizer = Index_Adam(model.parameters(), lr=1e-1)  # for LeNet5
        elif 'VGG' in args.model_name:
            optimizer = Index_SGD(model.parameters(), lr=1e-2, momentum=0.9)  # for VGG
        else:
            optimizer = Index_SGD(model.parameters(), lr=0.4, nesterov=True, momentum=0.9, weight_decay=1e-4)
    else:
        #optimizer = optim.SGD(model.parameters(), lr=args.lr)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_test_acc = 0
    corresp_train_acc = 0
    best_epoch = 0
    cur_step = 0
    for epoch in range(max_epoch):
        # adjust lr
        if 'LeNet' in args.model_name or 'VGG' in args.model_name:
            optimizer.param_groups[0]['lr'] *= 0.99
        else:
            if epoch < 10:
                optimizer.param_groups[0]['lr'] = 0.4 * (epoch + 1) / 10
            if epoch == 60 or epoch == 120 or epoch == 180:
                optimizer.param_groups[0]['lr'] *= 0.1
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
            if train_index:
                param_after_training_path = './checkpoints/final_param_' + args.model_name + '_' + args.dataset_name + '_train_index_prune.pth'
            else:
                param_after_training_path = './checkpoints/final_param_' + args.model_name + '_' + args.dataset_name + '_prune.pth'
            torch.save(model.state_dict(), param_after_training_path)
        else:
            cur_step += 1
            if cur_step == patience:
                break
    print("Training finished! Best test accuracy = {:.4f}%, corresponding training accuracy = {:.4f}%, "
          "found at Epoch {:05d}.".format(best_test_acc, corresp_train_acc, best_epoch))


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Fixed Point")

    parser.add_argument('--dataset_name', default='MNIST', help='choose dataset from: MNIST, CIFAR10, ImageNet')
    parser.add_argument('--model_name', default='LeNet5', help='choose architecture from: LeNet5, VGG16, ResNet18')
    parser.add_argument('--train_index', action='store_true', help='if true train index, else train in normal way')
    parser.add_argument('--max_epoch', type=int, default=250, help='max training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of optimizer')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")