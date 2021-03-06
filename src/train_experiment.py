from models.LeNet5 import LeNet5
from models.VGG import VGG_small
from models.ResNet import ResNet
from torch import nn, optim
from torchvision.datasets import MNIST, CIFAR10, ImageNet, ImageFolder
import torch
import numpy as np
import argparse
import os
import time
from utils.index_optimizer import Index_Adam_full, Index_SGD_full, Index_Adam, Index_SGD
from torchvision import transforms


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda Available")
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
        num_threads = 16
    torch.set_num_threads(num_threads)

    # load dataset
    in_channels, num_classes, dataloader_train, dataloader_test = load_dataset(args.dataset_name)

    # build neural network
    if args.model_name == 'LeNet5':
        model = LeNet5(in_channels=in_channels, num_classes=num_classes, normal_init=args.normal).to(device)
    elif 'VGG' in args.model_name:
        model = VGG_small(in_channels=in_channels, num_classes=num_classes, normal_init=args.normal).to(device)
    elif 'ResNet' in args.model_name:
        model = ResNet(in_channels=in_channels, num_classes=num_classes, normal_init=args.normal).to(device)
    else:
        print('Architecture not supported! Please choose from: LeNet5, VGG and ResNet.')

    # train
    if args.init_param_path == None:
        if args.train_index:
            init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '_train_index.pth'
        else:
            init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '.pth'
    else:
        init_param_path = args.init_param_path
    # save initial parameters
    torch.save(model.state_dict(), init_param_path)
    train(model, dataloader_train, dataloader_test, args)


def load_dataset(dataset_name):
    # load dataset
    if dataset_name == 'MNIST':
        num_workers = 0
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
        data_train = ImageFolder(root='/itet-stor/yiflu/net_scratch/imagenet/train', transform=transform_train)
        data_test = ImageFolder(root='/itet-stor/yiflu/net_scratch/imagenet/val', transform=transform_val)
    else:
        print('Dataset is not supported! Please choose from: MNIST, CIFAR10 and ImageNet.')
    in_channels = data_train[0][0].shape[0]
    num_classes = len(data_train.classes)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
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
            _, pred = torch.max(x, 1)
            pred = pred.data.cpu()
            total += x.size(0)
            correct += torch.sum(pred == labels)
    return correct*100.0/total



def train(model, dataloader_train, dataloader_test, args):
    dur = []  # duration for training epochs
    loss_func = nn.CrossEntropyLoss()
    if args.train_index:
        if 'LeNet' in args.model_name:
            optimizer = Index_Adam_full(model.parameters(), lr=1e-3, ste=args.ste, params_prime=model.parameters(),
                                        granularity_channel=args.granularity_channel,
                                        granularity_kernel=args.granularity_kernel)  # for LeNet5
        elif 'VGG' in args.model_name:
            # if args.granularity_kernel or args.granularity_channel:
                # model = torch.nn.DataParallel(model).to(device)
            optimizer = Index_SGD_full(model.parameters(), lr=1e-2, momentum=0.9, ste=args.ste,
                                       params_prime=model.parameters(), granularity_channel=args.granularity_channel,
                                       granularity_kernel=args.granularity_kernel)  # for VGG
        else:
            model = torch.nn.DataParallel(model).to(device)
            optimizer = Index_SGD_full(model.parameters(), lr=0.4, nesterov=True, momentum=0.9, weight_decay=1e-4,
                                       ste=args.ste, params_prime=model.parameters(),
                                       granularity_channel=args.granularity_channel,
                                       granularity_kernel=args.granularity_kernel)
    else:
        if 'LeNet' in args.model_name:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        elif 'VGG' in args.model_name:
            optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    best_test_acc = 0
    corresp_train_acc = 0
    best_epoch = 0
    cur_step = 0
    for epoch in range(args.max_epoch):
        if args.train_index:
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
            if args.final_param_path == None:
                if args.train_index:
                    final_param_path = './checkpoints/final_param_' + args.model_name + '_' + args.dataset_name + '_train_index.pth'
                else:
                    final_param_path = './checkpoints/final_param_' + args.model_name + '_' + args.dataset_name + '.pth'
            else:
                final_param_path = args.final_param_path
            if args.dataset_name == 'ImageNet':
                torch.save(model.module.state_dict(), final_param_path)
            else:
                torch.save(model.state_dict(), final_param_path)
        else:
            cur_step += 1
            if cur_step == args.patience:
                break
    print("Training finished! Best test accuracy = {:.4f}%, corresponding training accuracy = {:.4f}%, "
          "found at Epoch {:05d}.".format(best_test_acc, corresp_train_acc, best_epoch))



if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Traditional Training")

    parser.add_argument('--dataset_name', default='MNIST', help='choose dataset from: MNIST, CIFAR10, ImageNet')
    parser.add_argument('--model_name', default='LeNet5', help='choose architecture from: LeNet5, VGG, ResNet18')
    parser.add_argument('--train_index', action='store_true', help='if true train index, else train in normal way')
    parser.add_argument('--max_epoch', type=int, default=100, help='max training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of optimizer')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
    parser.add_argument('--ste', action='store_true', help='if use straight through estimation or not')
    parser.add_argument('--granularity_channel', action='store_true', help='if true, update index inside one channel')
    parser.add_argument('--granularity_kernel', action='store_true', help='if true, update index inside one kernel')
    parser.add_argument('--normal', action='store_true', help='if true, initialize with normal distribution')
    parser.add_argument('--init_param_path', default=None)
    parser.add_argument('--final_param_path', default=None)
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")