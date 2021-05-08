from models.ResNet import ResNet_quantized
from torch import nn, optim
from torchvision.datasets import ImageFolder
import torch
import torchvision
import numpy as np
import argparse
import copy
import os
import time
from utils.index_optimizer import Index_SGD, Index_Adam
from torchvision import transforms


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")


def main():

    # set flag & parallel processing
    torch.backends.cudnn.benchmark = True
    num_threads = 8
    torch.set_num_threads(num_threads)

    check_dataset()
    # load dataset
    #in_channels, num_classes, dataloader_train, dataloader_test = load_dataset()

    #model = ResNet_quantized(ResNet_type='ResNet18', image_channels=in_channels, num_classes=num_classes, normal_init=True, small=False, extra_small=False).to(device)
    #train(model, dataloader_train, dataloader_test)


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path




def check_dataset():
    num_workers = 8
    batch_size = 128
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_train = ImageFolderWithPaths(root='/srv/beegfs01/projects/imagenet/data/train/', transform=transform_train)
    data_test = ImageFolderWithPaths(root='/srv/beegfs01/projects/imagenet/data/val/', transform=transform_val)

    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                   num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=num_workers)
    for i, (images, labels, paths) in enumerate(dataloader_train):
        print(paths)
    for i, (images, labels, paths) in enumerate(dataloader_test):
        print(paths)

def load_dataset():
    num_workers = 8
    batch_size = 128
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_train = ImageFolder(root='/srv/beegfs01/projects/imagenet/data/train/', transform=transform_train)
    data_test = ImageFolder(root='/srv/beegfs01/projects/imagenet/data/val/', transform=transform_val)

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



def train(model, dataloader_train, dataloader_test):
    dur = []  # duration for training epochs
    loss_func = nn.CrossEntropyLoss()
    optimizer = Index_SGD(model.parameters(), lr=0.4, nesterov=True, momentum=0.9, weight_decay=1e-4)
    best_test_acc = 0
    corresp_train_acc = 0
    best_epoch = 0
    cur_step = 0
    for epoch in range(250):
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
        else:
            cur_step += 1
            if cur_step == 20:
                break
    print("Training finished! Best test accuracy = {:.4f}%, corresponding training accuracy = {:.4f}%, "
          "found at Epoch {:05d}.".format(best_test_acc, corresp_train_acc, best_epoch))


if __name__ == '__main__':
    # set flag & parallel processing
    torch.backends.cudnn.benchmark = True
    num_threads = 8
    torch.set_num_threads(num_threads)

    num_workers = 8
    batch_size = 128
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_train = ImageFolderWithPaths(root='/srv/beegfs01/projects/imagenet/data/train/', transform=transform_train)
    data_test = ImageFolderWithPaths(root='/srv/beegfs01/projects/imagenet/data/val/', transform=transform_val)

    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                   num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=num_workers)
    for i, (images, labels, paths) in enumerate(dataloader_train):
        print(paths)
    for i, (images, labels, paths) in enumerate(dataloader_test):
        print(paths)

    print("Finish!")