from models.LeNet5 import LeNet5_quantized, LeNet5
from models.VGG import VGG_small
from models.ResNet import ResNet
from torch import nn, optim
from torchvision.datasets import MNIST, CIFAR10, ImageNet, ImageFolder
import torch
import numpy as np
import argparse
import os
import time
from utils.index_optimizer import Index_Adam_full, Index_SGD_full, Index_Adam_Input
from torchvision import transforms
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")

def main(args):

    # load dataset
    in_channels, num_classes, dataset_train, dataset_test = load_dataset(args.dataset_name)
    # load image
    image_tensor = dataset_train[args.image_index][0]  # tensor after preprocess
    label = dataset_train[args.image_index][1]
    plot_image(image_tensor, name='Original Image')

    # build models & load parameters from given path
    model_1 = LeNet5_quantized(in_channels=in_channels, num_classes=num_classes, normal_init=True).to(device)
    model_1.load_state_dict(torch.load(args.checkpoint_path_1))
    for parameter in model_1.parameters():
        parameter.requires_grad = False
    model_2 = LeNet5_quantized(in_channels=in_channels, num_classes=num_classes, normal_init=True).to(device)
    model_2.load_state_dict(torch.load(args.checkpoint_path_2))
    for parameter in model_2.parameters():
        parameter.requires_grad = False
    model_3 = LeNet5_quantized(in_channels=in_channels, num_classes=num_classes, normal_init=True).to(device)
    model_3.load_state_dict(torch.load(args.checkpoint_path_3))
    for parameter in model_3.parameters():
        parameter.requires_grad = False

    # build discriminators & load parameters from path
    discriminator_1 = LeNet5(in_channels=in_channels, num_classes=2, normal_init=True).to(device)
    discriminator_1.load_state_dict(torch.load(args.discriminator_checkpoint_path_1))
    for parameter in discriminator_1.parameters():
        parameter.requires_grad = False
    discriminator_2 = LeNet5(in_channels=in_channels, num_classes=2, normal_init=True).to(device)
    discriminator_2.load_state_dict(torch.load(args.discriminator_checkpoint_path_2))
    for parameter in discriminator_2.parameters():
        parameter.requires_grad = False
    discriminator_3 = LeNet5(in_channels=in_channels, num_classes=2, normal_init=True).to(device)
    discriminator_3.load_state_dict(torch.load(args.discriminator_checkpoint_path_3))
    for parameter in discriminator_3.parameters():
        parameter.requires_grad = False

    global classes_average_images
    classes_average_images = average_image(dataset_train)

    # randomize the input image
    random_idx = torch.randperm(image_tensor.nelement())
    random_image_tensor = image_tensor.view(-1)[random_idx].view(image_tensor.size())
    plot_image(random_image_tensor, name='Image after Shuffling')
    acc = accuracy(random_image_tensor, random_idx)
    print('After shuffling, accracy = {:.4f}%.'.format(acc))

    # optimization on input image
    init_indices = random_idx.view(image_tensor.size())
    models = [model_1, model_2, model_3]
    discriminators = [discriminator_1, discriminator_2, discriminator_3]
    optimize_input(models, discriminators, random_image_tensor, init_indices, label, args)


def load_dataset(dataset_name):
    # load dataset
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = MNIST(root='../datasets', train=True, download=True, transform=transform)
        dataset_test = MNIST(root='../datasets', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = CIFAR10(root='../datasets', train=True, download=True, transform=transform_train)
        dataset_test = CIFAR10(root='../datasets', train=False, download=True, transform=transform_test)
    elif dataset_name == 'ImageNet':
        transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset_train = ImageFolder(root='/srv/beegfs01/projects/imagenet/data/train/', transform=transform_train)
        dataset_test = ImageFolder(root='/srv/beegfs01/projects/imagenet/data/val/', transform=transform_val)
    else:
        print('Dataset is not supported! Please choose from: MNIST, CIFAR10 and ImageNet.')
    in_channels = dataset_train[0][0].shape[0]
    num_classes = len(dataset_train.classes)
    return in_channels, num_classes, dataset_train, dataset_test


def average_image(dataset):
    average_images = torch.zeros([10, 1, 28, 28])
    count = torch.zeros([10])
    for i, (image, label) in enumerate(dataset):
        average_images[label] += image
        count[label] += 1
    for i in range(10):
        average_images[i] = average_images[i] / count[i]
    return average_images


def plot_image(image_tensor, name=None):
    if len(image_tensor.shape) == 4:
        image_tensor = torch.squeeze(image_tensor.clone().detach(), 0)
    else:
        image_tensor = image_tensor.clone().detach()
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.title(name)
    plt.show()


def optimize_input(models, discriminators, random_image_tensor, init_indices, label, args):
    dur = []  # duration for training epochs
    random_image_tensor = torch.unsqueeze(random_image_tensor, 0)
    init_indices = torch.unsqueeze(init_indices, 0)
    loss_func = nn.CrossEntropyLoss()
    random_image_tensor.requires_grad = True
    optimizer = Index_Adam_Input([random_image_tensor], lr=args.lr, indices=init_indices)  # for LeNet5
    best_acc = 0
    best_epoch = 0
    best_image_tensor = None
    cur_step = 0
    factor = 1
    global classes_average_images
    for epoch in range(args.max_epoch):
        # adjust lr
        # optimizer.param_groups[0]['lr'] *= 0.99
        t0 = time.time()  # start time
        random_image_tensor = random_image_tensor.to(device)
        label = torch.tensor([label]).to(device)
        optimizer.zero_grad()
        loss = 0
        for model_idx, model in enumerate(models):
            pred = model(random_image_tensor)
            loss += (1.0/3.0)*loss_func(pred, label)
        for discriminator_idx, discriminator in enumerate(discriminators):
            pred = discriminator(random_image_tensor)
            loss += (1.0/3.0)*loss_func(pred, torch.tensor([1], dtype=int))
        loss_pixel_diff = pixel_diff_loss(random_image_tensor)
        loss_border_pixel = border_pixel_loss(random_image_tensor)
        loss_average_image = average_image_loss(random_image_tensor, classes_average_images[label])
        factor *= 0.99
        loss = loss + factor * loss_pixel_diff + loss_border_pixel + loss_average_image
        #loss = loss + factor * loss_pixel_diff + loss_border_pixel
        loss.backward()
        optimizer.step()


        # compute accuracy
        dur.append(time.time() - t0)
        indices = optimizer.param_groups[0]['indices'][0]
        acc = float(accuracy(random_image_tensor, indices))
        print("Epoch {:05d} | Acc {:.4f}% | Time(s) {:.4f}".format(epoch + 1, acc, np.mean(dur)))

        if (epoch < 5) or (epoch % 5 == 0):
            plot_image(random_image_tensor, name='Epoch: {}'.format(epoch))

        # early stop
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            cur_step = 0
            best_image_tensor = random_image_tensor.clone().detach()
        else:
            cur_step += 1
            if cur_step == args.patience:
                break
    plot_image(best_image_tensor, name='Image after Optimization')
    print("Training finished! Best accuracy = {:.4f}%, found at Epoch {:05d}.".format(best_acc, best_epoch))


def average_image_loss(image_batch_tensor, class_average_image):
    loss = (image_batch_tensor[0] - class_average_image).abs().sum()
    return loss/(28*28)


def accuracy(image_tensor, indices):
    img_shape = image_tensor.shape
    total = img_shape[-1] * img_shape[-2]
    correct_indices = [indices.view(-1)[i] == i for i in range(total)]
    correct = torch.sum(torch.tensor(correct_indices))
    return correct*100.0/total


def pixel_diff_loss(image_batch_tensor):
    num_pixels = 28*28*8
    kernel = torch.FloatTensor([
        [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, -1]]
    ]).unsqueeze(1)
    loss = torch.nn.functional.conv2d(image_batch_tensor, kernel).abs().sum()
    loss = loss/num_pixels
    return loss


def border_pixel_loss(image_batch_tensor):
    loss = 0
    for row_id, row in enumerate(image_batch_tensor[0, 0]):
        if row_id < 3 or row_id > 24:
            for pixel in row:
                loss += pixel
        else:
            for pixel in row[:3]:
                loss += pixel
            for pixel in row[24:]:
                loss += pixel
    return loss/(28*12-9*4)

if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Traditional Training")

    parser.add_argument('--dataset_name', default='MNIST', help='choose dataset from: MNIST, CIFAR10, ImageNet')
    parser.add_argument('--image_index', type=int, default=1, help='if true train index, else train in normal way')
    parser.add_argument('--max_epoch', type=int, default=250, help='max optimization iteration')
    parser.add_argument('--lr', type=float, default=1000, help='learning rate of optimizer')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
    parser.add_argument('--checkpoint_path_1', default='./checkpoints/final_param_LeNet5_MNIST_train_index_STE_4.pth')
    parser.add_argument('--checkpoint_path_2', default='./checkpoints/final_param_LeNet5_MNIST_train_index_STE_5.pth')
    parser.add_argument('--checkpoint_path_3', default='./checkpoints/final_param_LeNet5_MNIST_train_index_STE_6.pth')
    parser.add_argument('--discriminator_checkpoint_path_1', default='./checkpoints/final_param_discriminator_MNIST_1.pth')
    parser.add_argument('--discriminator_checkpoint_path_2', default='./checkpoints/final_param_discriminator_MNIST_2.pth')
    parser.add_argument('--discriminator_checkpoint_path_3', default='./checkpoints/final_param_discriminator_MNIST_3.pth')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")