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
    torch.set_num_threads(num_threads)

    # load dataset
    in_channels, num_classes, dataloader_train, dataloader_test = load_dataset(args.dataset_name)

    # build neural network
    if args.model_name == 'LeNet5':
        model = LeNet5(in_channels=in_channels, num_classes=2, normal_init=True).to(device)
        #model.load_state_dict(torch.load('checkpoints/final_param_discriminator_MNIST_4.pth'))
        #for parameter in list(model.parameters())[1:]:
        #    parameter.requires_grad = False
    elif 'VGG' in args.model_name:
        model = VGG_small(in_channels=in_channels, num_classes=2, normal_init=True).to(device)
    else:
        print('Architecture not supported! Please choose from: LeNet5, VGG.')

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
        batch_size = 512
        transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        data_train = ImageFolder(root='/srv/beegfs01/projects/imagenet/data/train/', transform=transform_train)
        data_test = ImageFolder(root='/srv/beegfs01/projects/imagenet/data/val/', transform=transform_val)
    else:
        print('Dataset is not supported! Please choose from: MNIST, CIFAR10 and ImageNet.')
    in_channels = data_train[0][0].shape[0]
    num_classes = len(data_train.classes)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    return in_channels, num_classes, dataloader_train, dataloader_test



def validate(model, dataloader_test, loss_func):
    # validate
    total = 0
    correct = 0
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader_test):
            # real image
            images = images.to(device)
            real_labels = torch.ones([images.shape[0]], dtype=int).to(device)
            real_x = model(images)
            _, real_pred = torch.max(real_x, 1)
            real_pred = real_pred.data.cpu()
            total += real_x.size(0)
            correct += torch.sum(real_pred == 1)
            real_loss = loss_func(real_x, real_labels)
            # random image
            random_images = shuffle_image(images.clone().detach().to(device))
            random_labels = torch.zeros([images.shape[0]], dtype=int).to(device)
            random_x = model(random_images)
            _, random_pred = torch.max(random_x, 1)
            random_pred = random_pred.data.cpu()
            total += random_x.size(0)
            correct += torch.sum(random_pred == 0)
            random_loss = loss_func(random_x, random_labels)
            losses.append(0.5 * (real_loss.item() + random_loss.item()))
    mean_loss = np.mean(np.array(losses))
    return correct*100.0/total, mean_loss


def shuffle_image(batch_image_tensor):
    random_batch_image_tensor = torch.zeros(batch_image_tensor.shape).to(device)
    for i, image_tensor in enumerate(batch_image_tensor):
        random_idx = torch.randperm(image_tensor.nelement())
        random_batch_image_tensor[i] = image_tensor.view(-1)[random_idx].view(image_tensor.size())
    return random_batch_image_tensor


def train(model, dataloader_train, dataloader_test, args):
    dur = []  # duration for training epochs
    loss_func = nn.CrossEntropyLoss()
    if args.train_index:
        if 'LeNet' in args.model_name:
            optimizer = Index_Adam_full(model.parameters(), lr=1e-3, ste=args.ste, params_prime=model.parameters(),
                                        granularity_channel=False, granularity_kernel=False)  # for LeNet5
        elif 'VGG' in args.model_name:
            # optimizer = Index_SGD_full(model.parameters(), lr=1e-2, momentum=0.9, ste=args.ste,
            #                            params_prime=model.parameters(), granularity_channel=args.granularity_channel,
            #                            granularity_kernel=args.granularity_kernel)  # for VGG
            optimizer = Index_SGD(model.parameters(), lr=1e-2, momentum=0.9)  # for VGG
    else:
        #optimizer = optim.SGD(model.parameters(), lr=args.lr)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_test_acc = 0
    best_test_loss = float('inf')
    corresp_train_acc = 0
    corresp_train_loss = 0
    best_epoch = 0
    cur_step = 0
    for epoch in range(args.max_epoch):
        # adjust lr
        if 'LeNet' in args.model_name or 'VGG' in args.model_name:
            optimizer.param_groups[0]['lr'] *= 0.99
        t0 = time.time()  # start time
        model.train()
        for i, (images, labels) in enumerate(dataloader_train):
            images = images.to(device)
            real_labels = torch.ones([images.shape[0]], dtype=int).to(device)
            random_images = shuffle_image(images.clone().detach().to(device))
            random_labels = torch.zeros([images.shape[0]], dtype=int).to(device)
            optimizer.zero_grad()
            real_pred = model(images)
            real_loss = loss_func(real_pred, real_labels)
            random_pred = model(random_images)
            random_loss = loss_func(random_pred, random_labels)
            loss = 0.5 * (real_loss + random_loss)
            loss.backward()
            optimizer.step()

        # validate
        dur.append(time.time() - t0)
        train_accuracy, train_mean_loss = validate(model, dataloader_train, loss_func)
        test_accuracy, test_mean_loss = validate(model, dataloader_test, loss_func)
        print("Epoch {:05d} | Training Acc {:.4f}% | Training Loss {:.4f}% | Test Acc {:.4f}% | Test Loss {:.4f}%| Time(s) {:.4f}"
              .format(epoch + 1, train_accuracy, train_mean_loss, test_accuracy, test_mean_loss, np.mean(dur)))

        # early stop
        if test_accuracy > best_test_acc or test_mean_loss < best_test_loss:
            best_test_acc = test_accuracy
            best_test_loss = test_mean_loss
            corresp_train_acc = train_accuracy
            corresp_train_loss = train_mean_loss
            best_epoch = epoch + 1
            cur_step = 0
            # save checkpoint
            final_param_path = args.final_param_path
            torch.save(model.state_dict(), final_param_path)
        else:
            cur_step += 1
            if cur_step == args.patience:
                break
    print("Training finished! Best test accuracy = {:.4f}%, test accuracy = {:.4f},"
          "corresponding training accuracy = {:.4f}%, training loss = {:.4f}, found at Epoch {:03d}."
          .format(best_test_acc, best_test_loss, corresp_train_acc, corresp_train_loss, best_epoch))



if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Traditional Training")

    parser.add_argument('--dataset_name', default='MNIST', help='choose dataset from: MNIST, CIFAR10, ImageNet')
    parser.add_argument('--model_name', default='LeNet5', help='choose architecture from: LeNet5, VGG, ResNet18')
    parser.add_argument('--final_param_path', default='./checkpoints/final_param_discriminator_MNIST.pth')
    parser.add_argument('--train_index', action='store_true', help='if true train index, else train in normal way')
    parser.add_argument('--max_epoch', type=int, default=100, help='max training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of optimizer')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
    parser.add_argument('--ste', action='store_true', help='if use straight through estimation or not')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")