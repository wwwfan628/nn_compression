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
from utils.plot import plot_params_distribution, plot_tensor_distribution, plot_distribution, plot_dict
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# used for hook
Call_forward_hook_times = 0
Call_backward_hook_times = 0
Model_name = None
Model_depth = 0
LeNet5_layer_names = ['Layer1:CNN','Layer1:CNN ReLu',
                     'Layer2:CNN','Layer2:CNN ReLu',
                     'Layer3:Linear','Layer3:Linear ReLu',
                     'Layer4:Linear']
LeNet5_param = {key:[] for i,key in enumerate(LeNet5_layer_names) if i%2==0}
LeNet5_param_grad = {key:[] for i,key in enumerate(LeNet5_layer_names) if i%2==0}
LeNet5_layer_out = {key:[] for key in LeNet5_layer_names}
LeNet5_layer_out_grad = {key:[] for key in LeNet5_layer_names}
VGG_layer_names = ['Layer1:CNN','Layer1:CNN ReLu',
                   'Layer2:CNN','Layer2:CNN ReLu',
                   'Layer3:CNN','Layer3:CNN ReLu',
                   'Layer4:CNN','Layer4:CNN ReLu',
                   'Layer5:CNN','Layer5:CNN ReLu',
                   'Layer6:CNN','Layer6:CNN ReLu',
                   'Layer7:Linear','Layer7:Linear ReLu',
                   'Layer8:Linear','Layer8:Linear ReLu',
                   'Layer9:Linear']
VGG_param = {key:[] for i,key in enumerate(VGG_layer_names) if i%2==0}
VGG_param_grad = {key:[] for i,key in enumerate(VGG_layer_names) if i%2==0}
VGG_layer_out = {key:[] for key in VGG_layer_names}
VGG_layer_out_grad = {key:[] for key in VGG_layer_names}

# tensorboard
writer = SummaryWriter()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")

def main(args):

    # check whether outputs & checkpoints directory exist
    path = os.path.join(os.getcwd(), '../outputs')
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(os.getcwd(), '../checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)

    # load dataset
    in_channels, num_classes, dataloader_train, dataloader_test = load_dataset(args.dataset_name)

    global Model_name, Model_depth
    # build neural network
    if args.model_name == 'LeNet5':
        Model_name = 'LeNet5'
        Model_depth = len(LeNet5_layer_names)
        model = LeNet5(in_channels=in_channels, num_classes=num_classes).to(device)
    elif 'VGG' in args.model_name:
        Model_name = 'VGG'
        Model_depth = len(VGG_layer_names)
        model = VGG_small(in_channels=in_channels, num_classes=num_classes).to(device)
    elif 'ResNet' in args.model_name:
        model = ResNet(in_channels=in_channels, num_classes=num_classes).to(device)
    else:
        print('Architecture not supported! Please choose from: LeNet5, VGG and ResNet.')

    # plot after initialization
    l = [module for module in model.modules() if
         isinstance(module, nn.ReLU) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)]
    for layer_idx, layer in enumerate(l):
        if layer_idx % 2 == 0:
            if Model_name == 'LeNet5':
                LeNet5_param[LeNet5_layer_names[layer_idx]].append(layer.weight)
            elif 'VGG' in Model_name:
                VGG_param[VGG_layer_names[layer_idx]].append(layer.weight)
    if Model_name == 'LeNet5':
        param_fig = plot_dict(LeNet5_param)
    elif 'VGG' in Model_name:
        param_fig =plot_dict(VGG_param)
    writer.add_figure(tag="Distribution of Parameter, Initilization", figure=param_fig)

    # train
    if args.train_index:
        init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '_train_index.pth'
        # save initial parameters
        torch.save(model.state_dict(), init_param_path)
        train(model, dataloader_train, dataloader_test, train_index=True)
    else:
        init_param_path = './checkpoints/init_param_' + args.model_name + '_' + args.dataset_name + '.pth'
        # save initial parameters
        torch.save(model.state_dict(), init_param_path)
        train(model, dataloader_train, dataloader_test)


def load_dataset(dataset_name, batch_size=128):
    # load dataset
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_train = MNIST(root='../datasets', train=True, download=True, transform=transform)
        data_test = MNIST(root='../datasets', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
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



def train(model, dataloader_train, dataloader_test, train_index=False, max_epoch=100, lr=1e-3, patience=10):
    dur = []  # duration for training epochs
    loss_func = nn.CrossEntropyLoss()
    if train_index:
        #optimizer = Index_SGD(model.parameters(), lr=1e-2, momentum=0.9)  # for VGG
        optimizer = Index_Adam(model.parameters())   # for LeNet5
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
        # add hook function to every layer
        if epoch % 10 == 0:
            # clear dict to save new values
            if Model_name == 'LeNet5':
                global LeNet5_param, LeNet5_param_grad, LeNet5_layer_out, LeNet5_layer_out_grad
                LeNet5_param = {key: [] for i, key in enumerate(LeNet5_layer_names) if i % 2 == 0}
                LeNet5_param_grad = {key: [] for i, key in enumerate(LeNet5_layer_names) if i % 2 == 0}
                LeNet5_layer_out = {key: [] for key in LeNet5_layer_names}
                LeNet5_layer_out_grad = {key: [] for key in LeNet5_layer_names}
            elif 'VGG' in Model_name:
                global VGG_param, VGG_param_grad, VGG_layer_out, VGG_layer_out_grad
                VGG_param = {key: [] for i, key in enumerate(VGG_layer_names) if i % 2 == 0}
                VGG_param_grad = {key: [] for i, key in enumerate(VGG_layer_names) if i % 2 == 0}
                VGG_layer_out = {key: [] for key in VGG_layer_names}
                VGG_layer_out_grad = {key: [] for key in VGG_layer_names}
            l = [module for module in model.modules() if
                 isinstance(module, nn.ReLU) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)]
            handles = []
            for layer in l:
                handles.append(layer.register_forward_hook(save_layer_output))
                handles.append(layer.register_backward_hook(save_grad_output))
        for i, (images, labels) in enumerate(dataloader_train):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()
            # save parameter's gradient per 10 epochs
            if epoch % 10 == 0:
                for layer_idx, layer in enumerate(l):
                    if layer_idx % 2 == 0:
                        if Model_name == 'LeNet5':
                            LeNet5_param_grad[LeNet5_layer_names[layer_idx]].append(layer.weight.grad)
                        elif 'VGG' in Model_name:
                            VGG_param_grad[VGG_layer_names[layer_idx]].append(layer.weight.grad)
        # plot distribution each 10 epochs
        if epoch % 10 == 0:
            for layer_idx, layer in enumerate(l):
                if layer_idx % 2 == 0:
                    if Model_name == 'LeNet5':
                        LeNet5_param[LeNet5_layer_names[layer_idx]].append(layer.weight)
                    elif 'VGG' in Model_name:
                        VGG_param[VGG_layer_names[layer_idx]].append(layer.weight)
            if Model_name == 'LeNet5':
                param_fig, param_grad_fig, layer_out_fig, layer_out_grad_fig = \
                    plot_distribution(LeNet5_param, LeNet5_param_grad, LeNet5_layer_out, LeNet5_layer_out_grad)
            elif 'VGG' in Model_name:
                param_fig, param_grad_fig, layer_out_fig, layer_out_grad_fig = \
                    plot_distribution(VGG_param, VGG_param_grad, VGG_layer_out, VGG_layer_out_grad)
            writer.add_figure(tag="Distribution of Parameter, Training Epoch: {:03d}".format(epoch + 1), figure=param_fig)
            writer.add_figure(tag="Distribution of Parameter Gradient, Training Epoch: {:03d}".format(epoch + 1), figure=param_grad_fig)
            writer.add_figure(tag="Distribution of Layer Output, Training Epoch: {:03d}".format(epoch + 1), figure=layer_out_fig)
            writer.add_figure(tag="Distribution of Layer Output Gradient, Training Epoch: {:03d}".format(epoch + 1), figure=layer_out_grad_fig)
            for handle in handles:  # remove hooks
                handle.remove()
        # validate
        dur.append(time.time() - t0)
        train_accuracy = float(validate(model, dataloader_train))
        test_accuracy = float(validate(model, dataloader_test))
        print("Epoch {:05d} | Training Acc {:.4f}% | Test Acc {:.4f}% | Time(s) {:.4f}".format(epoch + 1, train_accuracy, test_accuracy, np.mean(dur)))
        # plot accuracy
        writer.add_scalar('Train Accuracy', train_accuracy, epoch)
        writer.add_scalar('Test Accuracy', test_accuracy, epoch)
        # early stop
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            corresp_train_acc = train_accuracy
            best_epoch = epoch + 1
            cur_step = 0
            # save checkpoint
            if train_index:
                final_param_path = './checkpoints/final_param_' + args.model_name + '_' + args.dataset_name + '_train_index.pth'
            else:
                final_param_path = './checkpoints/final_param_' + args.model_name + '_' + args.dataset_name + '.pth'
            torch.save(model.state_dict(), final_param_path)
        else:
            cur_step += 1
            if cur_step == patience:
                break
    print("Training finished! Best test accuracy = {:.4f}%, corresponding training accuracy = {:.4f}%, "
          "found at Epoch {:05d}.".format(best_test_acc, corresp_train_acc, best_epoch))


def save_grad_output(self, grad_input, grad_output):
    global Call_backward_hook_times
    layer_idx = - ((Call_backward_hook_times % Model_depth)+1)
    if Model_name == 'LeNet5':
        layer_name = LeNet5_layer_names[layer_idx]
        LeNet5_layer_out_grad[layer_name].append(grad_output[0])
    elif 'VGG' in Model_name:
        layer_name = VGG_layer_names[layer_idx]
        VGG_layer_out_grad[layer_name].append(grad_output[0])
    #print("BACKWARD HOOK")
    #print("layer name: ", self.__class__.__name__)
    #print("grad_output size: ", len(grad_output), grad_output[0].shape)
    Call_backward_hook_times += 1

def save_layer_output(self, input, output):
    global Call_forward_hook_times
    layer_idx = Call_forward_hook_times % Model_depth
    if Model_name == 'LeNet5':
        layer_name = LeNet5_layer_names[layer_idx]
        LeNet5_layer_out[layer_name].append(output)
    elif 'VGG' in Model_name:
        layer_name = VGG_layer_names[layer_idx]
        VGG_layer_out[layer_name].append(output)
    #print("FORWARD HOOK")
    #print("layer name: ", self.__class__.__name__)
    #print("output size: ", len(output), output.shape)
    Call_forward_hook_times+=1


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Fixed Point")

    parser.add_argument('--dataset_name', default='MNIST', help='choose dataset from: MNIST, CIFAR10, ImageNet')
    parser.add_argument('--model_name', default='LeNet5', help='choose architecture from: LeNet5, VGG, ResNet18')
    parser.add_argument('--train_index', action='store_true', help='if true train index, else train in normal way')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")