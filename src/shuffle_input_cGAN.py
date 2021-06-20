import numpy as np
import os
import argparse
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")

writer = SummaryWriter()

class ste_function_gray(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gen_images, init_images):
        gen_pixels = gen_images.view(gen_images.size()[0], -1)
        init_pixels = init_images.view(init_images.size()[0], -1)
        for image_idx, image_init_pixels in enumerate(init_pixels):
            image_init_pixels_tmp, _ = torch.sort(image_init_pixels)
            image_init_pixels[torch.argsort(gen_pixels[image_idx])] = image_init_pixels_tmp
        init_pixels = init_pixels.view(init_images.size())
        return init_pixels

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, torch.zeros(grad_output.shape).to(device)


class ste_function_rgb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gen_images, init_images):
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        gen_pixels_grayscale = transform(gen_images).view(gen_images.size()[0], -1)
        init_pixels_grayscale = transform(init_images).view(init_images.size()[0], -1)
        tmp = torch.zeros(init_images.size()[2]*init_images.size()[3]).to(device)
        for image_idx, image_init_pixels in enumerate(init_images):
            init_idx = torch.argsort(init_pixels_grayscale[image_idx])
            gen_idx = torch.argsort(gen_pixels_grayscale[image_idx])
            image_init_pixels_tmp_0 = image_init_pixels[0].clone().detach().to(device).view(-1)[init_idx]
            tmp[gen_idx] = image_init_pixels_tmp_0
            image_init_pixels[0] = tmp.view(image_init_pixels[0].size())
            image_init_pixels_tmp_1 = image_init_pixels[1].clone().detach().to(device).view(-1)[init_idx]
            tmp[gen_idx] = image_init_pixels_tmp_1
            image_init_pixels[1] = tmp.view(image_init_pixels[1].size())
            image_init_pixels_tmp_2 = image_init_pixels[2].clone().detach().to(device).view(-1)[init_idx]
            tmp[gen_idx] = image_init_pixels_tmp_2
            image_init_pixels[2] = tmp.view(image_init_pixels[2].size())
        return init_images

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, torch.zeros(grad_output.shape).to(device)


# building generator
class Generator(nn.Module):
    def __init__(self, input_dim, num_classes, image_shape):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.latent_dim = 128
        self.image_shape = image_shape

        def init(n_input, n_output, normalize=True):
            layers = [nn.Linear(n_input, n_output)]
            if normalize:
                layers.append(nn.BatchNorm1d(n_output, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generator = nn.Sequential(
            *init(input_dim + num_classes, self.latent_dim),
            *init(self.latent_dim, self.latent_dim * 2),
            *init(self.latent_dim * 2, self.latent_dim * 4),
            *init(self.latent_dim * 4, self.latent_dim * 8),
            nn.Linear(self.latent_dim * 8, int(np.prod(image_shape))),
            nn.Tanh()
        )

    # torchcat needs to combine tensors
    def forward(self, noise, random_images, labels):
        # random_image_pixels = random_images.view(random_images.size()[0], -1).to(device)
        gen_input = torch.cat((self.label_embed(labels), noise), -1)
        img = self.generator(gen_input)
        img = img.view(img.size(0), *self.image_shape)
        if random_images.shape[1] == 1:
            img_quantized = ste_function_gray.apply(img, random_images.clone().detach().to(device))
        else:
            img_quantized = ste_function_rgb.apply(img, random_images.clone().detach().to(device))
        return img_quantized, img


class Discriminator(nn.Module):
    def __init__(self, num_classes, image_shape):
        super(Discriminator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.dropout = 0.4
        self.latent_dim = 512

        def init(n_input, n_output, normalize=True):
            layers = [nn.Linear(n_input, n_output)]
            if normalize:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.discriminator = nn.Sequential(
            *init(num_classes + int(np.prod(image_shape)), self.latent_dim, normalize=False),
            *init(self.latent_dim, self.latent_dim),
            *init(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_features = img.view(img.size(0), -1)
        input_features = torch.cat((img_features, self.label_embed(labels)), -1)
        validity = self.discriminator(input_features)
        return validity


# weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def shuffle_images(images_batch):
    random_images_batch = torch.zeros(images_batch.shape).to(device)
    for i, image in enumerate(images_batch):
        if image.shape[0] == 1:
            # randomize the input image
            random_idx = torch.randperm(image[0].nelement())
            random_images_batch[i, 0] = image[0].view(-1)[random_idx].view(image[0].size())
        else:
            # randomize the input image
            random_idx = torch.randperm(image[0].nelement())
            random_images_batch[i, 0] = image[0].view(-1)[random_idx].view(image[0].size())
            random_images_batch[i, 1] = image[1].view(-1)[random_idx].view(image[1].size())
            random_images_batch[i, 2] = image[2].view(-1)[random_idx].view(image[2].size())
    return random_images_batch


def train(generator, discriminator, dataloader_train, dataloader_test, args):
    # Loss functions
    loss_func = torch.nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta, args.beta1))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta, args.beta1))

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    # Labels
    real_label = 0.9
    fake_label = 0.0

    # training
    for epoch in range(args.max_epoch):
        for i, (images, labels) in enumerate(dataloader_train):
            batch_size = images.shape[0]

            # convert img, labels into proper form
            images = images.to(device)
            labels = labels.to(device)

            # creating real and fake tensors of labels
            real_labels = torch.zeros([batch_size, 1]).fill_(real_label).to(device)
            fake_labels = torch.zeros([batch_size, 1]).fill_(fake_label).to(device)

            # initializing gradient
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            #### TRAINING GENERATOR ####
            # Feeding generator random shuffled images and labels
            random_images = shuffle_images(images.clone().detach().to(device)).to(device)
            gen_labels = labels.clone().detach().to(device)
            #noise = torch.tensor(np.random.normal(0, 1, (batch_size, args.input_dim))).float().to(device)
            noise = random_images.view(random_images.size()[0], -1).to(device)

            gen_imgs_quantized, gen_imgs = generator(noise, random_images.clone().detach().to(device), gen_labels)

            # Ability for discriminator to discern the real v generated images
            validity = discriminator(gen_imgs, gen_labels)

            # Generative loss function
            g_loss = loss_func(validity, real_labels)

            # Gradients
            g_loss.backward()
            g_optimizer.step()

            #### TRAINING DISCRIMINTOR ####
            d_optimizer.zero_grad()

            # Loss for real images and labels
            validity_real = discriminator(images, labels)
            d_real_loss = loss_func(validity_real, real_labels)

            # Loss for fake images and labels
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = loss_func(validity_fake, fake_labels)

            # Total discriminator loss
            d_loss = 0.5 * (d_fake_loss + d_real_loss)

            # calculates discriminator gradients
            d_loss.backward()
            d_optimizer.step()

            if i == 100:
                images_grid = vutils.make_grid(images, normalize=True)
                random_images_grid = vutils.make_grid(random_images, normalize=True)
                writer.add_image(tag='real_images', img_tensor=images_grid, global_step=epoch)
                writer.add_image(tag='random_images', img_tensor=random_images_grid, global_step=epoch)
                #vutils.save_image(images, '%s/real_images_epoch_%03d.png' % (args.output, epoch), normalize=True)
                #vutils.save_image(random_images, '%s/random_images_epoch_%03d.png' % (args.output, epoch), normalize=True)
                fake_quantized, fake = generator(noise, random_images, gen_labels)
                fake_quantized_grid = vutils.make_grid(fake_quantized, normalize=True)
                fake_grid = vutils.make_grid(fake, normalize=True)
                writer.add_image(tag='gen_images_quantized', img_tensor=fake_quantized_grid, global_step=epoch)
                writer.add_image(tag='gen_images', img_tensor=fake_grid, global_step=epoch)
                #vutils.save_image(fake_quantized.detach(), '%s/gen_images_quantized_epoch_%03d.png' % (args.output, epoch), normalize=True)
                #vutils.save_image(fake.detach(), '%s/gen_images_epoch_%03d.png' % (args.output, epoch), normalize=True)

        print("[Epoch: %d/%d]" "[D loss: %f]" "[G loss: %f]" % (epoch + 1, args.max_epoch, d_loss.item(), g_loss.item()))
        # plot loss functions
        writer.add_scalars('Generator & Discriminator Losses', {'Discriminator Loss': d_loss.item(),
                                                                'Generator Loss': g_loss.item()}, epoch)

        # checkpoints
        if epoch % 100 == 0:
            torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % (args.output, epoch))
            torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (args.output, epoch))


def load_dataset(dataset_name):
    # load dataset
    if dataset_name == 'MNIST':
        batch_size = 64
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        data_train = MNIST(root='../datasets', train=True, download=True, transform=transform)
        data_test = MNIST(root='../datasets', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        batch_size = 64
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        data_train = CIFAR10(root='../datasets', train=True, download=True, transform=transform)
        data_test = CIFAR10(root='../datasets', train=False, download=True, transform=transform)
    else:
        print('Dataset is not supported! Please choose from: MNIST or CIFAR10.')
    in_channels = data_train[0][0].shape[0]
    num_classes = len(data_train.classes)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, pin_memory=True)
    return in_channels, num_classes, dataloader_train, dataloader_test


def main(args):

    os.makedirs(args.output, exist_ok=True)

    if args.randomseed is None:
        args.randomseed = random.randint(1, 10000)
    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)

    # load dataset
    in_channels, num_classes, dataloader_train, dataloader_test = load_dataset(args.dataset_name)

    if args.dataset_name == 'MNIST':
        image_shape = (in_channels, 28, 28)
    elif args.dataset_name == 'CIFAR10':
        image_shape = (in_channels, 32, 32)
    input_dim = int(np.prod(image_shape))

    # Building generator
    generator = Generator(input_dim, num_classes, image_shape)

    # Building discriminator
    discriminator = Discriminator(num_classes, image_shape)
    discriminator.apply(init_weights)

    train(generator, discriminator, dataloader_train, dataloader_test, args)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Shuffle Input cGAN")

    parser.add_argument('--dataset_name', default='MNIST', help='choose dataset from: MNIST, CIFAR10, ImageNet')
    parser.add_argument('--model_name', default='LeNet5', help='choose architecture from: LeNet5, VGG, ResNet18')
    parser.add_argument('--input_dim', type=int, default=100, help='if true train index, else train in normal way')
    parser.add_argument('--max_epoch', type=int, default=200, help='max optimization iteration')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of optimizer')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
    parser.add_argument('--latent_dim', type=int, default=100, help='size of latent vector')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for adam optimizer')
    parser.add_argument('--output', default='./outputs', help='folder to output images and model checkpoints')
    parser.add_argument('--randomseed', type=int, help='seed')

    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")