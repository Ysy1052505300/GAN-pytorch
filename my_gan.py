import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.distributions
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.linear1 = nn.Linear(latent_dim, 128)
        self.leakyReLU1 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.leakyReLU2 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.leakyReLU3 = nn.LeakyReLU(0.2)
        self.linear4 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.leakyReLU4 = nn.LeakyReLU(0.2)
        self.linear5 = nn.Linear(1024, 784)

    def forward(self, z):
        # Generate images from z
        z = self.linear1(z)
        z = self.leakyReLU1(z)
        z = self.linear2(z)
        z = self.bn1(z)
        z = self.leakyReLU2(z)
        z = self.linear3(z)
        z = self.bn2(z)
        z = self.leakyReLU3(z)
        z = self.linear4(z)
        z = self.bn3(z)
        z = self.leakyReLU4(z)
        z = self.linear5(z)
        return torch.tanh(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.linear1 = nn.Linear(784, 512)
        self.leakyReLU1 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(512, 256)
        self.leakyReLU2 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(256, 1)


    def forward(self, imgs):
        # return discriminator score for img
        imgs = self.linear1(imgs)
        imgs = self.leakyReLU1(imgs)
        imgs = self.linear2(imgs)
        imgs = self.leakyReLU2(imgs)
        imgs = self.linear3(imgs)
        return torch.sigmoid(imgs)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):
    print('train')
    criterion = nn.BCELoss()
    # unloader = transforms.ToPILImage()
    for epoch in range(args.n_epochs):
        print(epoch)
        for i, (imgs, _) in enumerate(dataloader):
            bs, d1, d2, d3 = imgs.shape
            imgs.cuda()

            noise = torch.randn(bs, args.latent_dim).to(device)
            y_real = torch.ones(bs, 1).to(device)
            y_fake = torch.zeros(bs, 1).to(device)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            G_output = generator.forward(noise)
            D_output = discriminator.forward(G_output)
            G_loss = criterion(D_output, y_real)

            # gradient backprop & optimize ONLY G's parameters
            G_loss.backward(retain_graph=True)
            optimizer_G.step()


            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            x_real = imgs.reshape([bs, d2 * d3]).to(device)
            D_real_output = discriminator.forward(x_real)
            D_loss_real = criterion(D_real_output, y_real)
            D_loss_real.backward(retain_graph=True)

            x_fake = generator.forward(noise)
            D_fake_output = discriminator.forward(x_fake)
            D_loss_fake = criterion(D_fake_output, y_fake)
            D_loss_fake.backward(retain_graph=True)

            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #                 #            'images/{}.png'.format(batches_done),
                #                 #            nrow=5, normalize=True)

                torch.save(noise, './noise/noise')
                torch.save(generator.state_dict(), './g/g')
                torch.save(discriminator.state_dict(), './d/d')

                for index in range(25):
                    # print(index)
                    gen_image = G_output[index].reshape([d2, d3])
                    # gen_image = unloader(gen_image)
                    save_image(gen_image, './images/{}-{}.png'.format(batches_done, index), nrow=5, normalize=True)




def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)
    os.makedirs('noise', exist_ok=True)
    os.makedirs('g', exist_ok=True)
    os.makedirs('d', exist_ok=True)
    # load data
    dataloader = DataLoader(
        MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize([0.5],
                                                [0.5])])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    device = torch.device('cuda')

    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D ,device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
