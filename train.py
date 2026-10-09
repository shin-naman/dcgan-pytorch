import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


def main():
    #parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='number of channels in the input image')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='size of feature maps in discriminator')
    parser.add_argument('--niter', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--ngpu', type=int, default=1, choices=[0, 1], help='0: force CPU; 1: use one available GPU, otherwise CPU')
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")

    args = parser.parse_args()

    if args.imageSize != 64:
        parser.error("This DCGAN architecture currently requires --imageSize 64.")

    if args.nc != 3:
        parser.error("This data pipeline currently supports RGB images only: --nc 3.")

    os.makedirs(args.outf, exist_ok=True)

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    print(args)
    print("Random Seed: ", args.manualSeed)
    
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    torch.use_deterministic_algorithms(True)

    # Select the device. --ngpu 0 explicitly forces CPU.
    if args.ngpu == 0:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(args.imageSize),
                                transforms.CenterCrop(args.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=args.workers)

    # Temporary data-loading check.
    print(f"Dataset images: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    ngpu = int(args.ngpu)
    nz = int(args.nz)
    nc = int(args.nc)
    ngf = int(args.ngf)
    ndf = int(args.ndf)

    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Generator Code
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. ``(ngf*8) x 4 x 4``
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. ``(ngf*4) x 8 x 8``
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. ``(ngf*2) x 16 x 16``
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. ``(ngf) x 32 x 32``
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. ``(nc) x 64 x 64``
            )

        def forward(self, input):
            return self.main(input)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG, map_location=device, weights_only=True))
    print(netG)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is ``(nc) x 64 x 64``
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf) x 32 x 32``
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*2) x 16 x 16``
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*4) x 8 x 8``
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*8) x 4 x 4``
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD, map_location=device, weights_only=True))
    print(netD)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progress
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Training Loop
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(args.niter):
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, args.niter, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or (
                (epoch == args.niter - 1) and (i == len(dataloader) - 1)
            ):
                netG.eval()
                with torch.no_grad():
                    samples = netG(fixed_noise).cpu()
                netG.train()

                vutils.save_image(
                    samples,
                    os.path.join(args.outf, f'fake_iter_{iters:06d}.png'),
                    nrow=8,
                    padding=2,
                    normalize=True,
                    value_range=(-1, 1)
                )

            iters += 1

        torch.save(
            netG.state_dict(),
            os.path.join(args.outf, f'netG_epoch_{epoch + 1:03d}.pth')
        )
        torch.save(
            netD.state_dict(),
            os.path.join(args.outf, f'netD_epoch_{epoch + 1:03d}.pth')
        )
        np.savez(
            os.path.join(args.outf, 'losses.npz'),
            G_losses=G_losses,
            D_losses=D_losses
        )
        print(f'Saved model states and losses for epoch {epoch + 1}.', flush=True)

    print('Training complete. Outputs saved to:', os.path.abspath(args.outf))


if __name__ == '__main__':
    main()

