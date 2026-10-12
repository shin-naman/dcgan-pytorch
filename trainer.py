"""The original DCGAN optimization loop and its output artifacts."""

import os

import numpy as np
import torch
import torchvision.utils as vutils
from torch import nn, optim


def train_gan(netG, netD, dataloader, device, args):
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)
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
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
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