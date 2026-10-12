"""Entry point: configure, construct the components, and train."""

import os

import torch

from config import parse_args
from data import create_dataloader
from models import Discriminator, Generator, weights_init
from runtime import configure_runtime
from trainer import train_gan


def main():
    args = parse_args()
    os.makedirs(args.outf, exist_ok=True)
    device = configure_runtime(args)
    dataloader = create_dataloader(args)

    netG = Generator(nz=args.nz, nc=args.nc, ngf=args.ngf).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG, map_location=device, weights_only=True))
    print(netG)

    netD = Discriminator(nc=args.nc, ndf=args.ndf).to(device)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD, map_location=device, weights_only=True))
    print(netD)

    train_gan(netG, netD, dataloader, device, args)

if __name__ == '__main__':
    main()

