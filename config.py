"""Command-line arguments and basic DCGAN configuration validation."""

import argparse


def parse_args():
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
    parser.add_argument('--resume', default='', help='path to a full training checkpoint, e.g. output/checkpoints/latest.pth')

    args = parser.parse_args()

    if args.imageSize != 64:
        parser.error("This DCGAN architecture currently requires --imageSize 64.")

    if args.nc != 3:
        parser.error("This data pipeline currently supports RGB images only: --nc 3.")

    if args.resume and (args.netG or args.netD):
        parser.error('--resume cannot be combined with --netG or --netD (weights-only).')

    return args