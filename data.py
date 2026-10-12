"""Dataset transformations and DataLoader construction."""

import torch
from torchvision import datasets, transforms


def create_dataloader(args):
    dataset = datasets.ImageFolder(root=args.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(args.imageSize),
                                    transforms.CenterCrop(args.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=args.workers)

    print(f"Dataset images: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    return dataloader