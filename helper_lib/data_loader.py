import torch
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size, train=True, dataset="CIFAR10"):
    if dataset.upper() == "MNIST":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        ds = datasets.MNIST(root=data_dir, train=train, download=True, transform=tfm)
    else:  # CIFAR10 (default)
        tfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
