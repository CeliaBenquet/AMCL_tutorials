import torch
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader


def prepare_data(batch_size): 
    # transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # train and validation data
    train_data = datasets.MNIST(
        root='../input/data',
        train=True,
        download=True,
        transform=transform
    )
    val_data = datasets.MNIST(
        root='../input/data',
        train=False,
        download=True,
        transform=transform
    )

    # training and validation data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_data, val_data, train_loader, val_loader


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

