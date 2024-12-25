import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# set random seed
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_loaders(data_name, batch_size=64):
    """
    Returns the data loaders for the specified dataset (MNIST or CIFAR).
    """
    # Determine the root directory for storing data
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    if data_name.lower() == "mnist":
        # Transformations for MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
        ])
        # Load MNIST datasets
        train_dataset = datasets.MNIST(
            root=root_dir,
            train=True,
            transform=transform,
            download=True
        )
        test_dataset = datasets.MNIST(
            root=root_dir,
            train=False,
            transform=transform,
            download=True
        )
    elif data_name.lower() == "cifar":
        # Transformations for CIFAR
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
        ])
        # Load CIFAR10 datasets
        train_dataset = datasets.CIFAR10(
            root=root_dir,
            train=True,
            transform=transform,
            download=True
        )
        test_dataset = datasets.CIFAR10(
            root=root_dir,
            train=False,
            transform=transform,
            download=True
        )
    else:
        raise ValueError(f"Dataset {data_name} is not supported. Choose 'mnist' or 'cifar'.")
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# Function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)