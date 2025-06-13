import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from utils.logger import CustomLogger

logger = CustomLogger(__name__).logger

# Standard ImageNet preprocessing
imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_imagenet_datasets(data_root: str,
                           include_train: bool = False) -> Tuple[Optional[datasets.ImageNet], datasets.ImageNet]:
    """
    Loads the ImageNet train and validation datasets.

    Args:
        data_root (str): Root directory containing 'train' and 'val' folders.
        include_train (bool): Whether to load the training set.

    Returns:
        (train_dataset, val_dataset)
    """
    train_dataset = None
    if include_train:
        train_dataset = datasets.ImageNet(root=data_root, split='train', transform=imagenet_transform)
    val_dataset = datasets.ImageNet(root=data_root, split='val', transform=imagenet_transform)
    return train_dataset, val_dataset

def load_imagenet_dataloaders(data_root: str,
                               batch_size: int = 32,
                               include_train: bool = False) -> Tuple[Optional[DataLoader], DataLoader]:
    """
    Returns ImageNet DataLoaders.

    Args:
        data_root (str): Path to ImageNet data.
        batch_size (int): Dataloader batch size.
        include_train (bool): Whether to return a train loader.

    Returns:
        (train_loader, val_loader)
    """
    train_dataset, val_dataset = load_imagenet_datasets(data_root, include_train)
    train_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def main(data_root: str = "./data", batch_size: int = 16, include_train: bool = False):
    train_loader, val_loader = load_imagenet_dataloaders(data_root, batch_size, include_train)
    logger.info(f"Loaded ImageNet val set with {len(val_loader.dataset)} samples.")
    if train_loader:
        logger.info(f"Loaded ImageNet train set with {len(train_loader.dataset)} samples.")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
