import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Any
from utils.logger import CustomLogger

logger = CustomLogger(__name__).logger

# Normalization stats for CIFAR-100
CIFAR100_MEAN = [0.509, 0.487, 0.442]
CIFAR100_STD = [0.202, 0.200, 0.204]

# Validation transform
# val_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
# ])
#
# # Training transform
# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(degrees=45),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
# ])

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
])


val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
])



def load_cifar100_datasets(data_root: str,
                           include_train: bool = False
                           ) -> Tuple[Optional[CIFAR100], CIFAR100]:
    train_dataset = None
    if include_train:
        train_dataset = CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR100(root=data_root, train=False, download=True, transform=val_transform)
    return train_dataset, val_dataset


def load_cifar100_dataloaders(data_root: str,
                              batch_size: int = 32,
                              include_train: bool = False,
                              num_workers: int = 4,
                              pin_memory: bool = True
                              ) -> Tuple[Optional[DataLoader], DataLoader]:
    train_dataset, val_dataset = load_cifar100_datasets(data_root, include_train)
    train_loader = None
    if train_dataset:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def main(data_root: str = "./data", batch_size: int = 16, include_train: bool = False):
    train_loader, val_loader = load_cifar100_dataloaders(data_root, batch_size, include_train)
    logger.info(f"Loaded CIFAR-100 test set with {len(val_loader.dataset)} samples.")
    if train_loader:
        logger.info(f"Loaded CIFAR-100 train set with {len(train_loader.dataset)} samples.")
        for images, labels in train_loader:
            logger.info(f"Sample train batch: images={images.shape}, labels={labels.shape}")
            break


if __name__ == "__main__":
    import fire
    fire.Fire(main)
