import argparse
from decouple import config
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from typing import Tuple, List
from utils.logger import CustomLogger
logger = CustomLogger(__name__).logger


# Define transforms for data preprocessing
val_transform = transforms.Compose([
    # transforms.Resize((32, 32)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.491, 0.482, 0.447],
                         std=[0.202, 0.200, 0.201])  # Normalize images
])


train_transform = transforms.Compose([
    # transforms.Resize((32, 32)),  # Resize images to a fixed size,
    transforms.RandomCrop(size=32, padding=4),  # Random crop with padding of 4 pixels
    transforms.RandomHorizontalFlip(),         # Random horizontal flipping
    transforms.RandomRotation(degrees=15),      # Random rotation within the range of -15 to +15 degrees
    transforms.ToTensor(),                      # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.491, 0.482, 0.447],
                         std=[0.202, 0.200, 0.201])  # Normalize images
])




class CIFAR10Dataset:
    def __init__(self, root_dir: str = config('PARTITION'),
                 train: bool = True,
                 transform=val_transform):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        self.cifar10_dataset = CIFAR10(root=root_dir,
                                         train=train,
                                         download=True,
                                         transform=None)

    def __len__(self) -> int:
        return len(self.cifar10_dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image, target = self.cifar10_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, target

    def compute_mean_std(self) -> Tuple[List[float], List[float]]:
        """
        Returns: mean and std per channel
        """
        transform = transforms.ToTensor()
        mean = [0.0, 0.0, 0.0]
        std = [0.0, 0.0, 0.0]
        for images, _ in self.cifar10_dataset:
            images = transform(images)
            mean = [m + val for m, val in zip(mean, images.mean(dim=(1, 2)).tolist())]
            std = [m + val for m, val in zip(std, images.std(dim=(1, 2)).tolist())]
        num_samples = len(self.cifar10_dataset)
        mean = [m / num_samples for m in mean]
        std = [s / num_samples for s in std]
        logger.info(f'Mean: {mean}')
        logger.info(f'Std : {std}')
        return mean, std



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataRoot', type=str,
                        default=config('PARTITION'),
                        help='Path to CIFAR-10 dataset')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size for DataLoader')
    args = parser.parse_args()
    data_root = args.dataRoot
    batch_size = args.batch_size

    train_dataset = CIFAR10Dataset(root_dir=data_root,
                                    train=True,
                                    transform=train_transform)
    test_dataset = CIFAR10Dataset(root_dir=data_root,
                                   train=False,
                                   transform=val_transform)
    train_dataset.compute_mean_std()
    test_dataset.compute_mean_std()
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    logger.info(len(train_loader))


if __name__ == "__main__":
    main()
