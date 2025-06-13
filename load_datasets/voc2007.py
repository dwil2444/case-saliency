import argparse
from decouple import config
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

from typing import Tuple
from utils.logger import CustomLogger
logger = CustomLogger(__name__).logger

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class VOC2007Dataset:
    def __init__(self, root_dir: str = config('VOC_DIR'), 
                 image_set: str='train', 
                 transform=transform):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform

        self.voc_dataset = VOCDetection(root=root_dir, 
                                        year='2007', 
                                        image_set=image_set,
                                        download=True, 
                                        transform=None)

    def __len__(self) -> int:
        return len(self.voc_dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor,
                                        torch.Tensor]:
        image, target = self.voc_dataset[idx]
        if self.transform:
            image = self.transform(image)
        logger.info(type(image))
        logger.info(type(target))
        return image, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataRoot', type=str,
                        default=config('VOC_DIR'),
                        help='Proportion of Top Saliency Feature: (1/k)')
    parser.add_argument('--batch_size', type=int,
                        default=16,
                        help='Batch of Images for dataloader')
    args = parser.parse_args()
    dataRoot = args.dataRoot
    batch_size = args.batch_size
    root_dir = dataRoot
    train_dataset = VOC2007Dataset(root_dir, image_set='trainval', transform=transform)
    val_dataset = VOC2007Dataset(root_dir, image_set='val', transform=transform)
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False)


if __name__ == "__main__":
    main()