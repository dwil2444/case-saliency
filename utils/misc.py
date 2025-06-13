import gc as gc
import torch.nn as nn
import random
from torchvision import models
from typing import List, Tuple
import numpy as np
from utils.logger import CustomLogger
import torch as torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from typing import Union



logger = CustomLogger(__name__).logger


def CleanCuda():
    gc.collect()
    torch.cuda.empty_cache()


def GetDevice():
    """

    """
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        # print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logger.info(f'Device name: {torch.cuda.get_device_name(0)}')

    else:
        logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def sample_random_idx(dataset):
    """
        Args: (dataset)

        Returns: index
    """
    rand_idx = random.randint(0, len(dataset)-1)
    return rand_idx


def generate_random_number_in_range_excluding_index(start, end, exclude_index):
    """
    Args:
        start:
        end:
        exclude_index:
    """
    # Generate a list of numbers within the specified range excluding the exclude_index
    numbers = [i for i in range(start, end+1) if i != exclude_index]
    # Choose a random number from the list
    random_number = random.choice(numbers)
    return random_number


def get_positive_indices_and_apply_mask(grad_tensor: torch.Tensor, 
                                        activation_tensor: torch.Tensor)-> torch.Tensor:
    """
    Args:
        grad_tensor: tensor to generate mask
        activation_tensor: tensor to apply mask to

    Returns: positive activations with positive gradients
    """
    # # Convert grad_tensor to a boolean tensor
    # mask = (grad_tensor > 0).float()
    # # Use boolean masking to get the subset of activations
    # masked_activations = mask * activation_tensor
    # #return torch.abs(masked_activations)
    # return torch.clamp(masked_activations, min=0.0).float()
    # # TODO: Review impact of positive vs negative activations
    # #return torch.clamp(masked_activations, min=0.0)
    hirescam = grad_tensor * activation_tensor
    return torch.clamp(hirescam, min=0.0)



def blur_region_torch(image:torch.Tensor, 
                      pixel_list: np.ndarray, 
                      kernel_size:int=5, 
                      device='cuda') -> torch.Tensor:
    """
    Blur a specific region in the image defined by a list of pixels using PyTorch.

    Parameters:
    - image: The input image as a PyTorch tensor (CHW format).
    - pixel_list: A list of pixel coordinates (x, y) in the format [(x1, y1), (x2, y2), ...].
    - kernel_size: Size of the Gaussian kernel for blurring.
    - device: Device for the computation ('cuda' or 'cpu').

    Returns:
    - The image with the specified region blurred as a PyTorch tensor.
    """
    # Convert the pixel list to a tensor
    pixel_tensor = torch.tensor(pixel_list, dtype=torch.float32, device=device)
    # Create an empty mask
    mask = torch.zeros_like(image)
    # Set pixels in the mask to 1 for the specified region
    for pixel in pixel_tensor:
        x, y = pixel.long()
        mask[:, y, x] = 1
    # Apply Gaussian blur to the specified region
    blurred_region = transforms.functional.gaussian_blur(image,
                                                         kernel_size=kernel_size, 
                                                         sigma=None,) * mask

    # Combine the original image and the blurred region using the mask
    result = image * (1 - mask) + blurred_region
    return result


def zero_region_torch(image: torch.Tensor,
                      pixel_list: List[Tuple[int, int]],
                      device='cuda') -> torch.Tensor:
    """
    Set specific regions in the image defined by a list of pixels to zero using PyTorch.

    Parameters:
    - image: The input image as a PyTorch tensor (CHW format).
    - pixel_list: A list of pixel coordinates (x, y) in the format [(x1, y1), (x2, y2), ...].
    - device: Device for the computation ('cuda' or 'cpu').

    Returns:
    - The image with the specified region set to zero as a PyTorch tensor.
    """
    # Convert the pixel list to a tensor
    pixel_tensor = torch.tensor(pixel_list, dtype=torch.float32, device=device)
    # Create an empty mask
    mask = torch.zeros_like(image)
    # Set pixels in the mask to 1 for the specified region
    for pixel in pixel_tensor:
        x, y = pixel.long()
        mask[:, y, x] = 1
    # Zero out the specified region in the image
    result = image * (1 - mask)
    return result

# model.features.denseblock4.denselayer32.conv2]
# model.features.norm5]

def load_model_and_targets(model_name: str, device: str = 'cuda'):
    model_name = model_name.lower()

    if model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        target_layers = [model.conv1, model.layer4[2].conv3]

    elif model_name == 'vgg':
        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).to(device)
        target_layers = [model.features[0], model.features[49]]

    elif model_name == 'densenet':
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1).to(device)
        target_layers = [model.features.conv0, model.features.denseblock4.denselayer32.conv2]

    elif model_name == 'convnext':
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1).to(device)
        target_layers = [model.features[0][0], model.features[7][2].block[0]]

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model.eval(), target_layers


def imagenet_label_mapper(value: Union[int, str], file_path: str) -> Union[str, int]:
    """
    Maps an ImageNet class index to its label, or vice versa.

    Args:
        value (int or str): Either the integer index (0–999) or class label string.
        file_path (str): Path to a text file with one class label per line (in order).

    Returns:
        str if input is int, int if input is str
    """
    with open(file_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    if isinstance(value, int):
        if 0 <= value < len(labels):
            return labels[value]
        else:
            raise IndexError(f"Index {value} out of bounds for labels file with {len(labels)} entries.")
    elif isinstance(value, str):
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        if value in label_to_index:
            return label_to_index[value]
        else:
            raise ValueError(f"Label '{value}' not found in class names file.")
    else:
        raise TypeError("Value must be either an integer index or a string label.")


def cifar100_label_mapper(value: Union[int, str], file_path: str) -> Union[str, int]:
    """
    Maps a CIFAR-100 class index to its label, or vice versa.

    Args:
        value (int or str): Either the integer index (0–99) or class label string.
        file_path (str): Path to a text file with one CIFAR-100 label per line.

    Returns:
        str if input is int, int if input is str
    """
    with open(file_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    if isinstance(value, int):
        if 0 <= value < len(labels):
            return labels[value]
        else:
            raise IndexError(f"Index {value} out of bounds for labels file with {len(labels)} entries.")
    elif isinstance(value, str):
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        if value in label_to_index:
            return label_to_index[value]
        else:
            raise ValueError(f"Label '{value}' not found in class names file.")
    else:
        raise TypeError("Value must be either an integer index or a string label.")


def adjust_model_for_num_classes(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Adjust the final classification layer of a torchvision model to match the desired number of output classes.

    Args:
        model: A torchvision model (e.g., resnet50, vgg16, densenet121).
        num_classes: Number of output classes for the new task.

    Returns:
        The model with its final classification layer replaced.
    """
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        # For ResNet, ResNeXt, etc.
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        # For VGG, AlexNet
        if isinstance(model.classifier[-1], nn.Linear):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        # For DenseNet
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unsupported model architecture for classifier replacement.")
    return model


def get_target_layers(model, model_name):
    model_name = model_name.lower()
    if model_name == 'resnet':
        return [model.conv1, model.layer4[2].conv3]
    elif model_name == 'vgg':
        return [model.features[0], model.features[49]]
    elif model_name == 'densenet':
        return [model.features.conv0, model.features.denseblock4.denselayer32.conv2]
    elif model_name == 'convnext':
        return [model.features[0][0], model.features[7][2].block[0]]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
