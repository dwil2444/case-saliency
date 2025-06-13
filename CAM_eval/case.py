import numpy as np
import torch
import torch.nn as nn
from typing import Callable, List, Tuple
from tqdm import tqdm
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image
from utils.logger import CustomLogger
from .metrics import get_semantically_similar_classes

logger = CustomLogger(__name__).logger

class CASE:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 contrast_k: int = 5,
                 show_progress: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        self.contrast_k = contrast_k
        self.show_progress = show_progress

        if self.cuda:
            self.model = model.cuda()

        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    def get_cam_weights(self) -> torch.Tensor:
        grads = self.activations_and_grads.gradients[0]
        return grads.mean(dim=(2, 3))  # shape: (B, C)
    def forward(self, input_tensor: torch.Tensor, target: int) -> np.ndarray:
        if self.cuda:
            input_tensor = input_tensor.cuda()
        input_tensor = input_tensor.requires_grad_(True)

        outputs = self.activations_and_grads(input_tensor)
        activations = self.activations_and_grads.activations[0]  # (B, C, H, W)

        # 1. Compute gradient for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target] = 1
        outputs.backward(gradient=one_hot, retain_graph=True)
        grad_target = self.get_cam_weights().detach()

        # 2. Load semantic contrast set
        model_name = self.model.__class__.__name__.lower()
        if 'resnet' in model_name:
            model_name = 'resnet'
        elif 'densenet' in model_name:
            model_name = 'densenet'
        elif 'vgg' in model_name:
            model_name = 'vgg'
        elif 'convnext' in model_name:
            model_name = 'convnext'

        conf_path = f"/home/dw3zn/Desktop/Repos/CASE/conf_matrices/conf_matrix_{model_name}.npy"
        if not hasattr(self, 'conf_matrix'):
            self.conf_matrix = np.load(conf_path)
        contrast_classes = get_semantically_similar_classes(
            class_idx=target,
            confusion_matrix=self.conf_matrix,
            top_k=self.contrast_k
        )

        # 3. Streaming mean computation of contrast gradients
        avg_contrast_grad = None
        iterable = contrast_classes
        if self.show_progress:
            iterable = tqdm(contrast_classes, desc="CASE contrast gradients")
        for contrast_class in iterable:
            self.activations_and_grads.clear_activations_and_grads()
            self.model.zero_grad()
            outputs = self.activations_and_grads(input_tensor)
            one_hot = torch.zeros_like(outputs)
            one_hot[0, contrast_class] = 1
            outputs.backward(gradient=one_hot,
                             retain_graph=True)
            grad = self.get_cam_weights().detach()

            if avg_contrast_grad is None:
                avg_contrast_grad = grad
            else:
                avg_contrast_grad += grad
        avg_contrast_grad /= len(contrast_classes)

        # 4. Orthogonalize target gradient
        dot_product = (grad_target * avg_contrast_grad).sum(dim=1, keepdim=True)
        norm_sq = avg_contrast_grad.norm(dim=1, keepdim=True) ** 2 + 1e-8
        proj = (dot_product / norm_sq) * avg_contrast_grad
        orthogonal_grad = grad_target - proj

        # 5. Compute weighted saliency
        weights = orthogonal_grad[:, :, None, None]
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)

        target_size = self.get_target_width_height(input_tensor)
        scaled = scale_cam_image(cam.cpu().numpy(), target_size)
        return scaled

    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        return input_tensor.size(-1), input_tensor.size(-2)

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[int] = None) -> np.ndarray:
        return self.forward(input_tensor, targets[0])

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, Exception):
            logger.error(f"Error during OCAM execution: {exc_type}, {exc_value}")
            return True