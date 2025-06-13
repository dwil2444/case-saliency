import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from utils import imagenet_reverse_transform, cifar100_reverse_transform
# from .unpooled_case import OCAM as CASE
from .case import CASE as CASE
# from .case_logits import OCAM_TopKLogits as CASE
from utils.logger import CustomLogger
from utils.misc import GetDevice, load_model_and_targets, imagenet_label_mapper, cifar100_label_mapper
from load_datasets.imagenet import load_imagenet_datasets
from load_datasets.cifar100 import load_cifar100_datasets
from pytorch_grad_cam import GradCAM, HiResCAM, AblationCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .metrics import top_k_coords, agreement_coords

logger = CustomLogger(__name__).logger

cam_classes = {
    'grad_cam': GradCAM,
    'hires_cam': HiResCAM,
    'ablation_cam': AblationCAM,
    'grad_cam_plus_plus': GradCAMPlusPlus,
    'score_cam': ScoreCAM,
    'layer_cam': LayerCAM,
    'case': CASE
}

# Dataset + reverse transform maps
dataset_loader_map = {
    'imagenet': load_imagenet_datasets,
    'cifar100': load_cifar100_datasets,
}

reverse_transform_map = {
    'imagenet': imagenet_reverse_transform,
    'cifar100': cifar100_reverse_transform,
}

def generate_pair_heatmaps(model_name='resnet',
                           alg='grad_cam',
                           index=0,
                           k=5.0,
                           dataset='imagenet',
                           data_root='',
                           class_file='',
                           contrast_k=2,
                           output_dir='outputs/'):

    if dataset not in dataset_loader_map:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from {list(dataset_loader_map.keys())}")

    device = GetDevice()
    model, target_layers = load_model_and_targets(model_name, device)
    _, valset = dataset_loader_map[dataset](data_root, include_train=False)
    reverse_transform = reverse_transform_map[dataset]

    # Load and prepare image
    image_tensor, _ = valset[index]
    input_tensor = image_tensor.unsqueeze(0).to(device)
    original_image = reverse_transform(input_tensor.squeeze()).permute(1, 2, 0).cpu().detach().numpy()
    sm = nn.Softmax(dim=-1)
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = sm(logits)
        top2 = torch.topk(probs, 2, dim=-1).indices[0]
        c1, c2 = top2[0].item(), top2[1].item()

    if dataset == 'imagenet':
        label1 = imagenet_label_mapper(c1, class_file)
        label2 = imagenet_label_mapper(c2, class_file)
    elif dataset == 'cifar100':
        label1 = cifar100_label_mapper(c1, class_file)
        label2 = cifar100_label_mapper(c2, class_file)
    logger.info(f"Top-1 class: {label1}, Top-2 class: {label2}")

    try:
        if alg == 'case':
            cam = CASE(model=model,
                       target_layers=[target_layers[-1]],
                       use_cuda=True,
                       contrast_k=contrast_k)
            exp1 = cam(input_tensor=input_tensor, targets=[c1]).squeeze(0)
            exp2 = cam(input_tensor=input_tensor, targets=[c2]).squeeze(0)
        else:
            cam_class = cam_classes[alg]
            kwargs = {'model': model, 'target_layers': [target_layers[-1]]}
            if 'use_cuda' in cam_class.__init__.__code__.co_varnames:
                kwargs['use_cuda'] = True
            cam_instance = cam_class(**kwargs)
            exp1 = cam_instance(input_tensor, targets=[ClassifierOutputTarget(c1)])[0, :]
            exp2 = cam_instance(input_tensor, targets=[ClassifierOutputTarget(c2)])[0, :]

        exp1 = np.maximum(exp1, 0)
        exp2 = np.maximum(exp2, 0)

        # Normalize for overlay
        exp1 /= (np.max(exp1) + 1e-8)
        exp2 /= (np.max(exp2) + 1e-8)

        # 👉 Compute top-k agreement
        coords1 = top_k_coords(exp1, k)
        coords2 = top_k_coords(exp2, k)
        agreement = agreement_coords(coords1, coords2)
        logger.info(f"Top-{k:.1f}% feature agreement between {label1} and {label2}: {agreement:.2f}")

        # Convert original image to numpy
        original_np = np.array(original_image)

        # Plot original + side-by-side overlays with colorbars
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_np)
        axes[0].set_title('Original Image', pad=12)
        axes[0].axis('off')

        #axes[1].imshow(original_np)
        heat1 = axes[1].imshow(exp1, cmap='jet', alpha=0.9,)
        axes[1].set_title(f'Class: {label1}', pad=12)
        axes[1].axis('off')
        cbar1 = plt.colorbar(heat1, ax=axes[1], fraction=0.046, )
        cbar1.set_label('Saliency')

        #axes[2].imshow(original_np)
        heat2 = axes[2].imshow(exp2, cmap='jet', alpha=0.9)
        axes[2].set_title(f'Class: {label2}', pad=12)
        axes[2].axis('off')
        cbar2 = plt.colorbar(heat2, ax=axes[2], fraction=0.046, pad=0.04)
        cbar2.set_label('Saliency')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{alg}_{model_name}_{dataset}_idx{index}.pdf"
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path, format='pdf', bbox_inches='tight')
        logger.info(f"Saved heatmap pair to {full_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Failed to generate heatmaps: {str(e)}")

if __name__ == "__main__":
    import fire
    fire.Fire(generate_pair_heatmaps)
