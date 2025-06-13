import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from utils import imagenet_reverse_transform, cifar100_reverse_transform
from .case import CASE as CASE
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
                           output_dir='outputs/'):

    if dataset not in dataset_loader_map:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from {list(dataset_loader_map.keys())}")

    device = GetDevice()
    model, target_layers = load_model_and_targets(model_name, device)
    _, valset = dataset_loader_map[dataset](data_root, include_train=False)
    reverse_transform = reverse_transform_map[dataset]

    image_tensor, lbl = valset[index]
    logger.info(f"Ground Truth Label: {imagenet_label_mapper(lbl, class_file)}")
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
            cam = CASE(model=model, target_layers=[target_layers[-1]], use_cuda=True, contrast_k=2)
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
        exp1 /= (np.max(exp1) + 1e-8)
        exp2 /= (np.max(exp2) + 1e-8)

        # Threshold for "no explanation"
        NO_EXPLANATION_THRESHOLD = 1e-6
        no_exp1 = np.max(exp1) < NO_EXPLANATION_THRESHOLD
        no_exp2 = np.max(exp2) < NO_EXPLANATION_THRESHOLD

        # Generate binary masks or black out if no explanation
        if not no_exp1:
            k_pixels = int(exp1.size * (k / 100.0))
            thresh1 = np.partition(exp1.flatten(), -k_pixels)[-k_pixels]
            mask1 = (exp1 >= thresh1).astype(np.float32)
            masked1 = original_image * mask1[..., np.newaxis]
        else:
            masked1 = np.zeros_like(original_image)
            logger.warning(f"{alg}: No meaningful explanation for class {label1} (Top-1)")

        if not no_exp2:
            k_pixels = int(exp2.size * (k / 100.0))
            thresh2 = np.partition(exp2.flatten(), -k_pixels)[-k_pixels]
            mask2 = (exp2 >= thresh2).astype(np.float32)
            masked2 = original_image * mask2[..., np.newaxis]
        else:
            masked2 = np.zeros_like(original_image)
            logger.warning(f"{alg}: No meaningful explanation for class {label2} (Top-2)")

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(masked1)
        axes[1].set_title(f'Top {k:.0f}%: {label1}' + (" (None)" if no_exp1 else ""))
        axes[1].axis('off')

        axes[2].imshow(masked2)
        axes[2].set_title(f'Top {k:.0f}%: {label2}' + (" (None)" if no_exp2 else ""))
        axes[2].axis('off')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"top{k:.0f}percent_{alg}_{model_name}_{dataset}_idx{index}.pdf"
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path, format='pdf', bbox_inches='tight')
        logger.info(f"Saved top-{k:.0f}% heatmap to {full_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Failed to generate top-{k:.0f}% heatmaps: {str(e)}")

if __name__ == "__main__":
    import fire
    fire.Fire(generate_pair_heatmaps)
