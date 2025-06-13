import os
import csv
import fire
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.logger import CustomLogger
from utils.misc import GetDevice, load_model_and_targets
from load_datasets.imagenet import load_imagenet_datasets

logger = CustomLogger(__name__).logger


def get_activation_layer(model_name: str, model: torch.nn.Module):
    model_name = model_name.lower()
    if 'resnet' in model_name:
        return model.layer4[2].conv3
    elif 'densenet' in model_name:
        return model.features.denseblock4.denselayer32.conv2
    elif 'vgg' in model_name:
        return model.features[-1]
    elif 'convnext' in model_name:
        return model.features[7][2].block[0]
    else:
        raise ValueError(f"Unrecognized model type: {model_name}")


def extract_classwise_spatial_maps(model, dataloader, device, max_per_class=20):
    activations = defaultdict(list)

    def hook_fn(module, input, output):
        hook_fn.output = output.detach()

    hook_layer = get_activation_layer(model.__class__.__name__, model)
    handle = hook_layer.register_forward_hook(hook_fn)
    model.eval().to(device)

    seen_per_class = defaultdict(int)

    with torch.no_grad():
        for img, label in dataloader:
            label = label.item()
            if seen_per_class[label] >= max_per_class:
                continue

            img = img.to(device)
            _ = model(img)
            feat = hook_fn.output.squeeze(0).cpu().numpy()  # [C, H, W]
            activations[label].append(feat)
            seen_per_class[label] += 1

    handle.remove()
    return activations


def compute_intra_class_spatial_consistency(activations):
    consistency_scores = []

    for label, samples in activations.items():
        samples = np.stack(samples)  # [N, C, H, W]
        mean_map = np.mean(samples, axis=0)  # [C, H, W]
        mean_flat = mean_map.flatten()
        mean_flat /= (np.linalg.norm(mean_flat) + 1e-8)

        for sample in samples:
            sample_flat = sample.flatten()
            sample_flat /= (np.linalg.norm(sample_flat) + 1e-8)
            sim = np.dot(sample_flat, mean_flat)
            consistency_scores.append(sim)

    return np.mean(consistency_scores)


def analyze_intra_class_spatial_consistency(model_name='resnet',
                                            data_root='./imagenet_val/',
                                            max_per_class=20,
                                            output_csv='intra_class_spatial_consistency.csv'):
    device = GetDevice()
    model, _ = load_model_and_targets(model_name, device=device)
    _, valset = load_imagenet_datasets(data_root, include_train=False)
    dataloader = DataLoader(valset, batch_size=1, shuffle=False)

    logger.info(f"[INFO] Extracting spatial maps for {model_name} (max {max_per_class} per class)...")
    activations = extract_classwise_spatial_maps(model, dataloader, device, max_per_class)
    consistency = compute_intra_class_spatial_consistency(activations)

    logger.info(f"[RESULT] {model_name} | Intra-Class Spatial Consistency: {consistency:.4f}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'max_per_class', 'mean_intra_class_spatial_consistency'])
        writer.writerow([model_name, max_per_class, f"{consistency:.4f}"])


if __name__ == '__main__':
    fire.Fire(analyze_intra_class_spatial_consistency)
