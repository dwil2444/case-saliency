import os
import csv
import fire
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.logger import CustomLogger
from utils.misc import GetDevice, load_model_and_targets, imagenet_label_mapper
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


def extract_topk_channels_per_class(model, dataloader, device, k=10):
    channel_use_by_class = defaultdict(list)

    def hook_fn(module, input, output):
        hook_fn.output = output.detach()

    hook_layer = get_activation_layer(model.__class__.__name__, model)
    handle = hook_layer.register_forward_hook(hook_fn)
    model.eval().to(device)

    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            label = label.item()
            _ = model(img)
            feat = hook_fn.output  # shape: [1, C, H, W]
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze()  # shape: [C]
            topk = torch.topk(pooled, k).indices.cpu().numpy()
            channel_use_by_class[label].append(set(topk))

    handle.remove()
    return channel_use_by_class


def compute_mean_interclass_jaccard(channel_use_by_class):
    classes = list(channel_use_by_class.keys())
    jaccard_vals = []

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            c1 = classes[i]
            c2 = classes[j]

            all_c1 = set().union(*channel_use_by_class[c1])
            all_c2 = set().union(*channel_use_by_class[c2])

            if all_c1 or all_c2:
                intersection = len(all_c1 & all_c2)
                union = len(all_c1 | all_c2)
                jaccard = intersection / union if union > 0 else 0
                jaccard_vals.append(jaccard)

    return np.mean(jaccard_vals)


def analyze_channel_overlap(model_name='resnet',
                            data_root='./imagenet_val/',
                            k=10,
                            output_csv=None):
    device = GetDevice()
    model, _ = load_model_and_targets(model_name, device=device)
    _, valset = load_imagenet_datasets(data_root, include_train=False)
    dataloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)
    logger.info(f"[INFO] Extracting top-{k} channels for {model_name}...")
    channel_use_by_class = extract_topk_channels_per_class(model, dataloader, device, k=k)
    jaccard = compute_mean_interclass_jaccard(channel_use_by_class)

    logger.info(f"[RESULT] {model_name} | Mean Inter-Class Jaccard Overlap (Top-{k} channels): {jaccard:.4f}")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'top_k', 'mean_interclass_jaccard'])
            writer.writerow([model_name, k, f"{jaccard:.4f}"])


if __name__ == '__main__':
    fire.Fire(analyze_channel_overlap)


