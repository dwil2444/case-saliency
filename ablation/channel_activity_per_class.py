import os
import csv
import fire
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.logger import CustomLogger
from utils.misc import GetDevice, load_model_and_targets, imagenet_label_mapper
from load_datasets.imagenet import load_imagenet_datasets

logger = CustomLogger(__name__).logger

# return model.features.denseblock4.denselayer32.conv2
# return  model.features.norm5

def get_activation_layer(model_name: str, model: torch.nn.Module):
    model_name = model_name.lower()
    if 'resnet' in model_name:
        return model.layer4[2].conv3
    elif 'densenet' in model_name:
        return model.features.norm5
    elif 'vgg' in model_name:
        return model.features[-1]
    elif 'convnext' in model_name:
        return model.features[7][2].block[0]
    else:
        raise ValueError(f"Unrecognized model type: {model_name}")


def count_active_channels(model, dataloader, device, class_file, threshold=0.1, output_csv='active_channel_counts.csv', max_images=None):
    results = []

    def hook_fn(module, input, output):
        hook_fn.output = output.detach()

    hook_layer = get_activation_layer(model.__class__.__name__, model)
    handle = hook_layer.register_forward_hook(hook_fn)

    model.eval().to(device)
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for idx, (img, label) in enumerate(dataloader):
            if max_images and idx >= max_images:
                break

            img = img.to(device)
            label = label.item()
            _ = model(img)
            pred = torch.argmax(model(img)).item()

            if pred != label:
                continue  # skip if prediction is incorrect

            activation = hook_fn.output.squeeze(0)  # [C, H, W]
            pooled = F.adaptive_avg_pool2d(activation.unsqueeze(0), (1, 1)).squeeze()  # [C]
            pooled = pooled.cpu().numpy()

            num_active = np.sum(pooled > threshold)
            label_text = imagenet_label_mapper(label, class_file)

            results.append((idx, label_text, int(num_active)))

    handle.remove()

    # Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'label', 'active_channel_count'])
        for row in results:
            writer.writerow(row)

    logger.info(f"[DONE] Saved {len(results)} entries to {output_csv}")


def analyze_active_channels(model_name='resnet',
                            data_root='./imagenet_val/',
                            class_file='./imagenet_classes.txt',
                            threshold=0.1,
                            output_csv='active_channel_counts.csv',
                            max_images=None):
    device = GetDevice()
    model, _ = load_model_and_targets(model_name, device=device)
    _, valset = load_imagenet_datasets(data_root, include_train=False)
    dataloader = DataLoader(valset, batch_size=1, shuffle=False)

    logger.info(f"[INFO] Counting active channels for correct predictions on {model_name}...")
    count_active_channels(model, dataloader, device, class_file, threshold, output_csv, max_images)


if __name__ == '__main__':
    fire.Fire(analyze_active_channels)
