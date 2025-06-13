import os
import csv
import fire
import torch
import numpy as np
import torch.nn as nn
import inspect
from typing import List, Tuple
from CAM_eval.ocam_case import OCAM as CASE
from utils.logger import CustomLogger
from utils.misc import GetDevice, cifar100_label_mapper, get_target_layers
from pytorch_grad_cam import GradCAM, AblationCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from CAM_eval.metrics import top_k_coords, agreement_coords
from load_datasets.cifar100 import load_cifar100_datasets
logger = CustomLogger(__name__).logger


cam_classes = {
    'grad_cam': GradCAM,
    'ablation_cam': AblationCAM,
    'grad_cam_plus_plus': GradCAMPlusPlus,
    'score_cam': ScoreCAM,
    'layer_cam': LayerCAM,
}

ALL_METHODS = list(cam_classes.keys())

def has_use_cuda_arg(cls):
    return 'use_cuda' in inspect.signature(cls.__init__).parameters


def evaluate_agreement_custom(model_path: str,
                              model_name: str = 'resnet',
                              k: float = 5.0,
                              data_root: str = '',
                              class_file: str = '',
                              output_csv: str = 'agreement_results.csv',
                              num_samples: int = 500):

    device = GetDevice()
    sm = nn.Softmax(dim=-1)

    # Load model checkpoint
    logger.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)

    # NOTE: Assumes last conv layer is named consistently for target_layers
    target_layers = get_target_layers(model, model_name)
    _, valset = load_cifar100_datasets(data_root, include_train=False)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    total_len = len(valset)
    sample_indices = np.random.choice(total_len, size=min(num_samples, total_len), replace=False)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['idx', 'alg', 'model', 'top1_label', 'top2_label', 'agreement'])

        for alg in ALL_METHODS:
            logger.info(f"Evaluating method: {alg}")
            for idx in sample_indices:
                input_tensor = valset[idx][0].unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = sm(logits)
                    top2 = torch.topk(probs, 2, dim=-1).indices[0]
                    c1, c2 = top2[0].item(), top2[1].item()

                try:
                    if alg == 'case':
                        pass
                        cam = CASE(model=model, target_layers=target_layers, use_cuda=True)
                        exp1 = cam(input_tensor=input_tensor, targets=[c1]).squeeze(0)
                        exp2 = cam(input_tensor=input_tensor, targets=[c2]).squeeze(0)
                    elif alg in cam_classes:
                        cam_class = cam_classes[alg]
                        kwargs = {'model': model, 'target_layers': [target_layers[-1]]}
                        if has_use_cuda_arg(cam_class):
                            kwargs['use_cuda'] = True
                        cam_instance = cam_class(**kwargs)
                        exp1 = cam_instance(input_tensor, targets=[ClassifierOutputTarget(c1)])[0, :]
                        exp2 = cam_instance(input_tensor, targets=[ClassifierOutputTarget(c2)])[0, :]
                    else:
                        raise ValueError(f"Unsupported algorithm: {alg}")

                    if exp1 is None or exp2 is None:
                        logger.error("No Explanation Produced")
                        continue

                    exp1 = np.nan_to_num(np.maximum(exp1, 0), nan=0.0, posinf=0.0, neginf=0.0)
                    exp2 = np.nan_to_num(np.maximum(exp2, 0), nan=0.0, posinf=0.0, neginf=0.0)

                    coords1 = top_k_coords(exp1, k)
                    coords2 = top_k_coords(exp2, k)
                    agreement = agreement_coords(coords1, coords2)

                    label1 = cifar100_label_mapper(c1, class_file)
                    label2 = cifar100_label_mapper(c2, class_file)

                    writer.writerow([idx, alg, model_name, label1, label2, f"{agreement:.2f}"])

                except Exception as e:
                    logger.error(f"Error processing idx={idx}, alg={alg}: {str(e)}")
                    continue


if __name__ == "__main__":
    fire.Fire(evaluate_agreement_custom)
