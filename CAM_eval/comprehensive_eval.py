import os
import csv
import fire
import torch
import numpy as np
import torch.nn as nn
import inspect
from datetime import datetime
from .case import CASE as CASE
from .metrics import top_k_coords
from CWOX_HiRES.cwox_wrapper import CWOX_Wrapper
from utils.logger import CustomLogger
from utils.misc import GetDevice, zero_region_torch, load_model_and_targets, imagenet_label_mapper
from load_datasets.imagenet import load_imagenet_datasets
from pytorch_grad_cam import GradCAM, HiResCAM, AblationCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
logger = CustomLogger(__name__).logger


cam_classes = {
    'grad_cam': GradCAM,
    'hires_cam': HiResCAM,
    'ablation_cam': AblationCAM,
    'grad_cam_plus_plus': GradCAMPlusPlus,
    'score_cam': ScoreCAM,
    'layer_cam': LayerCAM,
}

ALL_METHODS = list(cam_classes.keys()) + ['case']
# ALL_METHODS = ['case', 'layer_cam']

def has_use_cuda_arg(cls):
    return 'use_cuda' in inspect.signature(cls.__init__).parameters


def evaluate_all(model_name='resnet',
                 k=50.0,
                 data_root='',
                 class_file='',
                 output_csv='./results.csv',
                 contrast_k=1,
                 num_samples=500):
    import random
    random.seed(42)
    np.random.seed(42)

    device = GetDevice()
    sm = nn.Softmax(dim=-1)
    model, target_layers = load_model_and_targets(model_name, device)
    _, valset = load_imagenet_datasets(data_root, include_train=False)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['idx', 'alg', 'model', 'gt_label', 'pred_label', 'gt_conf', 'new_conf', 'avg_drop'])

        for alg in ALL_METHODS:
            logger.info(f"Evaluating method: {alg}")
            indices = np.random.choice(len(valset), size=num_samples, replace=False)
            for idx in indices:
                input_tensor = valset[idx][0].unsqueeze(0).to(device)
                lbl = valset[idx][1]
                model.eval()

                with torch.enable_grad():
                    gt_conf, gt_pred = torch.max(sm(model(input_tensor)), dim=-1)
                    gt_conf = gt_conf.item()
                    gt_pred = gt_pred.item()

                    if gt_pred != lbl:
                        continue

                    lbl_text = imagenet_label_mapper(lbl, class_file)
                    pred_text = imagenet_label_mapper(gt_pred, class_file)

                    try:
                        if alg == 'case':
                            cam = CASE(model=model,
                                       contrast_k=contrast_k,
                                       target_layers=[target_layers[-1]],
                                       use_cuda=True)
                            exp = cam(input_tensor=input_tensor, targets=[lbl]).squeeze(0)
                        elif alg in cam_classes:
                            cam_class = cam_classes[alg]
                            kwargs = {'model': model, 'target_layers': [target_layers[-1]]}
                            if has_use_cuda_arg(cam_class):
                                kwargs['use_cuda'] = True
                            cam_instance = cam_class(**kwargs)
                            conf_targets = [ClassifierOutputTarget(lbl)]
                            exp = cam_instance(input_tensor, targets=conf_targets)[0, :]
                        elif alg == 'cwox':
                            cam = CWOX_Wrapper(model=model, target_layers=target_layers, use_cuda=True,
                                               htlm_path=os.environ.get('IMAGENET_DIR', data_root))
                            exp = cam(input_tensor=input_tensor, target=lbl)
                        else:
                            raise ValueError(f"Unsupported algorithm: {alg}")

                        if exp is None:
                            logger.error("No Explanation Produced")
                            continue

                        exp = np.nan_to_num(exp, nan=0.0, posinf=0.0, neginf=0.0)
                        exp = np.maximum(exp, 0)
                        top_coords = top_k_coords(exp, k)

                        res = zero_region_torch(image=input_tensor.squeeze(), pixel_list=top_coords, device=device)
                        new_conf = sm(model(res.unsqueeze(0)))[0, gt_pred].item()
                        delta_conf = gt_conf - new_conf
                        ad = max(delta_conf, 0) / gt_conf

                        writer.writerow([idx, alg, model_name, lbl_text, pred_text, f"{gt_conf:.4f}", f"{new_conf:.4f}", f"{ad:.6f}"])

                    except Exception as e:
                        logger.error(f"Error processing idx={idx}, alg={alg}: {str(e)}")
                        continue

if __name__ == "__main__":
    fire.Fire(evaluate_all)
