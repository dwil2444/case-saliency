import argparse
from .case import CASE
from decouple import config
import logging
from load_datasets.imagenet import load_imageNet_Datasets, idx_to_label
import matplotlib.pyplot as plt
import math
import os
from .metrics import top_k_coords, agreement_coords
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import sys
import torch as torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from CWOX_HiRES.cwox_wrapper import CWOX_Wrapper
from utils.misc import GetDevice, CleanCuda, sample_random_idx, blur_region_torch
from utils.logger import CustomLogger
from utils import reverse_transform
logger = CustomLogger(__name__).logger


reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Resize(256),
    transforms.CenterCrop(224)
])


def main():
    device = GetDevice()
    idx  = args.idx
    k = args.k
    model_name = args.model
    alg = args.alg
    eta = args.eta
    theta = args.theta
    omega = args.omega
    sm = nn.Softmax(dim=-1)
    if model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        target_layers = [model.conv1, model.layer4[2].conv3]
    elif model_name == 'vgg':
        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).to(device)
        target_layers  = [model.features[0], model.features[49]]
    elif model_name == 'densenet':
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1).to(device)
        target_layers = [model.features.conv0, model.features.denseblock4.denselayer32.conv2]
    elif model_name == 'convnext':
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1).to(device)
        target_layers = [model.features[0][0], model.features[7][2].block[0]]
    # low rank model -> do not load
    # model.load_state_dict(torch.load(f'/u/dw3zn/Repos/CASCADE/ablation/weights/{model_name}.pth'))
    _, valset = load_imageNet_Datasets(config('IMAGENET_DIR'))
    rand_idx = idx
    logger.info(f'Random Index Generated: {rand_idx}') 
    input_tensor = valset[rand_idx][0].unsqueeze(0).to(device)
    lbl = valset[rand_idx][1]
    model.eval()
    all_agreement = []
    with torch.enable_grad():
        #rgb_img = reverse_transform(input_tensor.squeeze()).permute(1,2,0).cpu().detach().numpy()
        gt_conf, gt_pred = torch.max(sm(model(input_tensor.to(device))), dim=-1)
        gt_conf = gt_conf.item()
        gt_pred = gt_pred.item()
        pretty_conf = round(gt_conf, 2)
        if gt_pred == lbl:
            logger.info(gt_pred)
            lbl_text = idx_to_label(idx=lbl,
                            file_path=config('CLASS_FILE'))
            logger.info(f'Label Text:  {lbl_text}')
            # get the explanation for top-k based on the method
            if alg == 'grad_cam':
                cam = GradCAM(model=model,
                    target_layers = [target_layers[-1]],
                    )
                conf_targets = [ClassifierOutputTarget(lbl)]
                exp = cam(input_tensor, targets=conf_targets)[0, :]
                exp = np.maximum(exp, 0)
            elif alg == 'hires_cam':
                cam = HiResCAM(model=model,
                    target_layers=[target_layers[-1]],
                    use_cuda=True)
                conf_targets = [ClassifierOutputTarget(lbl)]
                exp = cam(input_tensor, targets=conf_targets)[0, :]
                #exp = np.maximum(exp, 0)
            elif alg == 'case':
                cam =CASE(model=model,
                         target_layers=[target_layers[-1]],
                         use_cuda=True,)
                exp = cam(input_tensor=input_tensor,
                          targets=lbl)
                if len(exp.shape) > 2:
                    exp = exp[0]
                exp = np.maximum(exp, 0) # only positive activations
            elif alg == 'cwox':
                cam = CWOX_Wrapper(model=model, 
                    target_layers=target_layers,
                    use_cuda=True,
                    htlm_path=config('IMAGENET_DIR'))
                exp = cam(input_tensor=input_tensor,
                        target=lbl,)
                # check the shape of CWOX
            nonzero = np.count_nonzero(exp == 0)
            logger.info(f'Proportion of Zero Entries: {nonzero/(224*224)}')
            if exp is not None:
                exp = np.maximum(exp, 0)
                top_coords = top_k_coords(exp, k)
                # top_coords = [ (y, x) for x, y in top_coords ] # flip for gaussian blur
                logger.info(f'Number of Pixels Considered: {len(top_coords)}')
                res = blur_region_torch(image=input_tensor.squeeze(),
                        pixel_list=top_coords,
                        kernel_size=5,
                        device=device) # justify kernel size

                blurred_probs = sm(model(res.to(device).unsqueeze(0)))
                new_conf = blurred_probs[0, gt_pred].item()
            else:
                new_conf = gt_conf # if there is no explanation, then produce no score
            # delta_conf = np.log(gt_conf - new_conf) - np.log(k)
            delta_conf =  max((gt_conf - new_conf) , 0) / gt_conf
            logger.info(f'Model: {model_name} | Algorithm: {alg} | Confidence Delta: {delta_conf:.10f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        default='resnet',
                        help='Model to Evaluate')
    parser.add_argument('--idx', type=int,
                        default=0,
                        help='Index of the validation set')
    parser.add_argument('--k', type=float,
                        default=50.0,
                        help='Top Percentage of Pixels to keep (e.g. 50)')
    parser.add_argument('--alg', type=str, 
                        default='grad_cam', 
                        help='algorithm to compute confidence drop for')
    parser.add_argument('--eta',
                        type=float,
                        default='100.0',
                        help='How much information is shared in representations')
    parser.add_argument('--theta',
                        type=float,
                        default='1.0',
                        help='How much shared information is used')
    parser.add_argument('--omega',
                        type=float,
                        default='0.5',
                        help='What proportion of singular components are used to reflect efficacy')
    args = parser.parse_args()
    main()