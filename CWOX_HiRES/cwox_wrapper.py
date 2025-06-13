from decouple import config
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import ttach as tta
from typing import Callable, List, Tuple
from utils.misc import GetDevice, CleanCuda, sample_random_idx, blur_region_torch


from CWOX.apply_hltm import *
from CWOX.IOX import IOX
from CWOX.CWOX_2s import CWOX_2s
from CWOX.plt_wox import plot_cwox
from CWOX.hires_cam_cwox import hires_cam_cwox
from CWOX.grad_cam_cwox import grad_cam_cwox


class CWOX_Wrapper:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 htlm_path:str = None):
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        self.htlm_path = htlm_path
        if self.cuda:
            self.model = model.cuda()
            
    
    def get_name(self) -> str:
        """
        """
        check = self.target_layers[-1]
        for name, module in self.model.named_modules():
            if module is check:
                return name
        return None
    
    
    def find_class_in_cluster(self, 
                              sal_dict: dict, 
                              label: int) -> np.ndarray:
        """
        Args:
            sal_dict: the dictionary containing saliency maps for each
                    label considered in respective clusters


            label: the integer value representing the class of interest

        Returns: the saliency map for the label of interest
        """
        query = str(label)
        keyList = list(sal_dict.keys())
        for key in keyList:
            if query in key:
                return sal_dict[key]
        return None
            
    
    def forward(self,
                input_tensor: torch.Tensor,
                target: int,) -> np.ndarray:
        sm = nn.Softmax(dim=-1)
        if self.cuda:
            input_tensor = input_tensor.cuda()
        scores = sm(self.model(input_tensor)).squeeze().cpu().detach().numpy()
        self.model.zero_grad()
        top_5 = np.argsort(scores)[::-1][:5]
        clusterify = apply_hltm(cut_level=0,
                              json_path=self.htlm_path)
        cluster_use_final = clusterify.get_cluster(top_5)
        layer_name = self.get_name()
        IOX_class=IOX(grad_cam_cwox(self.model
                                     ,layer=layer_name))

        sal_dict=CWOX_2s(input_tensor.detach(),
                 cluster_use_final,
                 cluster_method=IOX_class,
                 class_method=IOX_class,
                 delta=50,
                 multiple_output=False)
        return self.find_class_in_cluster(sal_dict, target)
    
    
    def __call__(self,
               input_tensor: torch.Tensor,
                target: int, ) -> np.ndarray:
        return self.forward(input_tensor,
                           target)
