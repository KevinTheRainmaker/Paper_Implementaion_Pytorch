from typing import Dict

import numpy as np
import torch
from torch import nn

from graph_attention import GraphAttentionLayer

from labml import lab, monit, tracker, experiment
from labml.configs import BaseConfigs, option, calculate
from labml.utils import download
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_nn.optimizers.configs import OptimizerConfigs

'''
# Cora Dataset

2,708건의 Scientific Publication을 포함하는 데이터셋.
7개의 class로 분류되며, 각각의 publication을 노드로 하여 인용 네트워크를 이룬다.
해당 인용 네트워크는 5,429개의 link(edge)를 가지는 Directed Graph이다.

label prediction과 link prediction 등에서 Benchmarks Dataset로 사용된다.
'''

class CoraDataset:
    labels: torch.Tensor
    classes: Dict[str, int]
    features: torch.Tensor
    adj_mat: torch.Tensor
    
    # Download the Dataset
    @staticmethod
    def _download():
        if not (lab.get_data_path() / 'cora').exists():
            download.download_file('https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
                                   lab.get_data_path() / 'cora.tgz')
            download.extract_tar(lab.get_data_path() / 'cora.tgz', lab.get_data_path())
    