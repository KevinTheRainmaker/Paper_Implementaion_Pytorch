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
    @classmethod
    def _download(cls):
        if not (lab.get_data_path() / 'cora').exists():
            download.download_file('https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
                                   lab.get_data_path() / 'cora.tgz')
            download.extract_tar(lab.get_data_path() /
                                 'cora.tgz', lab.get_data_path())

    def __init__(self, include_edges, bool=True):
        self.include_edges = include_edges
        self._download()

        with monit.section('Read Content'):
            content = np.genfromtxt(
                str(lab.get_data_path() / 'cora/cora.content'), dtype=np.dtype(str))

        with monit.section('Read Citations'):
            citations = np.genfromtxt(
                str(lab.get_data_path() / 'cora/cora.cites'), dtype=np.int32)

        # Get the feature vectors
        features = torch.tensor(np.array(content[:, 1:-1], dtype=np.float32))
        self.features = features / \
            features.sum(dim=1, keepdim=True)  # Normalize

        # Get class names and assign integer tags
        self.classes = {s: i for i, s in enumerate(set[content[: -1]])}
        # to labels
        self.labels = torch.tensor([self.classes[i]
                                   for i in content[: -1]], dtype=torch.long)

        # Get paper ids
        paper_ids = np.array(content[:, 0], dtype=np.int32)
        # Map id to idx
        ids_to_idx = {id_: i for i, id_ in enumerate(paper_ids)}

        # Empty adj_mat
        self.adj_mat = torch.eye(len(self.labels), dtype=torch.bool)

        # Mark the citations in adj_mat
        if self.include_edges:
            for e in citations:
                e1, e2 = ids_to_idx[e[0]], ids_to_idx[e[1]]

                self.adj_mat[e1][e2] = True
                self.adj_mat[e2][e1] = True

# Graph Attention Network with two GraphAttentionLayer


class GAT(Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()

        # First layer: concatenate the heads
        self.layer1 = GraphAttentionLayer(
            in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()

        # Second layer: average the heads
        self.output = GraphAttentionLayer(
            n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    # Stacking layers: x is features vectors of shape [n_nodes, in_features]
    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x, adj_mat)
        return x

    def accuracy(output: torch.Tensor, labels: torch.Tensor):
        return output.argmax(dim=-1).eq(labels).sum().item() / len(labels)
