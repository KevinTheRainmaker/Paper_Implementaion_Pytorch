from operator import is_
from turtle import shearfactor
import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.graphs.gat.experiment import Configs as GATConfigs

from graph_attention_v2 import GraphAttentionV2Layer


class GATv2(Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int,
                 n_heads: int, dropout: float, share_weights: bool = True):
        super().__init__()

        # First layer: concatenate the heads
        self.layer1 = GraphAttentionV2Layer(in_features, n_hidden, n_heads,
                                            is_concat=True, dropout=dropout, share_weights=share_weights)
        # activation function: ELU
        self.activation = nn.ELU()

        # Second layer: average the heads
        self.layer2 = GraphAttentionV2Layer(n_hidden, n_heads, 1,
                                            is_concat=False, dropout=dropout, share_weights=share_weights)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        # apply dropout to the inputs
        x = self.dropout(x)

        # first layer
        x = self.layer1(x, adj_mat)

        # activation function
        x = self.activation(x)

        # apply dropout to layer2
        x = self.dropout(x)

        # output layer
        x = self.layer2(x, adj_mat)
