from graph_attention_v2 import GraphAttentionV2Layer
import torch
from torch import nn
from torch import package

from labml import tracker, monit
from labml.configs import BaseConfigs
from labml_helpers.module import Module
from labml_helpers.device import DeviceConfigs

from build_cora import CoraDataset


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


def accuracy(output: torch.Tensor, labels: torch.Tensor):
    return output.argmax(dim=-1).eq(labels).sum().item() / len(labels)


class Configs(BaseConfigs):
    model: GATv2

    # Number of nodes to train on
    training_samples: int = 500

    # Number of features per node in the input
    in_features: int

    # Number of features in the first graph attention layer
    n_hidden: int = 64

    # Number of heads
    n_heads: int = 8

    # Number of classes for classification
    n_classes: int

    # Dropout probability
    dropout: float = 0.6

    # Whether to share weights for source & target nodes of edges
    share_weights: bool = False

    # Whether to include the citation network
    include_edges: bool = True

    dataset: CoraDataset
    epochs: int = 1_000
    loss_func = nn.CrossEntropyLoss()
    device: torch.device = DeviceConfigs()
    optimizer: torch.optim.Adam

    def run(self):
        '''
        데이터셋이 크지 않으므로 full batch training 이용
        '''
        # Move data to device
        features = self.dataset.features.to(self.device)
        labels = self.dataset.labels.to(self.device)
        edges_adj = self.dataset.adj_mat.to(self.device)
        edges_adj = edges_adj.unsqueeze(-1)

        idx_rand = torch.randperm(len(labels))
        # Nodes for training: 2,208개
        idx_train = idx_rand[:self.training_samples]
        # Nodes for validation: 500개
        idx_valid = idx_rand[self.training_samples:]

        # training loop: Monitoring with labml.monit
        for epoch in monit.loop(self.epochs):
            self.model.train()  # Set the model to training mode
            self.optimizer.zero_grad()  # Make all gradients to zero

            output = self.model(features, edges_adj)

            # Get the loss of training nodes
            loss = self.loss_func(output[idx_train], labels[idx_train])

            # Calculate gradients with backpropagation
            loss.backward()

            self.optimizer.step()

            # Logging: using labml.tracker
            tracker.add('loss.train', loss)
            tracker.add('accuracy.train', accuracy(
                output[idx_train], labels[idx_train]))

            self.model.eval()  # Set the model to evaluation mode
            # We do not need to claculate gradients
            with torch.no_grad():
                # Evaluate the model again
                output = self.model(features, edges_adj)
                loss = self.loss_func(output[idx_valid], labels[idx_valid])

                # Logging
                tracker.add('loss.valid', loss)
                tracker.add('accuracy.valid', accuracy(
                    output[idx_valid], labels[idx_valid]))

            # Save logs
            tracker.save()
            path = './GATv2/gat_v2.pt'
            package_name = 'gat'
            resource_name = 'model.pkl'

            with package.PackageExporter(path) as exp:
                exp.extern('graph_attention_v2')
                exp.extern('__main__')
                exp.save_pickle(package_name, resource_name, self.model)
