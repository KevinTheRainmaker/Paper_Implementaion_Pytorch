'''
Graph Attention Network(이하 GAT)의 PyTorch 구현

GATs는 그래프 데이터에 대해 masked self-attention을 사용한 구조로, 
graph attention layer가 범용적으로 사용된다.

self-attention이란 어떠한 입력을 이해하기 위해 같이 입력된 요소들 중
무엇을 중요하게 고려해야 하는지를 수치적으로 나타내는 기법이다.
즉, GAT는 이러한 self-attention 메카니즘을 node embedding 과정에
적용한 Neural Network이다.

각 graph attention layer는 node embedding을 input으로 받아 
transformed embedding을 output으로 출력한다.
'''

import torch
from torch import nn

from labml_helpers.module import Module

class GraphAttentionLayer(Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        '''
        in_features: node 당 input feature의 수
        out_features: node 당 output feature의 수
        n_heads: attention head의 수 K
        is_concat: multi-head 결과가 concated 될지 averaged 될지 설정
        dropout: dropout 되는 hidden layer 비율
        leaky_relu_negative_slope: leaky_relu_activation의 negative slope
        '''
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        
        # wheter the nulti-head results should be concatenated or averaged
        if is_concat:
            # Calculate the number of dimensional per head
            assert out_features % n_heads == 0, "out_features could'nt be divided into n_heads"
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        
        # Transform the node embeddings before self-attention (for initial transformation)
        self.linear = nn.linear(in_features, self.n_hidden * n_heads, bias=False) # bias=False: the layer will not learn an additive bias
        
        # Linear layer to compute attention score $e_{ij}$
        self.attention = nn.Linear(self.n_hidden * 2, 1, bias=False)
        
        # The activation for $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        '''
        h: Input으로 들어오는 node embedding의 shape (i.e. [n_nodes, in_features])
        adj_mat: adjacency matrix의 shape (i.e. [n_nodes, n_nodes, n_heads])
        '''
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        
        # Calculate attention score
        '''
        e_ij: attention score from node j to node i
        a: attention mechanism
        
        e_ij = LeakyReLU(a^T[g_i||g_j])
        '''
        g_repeat = g.repeat(n_nodes, 1, 1) # {g_1, g_2, ... , g_n_nodes, g_1, g_2, ...}
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0) # {g_1, g_1, ... , g_2, g_2, ... , g_n_nodes, ...}
        
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1) 
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden) # g_concat[i,j] = g_i||g_j
        
        # e_ij = LeakyReLU(a^T[g_i||g_j])
        e = self.activation(self.attn(g_concat)) # e.shape = [n_nodes, n_nodes, n_heads, 1]
        e = e.squeeze(-1) # remove last dimension
        
        # check adjacency matrix: [n_nodes, n_nodes, n_heads] or [n_nodes, n_nodes, 1]
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        
        # Mask e_ij based on adjacency matrix 
        e = e.masked_fill(adj_mat == 0, float('-inf')) # if there is no edges btw i and j, e_ij = -inf
        
        # Normalize attention scores
        # \alpha_ij = softmax_j(e_ij)
        a = self.softmax(e)
        a = self.dropout(a) # apply dropout regularization
        
        # Calculate final output for each head
        # ommited unlinearity \theta to appy it on GAT model later
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)
        
        # Concatenate the head
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_head * self.n_hidden)
        
        # Or mean
        else:
            return attn_res.mean(dim=1)