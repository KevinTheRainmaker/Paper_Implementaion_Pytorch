'''
GAT v2 operator의 PyTorch 기반 구현.
GAT v2는 Graph data에 대해 GAT와 유사하게 동작한다.
GAT의 Static attention 문제를 해결한 것이 GAT v2의 핵심.
Static attention 문제는 키 노드의 attention이 쿼리 노드와
동일한 rank를 가질 때 발생하는 문제로, attention mechanism을
수정함으로써 dynamic attention을 가능하게 하여 해결하였다.

GAT: e(h_i, h_j) = LeakyReLU(a^T[W_{h_i}||W_{h_j}])
GATv2: e(h_i, h_j) = a^TLeakyReLU(W[h_i||h_j])
'''

import torch
from torch import nn

from labml_helpers.module import Module


class GraphAttentionV2Layer(Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        '''
        in_features: node 당 input feature의 수
        out_features: node 당 output feature의 수
        n_heads: attention head의 수 K
        is_concat: multi-head 결과가 concated 될지 averaged 될지 설정
        dropout: dropout 되는 hidden layer 비율
        leaky_relu_negative_slope: leaky_relu_activation의 negative slope
        share_weights: True로 설정 시 동일한 행렬이 모든 edge에 대한 source와 target에 apply 됨
        '''
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # wheter the nulti-head results should be concatenated or averaged
        if is_concat:
            # Calculate the number of dimensional per head
            assert out_features % n_heads == 0, "out_features should be divided into n_heads"
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        # Linear layer for initial source transformation before self-attention
        # bias=False: the layer will not learn an additive bias
        self.linear_l = nn.Linear(
            in_features, self.n_hidden * n_heads, bias=False)

        # if share_weights = True
        if share_weights:
            # the same linear layer is used for the target nodes
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(
                in_features, self.n_hidden * n_heads, bias=False)

        # Linear layer to compute attention score $e_{ij}$
        self.attention = nn.Linear(self.n_hidden, 1, bias=False)

        # The activation for $e_{ij}$
        self.activation = nn.LeakyReLU(
            negative_slope=leaky_relu_negative_slope)

        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        '''
        h: Input으로 들어오는 node embedding의 shape (i.e. [n_nodes, in_features])
        adj_mat: adjacency matrix의 shape (i.e. [n_nodes, n_nodes, n_heads] = [n_nodes, n_nodes, 1])
        '''

        # The initial transformations for each head
        # Two linear transformations and then split up for each head
        n_nodes = h.shape[0]
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)

        # Calculate attention score
        '''
        e_ij: attention score from node j to node i
        a: attention mechanism
        
        e_ij = LeakyReLU(a^T[g_i||g_j])
        '''
        g_l_repeat = g_l.repeat(
            n_nodes, 1, 1)  # {gl_1, gl_2, ... , gl_n_nodes, gl_1, gl_2, ...}

        # {gr_1, gr_1, ... , gr_2, gr_2, ... , gr_n_nodes, ...}
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)

        g_concat = g_l_repeat + g_r_repeat_interleave

        # g_concat[i,j] = vec(gli) + vec(grj)
        g_concat = g_concat.view(
            n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)

        # e_ij = a^TLeakyReLU([g_i||g_j]): Here is the main difference
        # e.shape = [n_nodes, n_nodes, n_heads, 1]
        e = self.attention(self.activation(g_concat))
        e = e.squeeze(-1)  # remove last dimension

        # check adjacency matrix: [n_nodes, n_nodes, n_heads] or [n_nodes, n_nodes, 1]
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        # Mask e_ij based on adjacency matrix
        # if there is no edges btw i and j, e_ij = -inf
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # Normalize attention scores
        # \alpha_ij = softmax_j(e_ij)
        a = self.softmax(e)
        a = self.dropout(a)  # apply dropout regularization

        # Calculate final output for each head
        # ommited unlinearity \theta to appy it on GAT model later
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        # Concatenate the head
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)

        # Or mean
        else:
            return attn_res.mean(dim=1)
