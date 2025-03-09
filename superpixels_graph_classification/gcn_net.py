import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
from layers.projection_head import projection_head

# class GCNNet(nn.Module):
#     def __init__(self, net_params):
#         super().__init__()
#         in_dim = net_params['in_dim']
#         hidden_dim = net_params['hidden_dim']
#         out_dim = net_params['out_dim']
#         n_classes = net_params['n_classes']
#         in_feat_dropout = net_params['in_feat_dropout']
#         dropout = net_params['dropout']
#         n_layers = net_params['L']
#         self.readout = net_params['readout']
#         self.graph_norm = net_params['graph_norm']
#         self.batch_norm = net_params['batch_norm']
#         self.residual = net_params['residual']
#
#         # embedding - N*GCNlayeer - Readout - head/mlp
#         self.embedding_h = nn.Linear(in_dim, hidden_dim)
#         self.in_feat_dropout = nn.Dropout(in_feat_dropout)
#
#         self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
#                                               self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
#         self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.graph_norm, self.batch_norm, self.residual))
#         self.MLP_layer = MLPReadout(out_dim, n_classes)
#         self.projection_head = projection_head(out_dim, out_dim)
#
#     def forward(self, g, h, e, snorm_n, snorm_e, mlp=True, head=False, return_graph=False):
#
#         h = self.embedding_h(h)
#         h = self.in_feat_dropout(h)
#         for conv in self.layers:
#             h = conv(g, h, snorm_n)
#         g.ndata['h'] = h
#
#         if return_graph:
#             return g
#
#         if self.readout == "sum":
#             hg = dgl.sum_nodes(g, 'h')
#         elif self.readout == "max":
#             hg = dgl.max_nodes(g, 'h')
#         elif self.readout == "mean":
#             hg = dgl.mean_nodes(g, 'h')
#         else:
#             hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
#
#         if mlp:
#             return self.MLP_layer(hg)
#         else:
#             if head:
#                 return self.projection_head(hg)
#             else:
#                 return hg
#
#
#     def loss(self, pred, label):
#         criterion = nn.CrossEntropyLoss()
#         loss = criterion(pred, label)
#         return loss


class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_classes,
                 in_feat_dropout, dropout, n_layers, readout,
                 graph_norm, batch_norm, residual):
        super().__init__()
        self.readout = readout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual

        # 输入层及 dropout
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # GCN 层，每层均使用相同超参数（除了输出维度）
        self.layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                     graph_norm, batch_norm, residual)
            for _ in range(n_layers - 1)
        ])
        # 最后一层输出 out_dim
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout,
                                     graph_norm, batch_norm, residual))
        # Readout 之后的 MLP 层
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        # 投影头，用于对比学习中得到图的表征
        self.projection_head = projection_head(out_dim, out_dim)

    def forward(self, g, h, e, snorm_n, snorm_e, mlp=True, head=False, return_graph=False):
        # h: 节点特征，e: 边特征（目前未在本函数内使用）
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h, snorm_n)
        g.ndata['h'] = h

        if return_graph:
            return g

        # 读取节点表征（这里采用不同的聚合方法）
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # 默认使用 mean

        if mlp:
            return self.MLP_layer(hg)
        else:
            if head:
                return self.projection_head(hg)
            else:
                return hg

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        return criterion(pred, label)