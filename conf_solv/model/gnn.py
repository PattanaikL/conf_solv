import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool
from .layers import DMPNNConv


class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=200, depth=4, n_layers=2, gnn_type='dmpnn', dropout=0.0):
        super(GNN, self).__init__()

        self.depth = depth
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gnn_type = gnn_type
        self.dropout = dropout

        if self.gnn_type == 'dmpnn':
            self.edge_init = nn.Linear(node_dim + edge_dim, hidden_dim, bias=False)
            self.edge_to_node = DMPNNConv(hidden_dim)
        else:
            self.node_init = nn.Linear(node_dim, hidden_dim)
            self.edge_init = nn.Linear(node_dim, hidden_dim)

        # layers
        if self.gnn_type == 'dmpnn':
            self.conv = DMPNNConv(hidden_dim)
        else:
            ValueError('Undefined GNN type called {}'.format(self.gnn_type))

        # pool
        self.pool = global_mean_pool

        # ffn
        self.ffn = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, edge_index, edge_attr, batch):

        if self.gnn_type == 'dmpnn':
            row, col = edge_index
            edge_attr = torch.cat([x[row], edge_attr], dim=-1)
            edge_attr = F.leaky_relu(self.edge_init(edge_attr))  # same as "message" in chemprop
        else:
            x = F.leaky_relu(self.node_init(x))
            edge_attr = F.leaky_relu(self.edge_init(edge_attr))

        x_list = [x]
        edge_attr_list = [edge_attr]

        # convolutions
        for l in range(self.depth):

            x_h, edge_attr_h = self.conv(x_list[-1], edge_index, edge_attr_list[-1])
            h = edge_attr_h if self.gnn_type == 'dmpnn' else x_h

            if l == self.depth - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.leaky_relu(h), self.dropout, training=self.training)

            if self.gnn_type == 'dmpnn':
                h += edge_attr_h
                edge_attr_list.append(h)
            else:
                h += x_h
                x_list.append(h)

        # dmpnn edge -> node aggregation
        if self.gnn_type == 'dmpnn':
            h, _ = self.edge_to_node(x_list[-1], edge_index, h)

        return self.pool(self.ffn(h).squeeze(-1), batch)


class MLP(nn.Module):
    """
    Creates a NN using nn.ModuleList to automatically adjust the number of layers.
    For each hidden layer, the number of inputs and outputs is constant.

    Inputs:
        in_dim (int):               number of features contained in the input layer.
        out_dim (int):              number of features input and output from each hidden layer,
                                    including the output layer.
        num_layers (int):           number of layers in the network
        activation (torch function): activation function to be used during the hidden layers
    """

    def __init__(self, in_dim, h_dim, out_dim, num_layers, activation=torch.nn.LeakyReLU(),
                layer_norm=False, batch_norm=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # create the input layer
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(nn.Linear(in_dim, h_dim))
            else:
                self.layers.append(nn.Linear(h_dim, h_dim))
            if layer_norm: self.layers.append(nn.LayerNorm(h_dim))
            if batch_norm: self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(activation)
        self.layers.append(nn.Linear(h_dim, out_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
