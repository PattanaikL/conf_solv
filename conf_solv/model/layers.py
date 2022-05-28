import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class DMPNNConv(MessagePassing):
    def __init__(self, hidden_dim):
        super(DMPNNConv, self).__init__(aggr='add', node_dim=0)
        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False),
                                 # nn.LayerNorm(hidden_dim),
                                 nn.LeakyReLU())

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)
        rev_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, a_message.size(-1), -1), dims=[1]).view_as(edge_attr)
        return a_message, self.mlp(a_message[row] - rev_message)

    def message(self, x_j, edge_attr):
        return F.leaky_relu(self.lin(edge_attr))
