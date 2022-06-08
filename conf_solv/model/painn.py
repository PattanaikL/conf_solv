#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
PaiNN module in PyG
Adapted from https://github.com/MaxH1996/PaiNN-in-PyG
Modified by Lagnajit Pattanaik
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Embedding

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import radius_graph
from torch_scatter import scatter


class BesselBasis(torch.nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, num_radial=None):
        """
        Args:
            cutoff: radial cutoff
            num_radial: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, num_radial + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        inputs = torch.norm(inputs, p=2, dim=1)
        a = self.freqs
        ax = torch.outer(inputs, a)
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[:, None]

        return y


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        # self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.cutoff = cutoff

    def forward(self, distances):
        """Compute cutoff.
        Args:
            distances (torch.Tensor): values of interatomic distances.
        Returns:
            torch.Tensor: values of cutoff function.
        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class UpdatePaiNN(torch.nn.Module):
    def __init__(self, num_feat):
        super(UpdatePaiNN, self).__init__()

        self.lin_up = Linear(2 * num_feat, num_feat)
        self.denseU = Linear(num_feat, num_feat, bias=False)
        self.denseV = Linear(num_feat, num_feat, bias=False)
        self.lin2 = Linear(num_feat, 3 * num_feat)
        self.silu = F.silu
        self.num_feat = num_feat

    def forward(self, s, v):

        v = v.flatten(-2, -1)

        flat_shape_v = v.shape[-1]

        v_u = v.reshape(-1, int(flat_shape_v / 3), 3)
        v_ut = torch.transpose(v_u, 1, 2)  # need transpose to get lin.comb a long feature dimension
        U_v = torch.transpose(self.denseU(v_ut), 1, 2)
        V_v = torch.transpose(self.denseV(v_ut), 1, 2)

        # form the dot product
        Uv_Vv_dot = torch.einsum("ijk,ijk->ij", U_v, V_v)

        # s_j channel
        V_v_norm = torch.norm(V_v, dim=-1)
        a = torch.cat([s, V_v_norm], dim=-1)  # stack
        a = self.lin_up(a)
        a = F.silu(a)
        a = self.lin2(a)

        # final split
        a_vv, a_sv, a_ss = torch.split(a, self.num_feat, dim=-1)

        # outputs
        dvu = torch.einsum("ijk,ij->ijk", U_v, a_vv)
        dsu = a_ss + a_sv * Uv_Vv_dot

        return dsu, dvu.reshape(-1, int(flat_shape_v / 3), 3)


class MessagePassPaiNN(MessagePassing):
    def __init__(self, num_feat, cutoff=5.0, num_radial=20):
        super(MessagePassPaiNN, self).__init__(aggr="add")

        self.lin1 = Linear(num_feat, num_feat)
        self.lin2 = Linear(num_feat, 3 * num_feat)
        self.lin_rbf = Linear(num_radial, 3 * num_feat)
        self.silu = F.silu

        self.RBF = BesselBasis(cutoff=cutoff, num_radial=num_radial)
        self.f_cut = CosineCutoff(cutoff)
        self.num_feat = num_feat

    def forward(self, s, v, edge_index, edge_attr):

        v = v.flatten(-2, -1)

        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]

        x = torch.cat([s, v], dim=-1)

        x = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            flat_shape_s=flat_shape_s,
            flat_shape_v=flat_shape_v,
        )

        return x

    def message(self, x_j, edge_attr, flat_shape_s, flat_shape_v):

        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)

        # r_ij channel
        rbf = self.RBF(edge_attr)
        ch1 = self.lin_rbf(rbf)
        cut = self.f_cut(edge_attr.norm(dim=-1))
        W = torch.einsum("ij,i->ij", ch1, cut)  # ch1 * f_cut

        # s_j channel
        phi = self.lin1(s_j)
        phi = self.silu(phi)
        phi = self.lin2(phi)

        # Split

        left, dsm, right = torch.split(phi * W, self.num_feat, dim=-1)

        # v_j channel
        normalized = F.normalize(edge_attr, p=2, dim=1)
        v_j = v_j.reshape(-1, int(flat_shape_v / 3), 3)
        hadamard_right = torch.einsum("ij,ik->ijk", right, normalized)
        hadamard_left = torch.einsum("ijk,ij->ijk", v_j, left)
        dvm = hadamard_left + hadamard_right

        # Prepare vector for aggregate -> update
        x_j = torch.cat((dsm, dvm.flatten(-2)), dim=-1)

        return x_j

    def update(self, out_aggr, flat_shape_s, flat_shape_v):

        s_j, v_j = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)

        return s_j, v_j.reshape(-1, int(flat_shape_v / 3), 3)


class PaiNN(torch.nn.Module):
    def __init__(
        self,
        num_feat,
        cutoff=5.0,
        num_radial=20,
        num_interactions=3,
    ):
        super(PaiNN, self).__init__()
        """PyG implementation of PaiNN network of Schütt et. al. Supports two arrays
           stored at the nodes of shape (num_nodes,num_feat,1) and (num_nodes, num_feat,3). For this
           representation to be compatible with PyG, the arrays are flattened and concatenated"""

        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.num_radial = num_radial
        self.num_feat = num_feat
        self.lin = Linear(num_feat, num_feat)
        self.silu = F.silu

        self.embedding = Embedding(95, num_feat)

        self.list_message = nn.ModuleList(
            [
                MessagePassPaiNN(num_feat, cutoff, num_radial)
                for _ in range(self.num_interactions)
            ]
        )
        self.list_update = nn.ModuleList(
            [
                UpdatePaiNN(num_feat)
                for _ in range(self.num_interactions)
            ]
        )

    def forward(self, z, pos, batch=None, mask=None): #Removed edge index, as it will be determined on the fly instead of being passed.

        s = self.embedding(z)
        v = torch.zeros_like(s, device=s.device).unsqueeze(-1).repeat(1, 1, 3)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch) #Compute edge index, as is done in dimenetpp.
        j, i = edge_index
        rij = pos[i] - pos[j]  #definition 

        for i in range(self.num_interactions):

            s_temp, v_temp = self.list_message[i](s, v, edge_index, rij)
            s, v = s_temp + s, v_temp + v
            s_temp, v_temp = self.list_update[i](s, v)
            s, v = s_temp + s, v_temp + v

        s = self.lin(s)
        s = self.silu(s)
        s = self.lin(s)

        #out = scatter(s, batch, dim=0, reduce="add") - Commenting this out. Lucky does the aggregation in model.py

        return s