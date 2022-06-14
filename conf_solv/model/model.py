import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_scatter import scatter

from .gnn import MLP, GNN
from .dimenet import DimeNet
from .schnet import SchNet
from .dimenet_pp import DimeNetPlusPlus
from .spherenet import SphereNet
from .painn import PaiNN


class ConfSolv(nn.Module):
    def __init__(self, config):
        super(ConfSolv, self).__init__()

        node_dim = config["node_dim"]
        edge_dim = config["edge_dim"]
        self.config = config
        self.relative_model = config["relative_model"]
        self.diff_type = config["diff_type"]
        self.solute_hidden_dim = config["solute_hidden_dim"]
        self.solute_model_type = config["solute_model"]

        self.solvent_model = GNN(
            node_dim,
            edge_dim,
            config["solvent_hidden_dim"],
            config["solvent_depth"],
            config["solvent_n_layers"],
            config["solvent_gnn_type"],
            config["solvent_dropout"],
        )

        if self.solute_model_type == 'DimeNet':
            self.solute_model = DimeNet(
                hidden_channels=config["hidden_channels"],
                out_channels=config["solute_hidden_dim"],
                num_blocks=config["num_blocks"],
                num_bilinear=config["dn_num_bilinear"],
                num_spherical=config["num_spherical"],
                num_radial=config["num_radial"],
                num_output_layers=2
            )

        elif self.solute_model_type == 'SchNet':
            self.solute_model = SchNet(
                hidden_channels=config["hidden_channels"],
                out_channels=config["solute_hidden_dim"],
                num_filters=config["sn_num_filters"],
                num_interactions=config["num_blocks"],
                num_gaussians=config["sn_num_gaussians"],
                cutoff=config["sn_cutoff"],
                readout=config["sn_readout"]
            )

        elif self.solute_model_type == 'DimeNetPP':
            self.solute_model = DimeNetPlusPlus(
                hidden_channels=config["hidden_channels"],
                out_channels=config["solute_hidden_dim"],
                num_blocks=config["num_blocks"],
                int_emb_size=config["int_emb_size"],
                basis_emb_size=config["basis_emb_size_dist"],
                out_emb_channels=config["solute_hidden_dim"],
                num_spherical=config["num_spherical"],
                num_radial=config["num_radial"],
                num_output_layers=2
            )

        elif self.solute_model_type == 'SphereNet':
            self.solute_model = SphereNet(
                energy_and_force=False,
                cutoff=config["cutoff"],
                num_layers=config["num_layers"],
                hidden_channels=config["hidden_channels"],
                out_channels=config["solute_hidden_dim"],
                int_emb_size=config["int_emb_size"],
                basis_emb_size_dist=config["basis_emb_size_dist"],
                basis_emb_size_angle=config["basis_emb_size_angle"],
                basis_emb_size_torsion=config["basis_emb_size_torsion"],
                out_emb_channels=config["out_emb_channels"],
                num_spherical=config["num_spherical"],
                num_radial=config["num_radial"],
            )
        
        elif self.solute_model_type == 'PaiNN':
            self.solute_model = PaiNN(
                num_feat=config["solute_hidden_dim"],
                cutoff=config["cutoff"],
                num_radial=config["num_radial"],
                num_interactions=config["num_blocks"],
            )

        self.ffn = MLP(
            in_dim=config["solvent_hidden_dim"] + config["solute_hidden_dim"],
            h_dim=config["ffn_hidden_dim"],
            out_dim=1,
            num_layers=config["ffn_n_layers"]
        )

        self.ffn1 = MLP(
            in_dim=config["solvent_hidden_dim"],
            h_dim=config["solvent_hidden_dim"],
            out_dim=config["solvent_hidden_dim"],
            num_layers=2
        )

    def forward(self, data, max_confs=10):
        # torch.autograd.set_detect_anomaly(True)
        x_solvent, edge_index_solvent, edge_attr_solvent, mol_attr_solvent, solvent_batch, solute_confs_batch = \
            data.x_solvent, data.edge_index_solvent, data.edge_attr_solvent, data.mol_attr_solvent, data.x_solvent_batch, data.solute_confs_batch

        h1 = self.solvent_model(x_solvent, edge_index_solvent, edge_attr_solvent, solvent_batch)

        # conformers are batched along first dimension (this is why we use repeat_interleave)
        h2_ = self.solute_model(data.z_solute, data.pos_solute, solute_confs_batch)

        if self.relative_model and self.diff_type == 'atom_diff':
            # don't sum over atoms in solute model
            solute_confs_mask = data.solute_mask.view(-1, max_confs).unsqueeze(-1) * \
                                data.solute_mask.view(-1, max_confs).unsqueeze(-2)
            solute_confs_mask = solute_confs_mask.unsqueeze(-1).unsqueeze(-1)

            h2_atoms, h2_atoms_mask = tg.utils.to_dense_batch(h2_, solute_confs_batch)
            h2_atoms_reshaped = h2_atoms.view(solvent_batch.max() + 1, max_confs, -1, self.solute_hidden_dim)
            h2_atoms_mask = h2_atoms_mask.view(solvent_batch.max() + 1, max_confs, 1, -1, 1)
            h2_atoms_diff = (h2_atoms_reshaped.unsqueeze(-3) - h2_atoms_reshaped.unsqueeze(
                -4)) * h2_atoms_mask * solute_confs_mask  # added
            h2 = self.ffn1(h2_atoms_diff.sum(dim=-2))
            h1 = h1.unsqueeze(-2).unsqueeze(-2).repeat(1, max_confs, max_confs, 1)

        else:
            if self.solute_model_type != 'SphereNet':
                h2_ = scatter(h2_, solute_confs_batch, dim=0, reduce="add")
            h2 = torch.where(data.solute_mask.unsqueeze(-1) == False, torch.tensor([0.], device=h1.device), h2_)
            h1 = torch.repeat_interleave(h1, max_confs, dim=0)

        h = torch.cat([h1, h2], dim=-1)

        if self.relative_model:
            mask_ = data.solute_mask.view(solvent_batch.max() + 1, max_confs)
            mask = mask_.unsqueeze(-2) * mask_.unsqueeze(-1)
        else:
            mask = data.solute_mask

        if self.relative_model and self.diff_type == 'mol_diff':
            h_reshaped = h.view(solvent_batch.max() + 1, max_confs, -1)
            f1 = h_reshaped.unsqueeze(-2) - h_reshaped.unsqueeze(-3)
            f2 = h_reshaped.unsqueeze(-3) - h_reshaped.unsqueeze(-2)
            g = torch.squeeze(self.ffn(f1) - self.ffn(f2), -1)
            return g * mask

        else:
            return self.ffn(h).squeeze(-1) * mask
