from rdkit import Chem
import numpy as np
import pandas as pd
from ase import Atoms
import random

import torch
import pytorch_lightning as pl
import torch_geometric as tg
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .features import MolGraph, SOLVENTS, SOLVENTS_REVERSE


class PairData(Data):
    def __init__(self, x_solvent=None, edge_index_solvent=None, edge_attr_solvent=None, mol_attr_solvent=None,
                 x_solute=None, z_solute=None, pos_solute=None, solute_mask=None):

        super(PairData, self).__init__()

        self.x_solvent = x_solvent
        self.edge_index_solvent = edge_index_solvent
        self.edge_attr_solvent = edge_attr_solvent
        self.mol_attr_solvent = mol_attr_solvent

        self.x_solute = x_solute
        self.z_solute = z_solute

        self.pos_solute = pos_solute
        self.solute_mask = solute_mask

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_solvent':
            return self.x_solvent.size(0)
        if key == 'edge_index_solute':
            return self.x_solute.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def create_pairdata(solvent_molgraph, mols, max_confs=10):

    x_solvent = torch.tensor(solvent_molgraph.f_atoms, dtype=torch.float)
    edge_index_solvent = torch.tensor(solvent_molgraph.edge_index, dtype=torch.long).t().contiguous()
    edge_attr_solvent = torch.tensor(solvent_molgraph.f_bonds, dtype=torch.float)
    mol_attr_solvent = torch.tensor(solvent_molgraph.f_mol, dtype=torch.float).unsqueeze(0)
    n_atoms = mols[0].get_global_number_of_atoms()

    x_solute = torch.tensor(mols[0].get_atomic_numbers(), dtype=torch.int64)  # for bookeeping
    z_solute = torch.tensor(mols[0].get_atomic_numbers().tolist() * max_confs, dtype=torch.int64)

    pos_solute = torch.rand([z_solute.size(0), 3])
    pos_solute[0:n_atoms * len(mols)] = \
        torch.tensor(np.vstack([c.positions for c in mols]), dtype=torch.float32)
    solute_mask = torch.BoolTensor(np.arange(max_confs) < len(mols))
    return PairData(x_solvent=x_solvent, edge_index_solvent=edge_index_solvent, edge_attr_solvent=edge_attr_solvent,
                    mol_attr_solvent=mol_attr_solvent, x_solute=x_solute, z_solute=z_solute, pos_solute=pos_solute,
                    solute_mask=solute_mask)


class SolventData3D(Dataset):
    def __init__(self, config, coords_df, energies_df, split, scaler, mode):
        super(Dataset, self).__init__()

        self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        self.split = split[self.split_idx]

        if mode == 'train' and config["n_training_points"]:
            self.split = self.split[:config["n_training_points"]]

        self.config = config
        self.scaler = scaler
        self.max_confs = config["max_confs"]
        self.mode = mode
        self.solvent = config["solvent"]

        energies = energies_df[energies_df['mol_id'].isin(self.split)]
        energies = energies[energies['dG'] < config["threshold"]]
        self.energies = energies.dropna(subset=['dG'])

        coords = coords_df[coords_df['mol_id'].isin(self.split)]
        self.mol_ids = list(set(list(coords['mol_id'].unique())).intersection(set(list(self.energies['mol_id'].unique()))))

        self.coords = coords.set_index(['mol_id', 'conf_id'])
        self.energies = energies.set_index(['mol_id', 'conf_id', 'solvent'])

        if mode == 'train':  # figure out scaling
            # gsolvs = [g for conf_gs in self.conf_energies for g in conf_gs]
            self.mean = 0  # np.mean(gsolvs)
            self.std = 1  # np.std(gsolvs)

    def process_key(self, key):

        if self.solvent:
            solvent_smi = self.solvent
        else:
            solvent_smi = random.choice(list(SOLVENTS.values())[1:])

        mol_id = self.mol_ids[key]
        sample_coords = self.coords.loc[(mol_id, slice(None))]
        sample_energies = self.energies.loc[(mol_id, slice(None), SOLVENTS_REVERSE[solvent_smi])]

        sample_coord_confs = sample_coords.index.get_level_values("conf_id").values
        sample_energy_confs = sample_energies.index.get_level_values("conf_id").values

        available_confs = np.intersect1d(sample_coord_confs, sample_energy_confs)
        if len(available_confs) == 0:
            return None

        if self.mode == 'train':  # random sampling during training
            conf_ids = random.sample(list(available_confs), min(self.max_confs, len(available_confs)))
        else:  # pick first confs (should be low energy ones)
            conf_ids = sample_energy_confs[np.in1d(sample_energy_confs, available_confs)][:min(self.max_confs, len(available_confs))]

        mols = sample_coords.loc[conf_ids]['mol'].values
        dG = sample_energies.loc[(mol_id, conf_ids, slice(None))]["dG"].values

        solvent_molgraph = MolGraph(solvent_smi)
        pair_data = create_pairdata(solvent_molgraph, mols, self.max_confs)

        pair_data.y = torch.zeros([self.max_confs])
        scaled_y = self.scaler.transform(self.scaler.transform(dG.reshape(-1, 1)))
        pair_data.y[:len(scaled_y)] = torch.tensor(scaled_y.reshape(-1), dtype=torch.float)
        pair_data.mol_id = mol_id

        return pair_data

    def __len__(self):
        return len(self.mol_ids)

    def __getitem__(self, key):
        return self.process_key(key)


class SolventData3DModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

        self.coords_df = pd.read_pickle(config["coords_path"])
        self.energies_df = pd.read_pickle(config["energies_path"])
        self.split = np.load(config["split_path"], allow_pickle=True)
        self.node_dim, self.edge_dim = self.get_dims()

        dG_train = self.energies_df["dG"][self.energies_df.index.isin(self.split[0])].values
        if config["scaler_type"] == "standard":
            self.scaler = StandardScaler().fit(dG_train.reshape(-1, 1))
        elif config["scaler_type"] == "min_max":
            self.scaler = MinMaxScaler().fit(dG_train.reshape(-1, 1))
        else:
            self.scaler = StandardScaler().fit(np.array([[0]]))

    def train_dataloader(self):
        train_dataset = SolventData3D(self.config, self.coords_df, self.energies_df, self.split, self.scaler, mode="train")
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            follow_batch=["x_solvent", "x_solute"],
        )

    def val_dataloader(self):
        val_dataset = SolventData3D(self.config, self.coords_df, self.energies_df, self.split, self.scaler, mode="val")
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            follow_batch=["x_solvent", "x_solute"],
        )

    def test_dataloader(self):
        test_dataset = SolventData3D(self.config, self.coords_df, self.energies_df, self.split, self.scaler, mode="test")
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            follow_batch=["x_solvent", "x_solute"],
        )

    @staticmethod
    def get_dims():
        test_graph = MolGraph('CCC')
        node_dim = len(test_graph.f_atoms[0])
        edge_dim = len(test_graph.f_bonds[0])
        return node_dim, edge_dim
