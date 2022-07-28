from rdkit import Chem
import numpy as np
import pandas as pd
from ase import Atoms
import random

import torch
import pytorch_lightning as pl
import torch_geometric as tg
from torch_geometric.data import Data, Dataset
from .collate import DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .features import MolGraph, SOLVENTS, SOLVENTS_REVERSE, IONIC_SOLVENTS, IONIC_SOLVENT_SMILES
from .scalers import TorchMinMaxScaler, TorchStandardScaler


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
        self.param_name = config["param_name"]
        self.scaler = scaler
        self.max_confs = config["max_confs"]
        self.mode = mode
        self.solvent = config["solvent"]
        self.solvents = list(SOLVENTS.values())[1:]

        if config["no_ionic_solvents"]:
            self.solvents = [s for s in self.solvents if s not in IONIC_SOLVENT_SMILES]
            energies_df = energies_df[~energies_df['solvent'].isin(IONIC_SOLVENTS)]

        energies_df = energies_df[energies_df['mol_id'].isin(self.split)]
        energies_df = energies_df[energies_df[self.param_name] < config["threshold"]]
        energies_df = energies_df.dropna(subset=[self.param_name])

        coords_df = coords_df[coords_df['mol_id'].isin(self.split)]
        self.mol_ids = list(set(list(coords_df['mol_id'].unique())).intersection(set(list(energies_df['mol_id'].unique()))))

        self.coords = coords_df.set_index(['mol_id', 'conf_id'])
        self.energies = energies_df.set_index(['mol_id', 'conf_id', 'solvent'])

        if mode == 'train':  # figure out scaling
            # gsolvs = [g for conf_gs in self.conf_energies for g in conf_gs]
            self.mean = 0  # np.mean(gsolvs)
            self.std = 1  # np.std(gsolvs)

    def process_key(self, key):

        if self.solvent:
            solvent_smi = self.solvent
        else:
            solvent_smi = random.choice(self.solvents)

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
            max_confs = self.max_confs
        elif self.mode == 'val':  # pick first confs (should be low energy ones)
            conf_ids = sample_energy_confs[np.in1d(sample_energy_confs, available_confs)][:min(self.max_confs, len(available_confs))]
            max_confs = self.max_confs
        elif self.mode =='test':
            conf_ids = sample_energy_confs[np.in1d(sample_energy_confs, available_confs)][:min(200, len(available_confs))]
            max_confs = len(conf_ids)
        mols = sample_coords.loc[conf_ids]['mol'].values
        dG = torch.tensor(sample_energies.loc[(mol_id, conf_ids, slice(None))][self.param_name].values, dtype=torch.float)

        solvent_molgraph = MolGraph(solvent_smi)
        pair_data = create_pairdata(solvent_molgraph, mols, max_confs)

        pair_data.y = torch.zeros([max_confs])
        pair_data.y[:len(dG)] = dG
        pair_data.mol_id = mol_id
        pair_data.solvent_name = SOLVENTS_REVERSE[solvent_smi]
        pair_data.conf_ids = conf_ids

        return pair_data

    def __len__(self):
        return len(self.mol_ids)

    def __getitem__(self, key):
        data = None
        while data is None:
            try:
                data = self.process_key(key)
            except:
                continue
            if data is None:
                key = random.choice(range(len(self.mol_ids)))
        return data


class SolventData3DModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.param_name = config["param_name"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

        self.coords_df = pd.read_pickle(config["coords_path"])
        self.energies_df = pd.read_pickle(config["energies_path"])
        self.split = np.load(config["split_path"], allow_pickle=True)
        self.node_dim, self.edge_dim = self.get_dims()

        dG_train = self.energies_df[self.param_name][self.energies_df["mol_id"].isin(self.split[0])].values
        # dG is relative; allow for negatives
        dG_train = np.concatenate([dG_train, -dG_train]) if config["relative_loss"] else dG_train

        if config["scaler_type"] == "standard":
            self.scaler = TorchStandardScaler()
            self.scaler.fit(torch.from_numpy(dG_train.ravel()))

        elif config["scaler_type"] == "min_max":
            self.scaler = TorchMinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(torch.from_numpy(dG_train.reshape(-1, 1)))
        else:
            self.scaler = TorchStandardScaler()
            self.scaler.mean = torch.tensor(0.)
            self.scaler.std = torch.tensor(1.)

    def train_dataloader(self):
        train_dataset = SolventData3D(self.config, self.coords_df, self.energies_df, self.split, self.scaler, mode="train")
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            follow_batch=["x_solvent", "x_solute"],
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataset = SolventData3D(self.config, self.coords_df, self.energies_df, self.split, self.scaler, mode="val")
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            follow_batch=["x_solvent", "x_solute"],
            pin_memory=True,
        )

    def test_dataloader(self):
        test_dataset = SolventData3D(self.config, self.coords_df, self.energies_df, self.split, self.scaler, mode="test")
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            follow_batch=["x_solvent", "x_solute"],
            pin_memory=True,
        )

    def predict_dataloader(self):
        test_dataset = SolventData3D(self.config, self.coords_df, self.energies_df, self.split, self.scaler, mode="test")
        return DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            follow_batch=["x_solvent", "x_solute"],
            pin_memory=True,
        )

    @staticmethod
    def get_dims():
        test_graph = MolGraph('CCC')
        node_dim = len(test_graph.f_atoms[0])
        edge_dim = len(test_graph.f_bonds[0])
        return node_dim, edge_dim
