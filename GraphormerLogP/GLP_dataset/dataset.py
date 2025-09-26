from typing import Optional
import pandas as pd
import pytorch_lightning as pl
import torch
from chython import smiles
from chython.exceptions import InvalidAromaticRing
from chytorch.utils.data import MoleculeDataset, chained_collate, collate_molecules
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)


class PandasData(pl.LightningDataModule):
    def __init__(
        self,
        csv: str,
        structure: str,
        property1: str,
        dataset_type: str,
        prepared_df_path: str,
        batch_size: int = 32,
    ):
        super().__init__()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.validation_x = None
        self.validation_y = None
        self.prepared_df_path = prepared_df_path
        self.csv = csv
        self.structure = structure
        self.property1 = property1
        self.dataset_type = dataset_type
        self.batch_size = batch_size

    @staticmethod
    def prepare_mol(mol_smi):
        try:
            mol = smiles(mol_smi)
            try:
                mol.kekule()
            except InvalidAromaticRing:
                mol = None
        except Exception:
            mol = None
        return mol

    def prepare_data(self):
        df = pd.read_csv(self.csv)
        df = df[[self.structure, self.property1, self.dataset_type]]
        df[self.structure] = df[self.structure].apply(self.prepare_mol)
        df.dropna(inplace=True, axis=1, how="all")
        df.to_pickle(self.prepared_df_path)

    def setup(self, stage: Optional[str] = None):
        df = pd.read_pickle(self.prepared_df_path)
        if stage == "fit" or stage is None:
            df_train = df[df.dataset_type == "train"]
            mols = df_train[self.structure].to_list()
            self.train_x = MoleculeDataset(mols)
            self.train_y = torch.Tensor(df_train[[self.property1]].to_numpy())

        if stage == "validation" or stage is None:
            df_validation = df[df.dataset_type == "validation"]
            mols = df_validation[self.structure].to_list()
            self.validation_x = MoleculeDataset(mols)
            self.validation_y = torch.Tensor(df_validation[[self.property1]].to_numpy())

        if stage == "test" or stage is None:
            df_test = df[df.dataset_type == "test"]
            mols = df_test[self.structure].to_list()
            self.test_x = MoleculeDataset(mols)
            self.test_y = torch.Tensor(df_test[[self.property1]].to_numpy())

    def train_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.train_x, self.train_y),
            collate_fn=chained_collate(collate_molecules, torch.stack),
            batch_size=self.batch_size,
            shuffle=True, num_workers=7
        )

    def validation_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.validation_x, self.validation_y),
            collate_fn=chained_collate(collate_molecules, torch.stack),
            batch_size=self.batch_size,
            shuffle=False, num_workers=7
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.test_x, self.test_y),
            collate_fn=chained_collate(collate_molecules, torch.stack),
            batch_size=self.batch_size,
            shuffle=False, num_workers=7
        )