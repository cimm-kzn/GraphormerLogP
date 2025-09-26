import click
import matplotlib.pyplot as plt
import os
import pandas as pd
from loguru import logger
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Draw import MolsToGridImage
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from chython import smiles
import numpy as np

from checkers import Molecule


class RawDatasetsParser:
    def __init__(self, data_path, data_type):
        self.smiles_list = []
        self.logp_list = []
        self.units_list = []
        self.source_list = []
        self.final_data = {}
        self.data_path = data_path
        self.sdf_batch_one = ['OpenChem.sdf', 'DB4.sdf', 'DB3.sdf', 'DB2.sdf', 'DB1.sdf', 'SAMPL6.sdf', 'SAMPL7.sdf']
        self.sdf_batch_two = ['Huuskonen.sdf', 'Star.sdf', 'NonStar.sdf']
        self.xls_data = 'Dataset_and_Predictions.xlsx'
        self.csv_data = 'AstraZeneca_logP.csv'
        self.external_data = 'prev_lod_logp.csv'
        self.data_type = data_type

    def _load_sdf_data(self):
        sdf_paths = [i for i in os.listdir(self.data_path) if i.split('.')[1] == 'sdf']
        all_data = {}
        for sdf_path in sdf_paths:
            suppl = Chem.SDMolSupplier("{}/{}".format(self.data_path,sdf_path))
            all_data[sdf_path] = []
            for sdf in suppl:
                all_data[sdf_path].append(sdf)
        for sdf_path in sdf_paths:
            logger.info("Raw dataset {} contains {} molecules".format(sdf_path, len(all_data[sdf_path])))
        return all_data

    def get_sdf_data(self, all_data):
        for sdf_name in self.sdf_batch_one:
            for mol_num, mol in enumerate(all_data[sdf_name], 1):
                if mol is None:
                    print(f"Файл {sdf_name}, молекула #{mol_num} не распарсилась (None)")
                try:
                    self.smiles_list.append(mol.GetPropsAsDict()['smiles'])
                    self.logp_list.append(mol.GetPropsAsDict()['logP'])
                    self.units_list.append(None)
                    self.source_list.append(sdf_name)
                except UnicodeDecodeError as e:
                    logger.error(f"UnicodeDecodeError in file {sdf_name}, molecule #{mol_num}: {e}")
                    continue
        for sdf_name in ['Huuskonen.sdf', 'Star.sdf', 'NonStar.sdf']:
            for mol_num, mol in enumerate(all_data[sdf_name], 1):
                try:
                    self.smiles_list.append(mol.GetPropsAsDict()['SMILES'])
                    self.logp_list.append(mol.GetPropsAsDict()['logPow {measured}'])
                    self.units_list.append(mol.GetPropsAsDict()['UNIT {logPow}'])
                    self.source_list.append(sdf_name)
                except UnicodeDecodeError as e:
                    logger.error(f"UnicodeDecodeError in file {sdf_name}, molecule #{mol_num}: {e}")
                    continue

    def get_xlsx_data(self):
        xlsx_path = '{}/{}'.format(self.data_path, self.xls_data)
        raw_df = pd.read_excel(xlsx_path)
        counter=0
        for smi, logp in zip(raw_df['SMILES'].to_list(), raw_df['logP\nexperimental\n(corrected)']):
            counter+=1
            self.smiles_list.append(smi)
            self.logp_list.append(logp)
            self.units_list.append(None)
            self.source_list.append('Dataset_and_Predictions')
        logger.info("Raw dataset {} contains {} molecules".format(xlsx_path, counter))

    def get_csv_data(self):
        if self.data_type == "internal":
            csv_path = '{}/{}'.format(self.data_path, self.csv_data)
            df_az_raw = pd.read_csv(csv_path, sep=';')
            counter = 0
            for smi, logp in zip(df_az_raw[df_az_raw['Standard Type'] == 'LogD7.4']['Smiles'].to_list(),
                                 df_az_raw[df_az_raw['Standard Type'] == 'LogD7.4']['Standard Value']):
                counter += 1
                self.smiles_list.append(smi)
                self.logp_list.append(logp)
                self.units_list.append(None)
                self.source_list.append('az_dataset')

        elif self.data_type == "external":
            csv_path = '{}/{}'.format(self.data_path, self.external_data)
            df = pd.read_csv(csv_path, sep=',')
            df = df[df['dataset_type'] == 'test']
            counter = 0
            for smi, logp in zip(df['molecule'], df['logp']):
                counter += 1
                self.smiles_list.append(smi)
                self.logp_list.append(logp)
                self.units_list.append(None)
                self.source_list.append('external')

        logger.info("Raw dataset {} contains {} molecules".format(csv_path, counter))

    def aggregator(self):
        self.final_data['smiles'] = self.smiles_list
        self.final_data['logp'] = self.logp_list
        self.final_data['source'] = self.source_list
        self.final_data['units'] = self.units_list
        df = pd.DataFrame(self.final_data)
        logger.info("Aggregated dataset shape {}", df.drop_duplicates(['smiles', 'logp']).shape)
        return df

    def get_datasets(self):
        if self.data_type == 'internal':
            all_sdf_data = self._load_sdf_data()
            self.get_sdf_data(all_sdf_data)
            self.get_xlsx_data()
            self.get_csv_data()
            df = self.aggregator()
        elif self.data_type == 'external':
            self.get_csv_data()
            df = self.aggregator()
        return df


class DatasetCleaner:
    def __init__(self, df, save_dir='../intermediate_results'):
        self.df = df.copy()
        self.removed_dir = os.path.join(save_dir, "removed")
        self.intermediate_dir = os.path.join(save_dir, "intermediate")
        self.upper_mw = 850
        self.lowest_mw = 80

        os.makedirs(self.removed_dir, exist_ok=True)
        os.makedirs(self.intermediate_dir, exist_ok=True)

    def _save_removed(self, removed_df, stage, reason):
        if not removed_df.empty:
            removed_df = removed_df.copy()
            if reason is not None:
                removed_df['reason'] = reason
            elif 'reason' not in removed_df.columns and 'invalid_reason' in removed_df.columns:
                removed_df['reason'] = removed_df['invalid_reason']
            removed_df.to_csv(os.path.join(self.removed_dir, f"removed_{stage}.csv"), index=False)
    def _save_intermediate(self, stage):
        self.df.to_csv(os.path.join(self.intermediate_dir, f"intermediate_{stage}.csv"), index=False)

    def stage_2_prepare_smiles(self):
        mol_objs = []
        smi_stds = []
        is_valid_flags = []
        reasons = []

        for smi in self.df['smiles']:
            mol = Molecule(smi)
            smi_std = mol.prepare()
            mol_objs.append(mol)
            smi_stds.append(smi_std)
            is_valid_flags.append(pd.notna(smi_std))
            reasons.append(mol.invalid_reason if smi_std is np.nan else None)

        self.df['smi_std'] = smi_stds
        self.df['is_valid'] = is_valid_flags
        self.df['mol_obj'] = mol_objs
        self.df['invalid_reason'] = reasons

        radical_rows = []
        for idx, row in self.df.iterrows():
            mol = row['mol_obj']
            if not row['is_valid']:
                continue
            if mol.has_radicals:
                radical_rows.append(idx)

        self.df['has_radical'] = False
        self.df.loc[radical_rows, 'has_radical'] = True

        truly_invalid_df = self.df[~self.df['is_valid']].copy()
        truly_invalid_df['reason'] = truly_invalid_df['invalid_reason']
        self._save_removed(truly_invalid_df, "02_invalid", None)

        radicals_df = self.df.loc[radical_rows].copy()
        self._save_removed(radicals_df, "02_radicals", "Has radicals")

        to_remove = set(truly_invalid_df.index) | set(radical_rows)
        self.df = self.df.drop(index=to_remove)

        self.df.drop(columns=['is_valid', 'mol_obj', 'has_radical', 'invalid_reason'], inplace=True)
        self.df['molecule'] = self.df['smi_std'].apply(Chem.MolFromSmiles)
        self.df = self.df.dropna(subset=['molecule'])

        self._save_intermediate("02_prepare")
        logger.info(f"Stage 2: Molecules remaining after cleaning: {self.df.shape}")

    def stage_3_filter_molwt(self):
        self.df['molwt'] = self.df['molecule'].apply(Descriptors.MolWt)
        removed = self.df[(self.df['molwt'] < self.lowest_mw) | (self.df['molwt'] > self.upper_mw)]
        self._save_removed(removed, "04_molwt", "MW not in [80, 850]")
        self.df = self.df[(self.df['molwt'] <= self.upper_mw) & (self.df['molwt'] > self.lowest_mw)]
        logger.info("Stage 3: Filtering by molecular weight: {}".format(self.df.shape))
        self._save_intermediate("04_molwt")

    def stage_4_filter_logp(self):
        self.df['logp'] = pd.to_numeric(self.df['logp'], errors='coerce')
        removed = self.df[(self.df['logp'] < -10) | (self.df['logp'] > 10)]
        self._save_removed(removed, "05_logp", "logP not in [-10, 10]")
        self.df = self.df[(self.df['logp'] >= -10) & (self.df['logp'] <= 10)]
        logger.info("Stage 4: Filtering by logP: {}".format(self.df.shape))
        self._save_intermediate("05_logp")

    def stage_5_remove_duplicates(self):
        before = len(self.df)
        duplicated_mask = self.df.duplicated(subset=['smi_std', 'logp'], keep='first')
        removed = self.df[duplicated_mask].copy()
        self._save_removed(removed, "06_duplicates", "Duplicate (smi_std, logp)")
        self.df = self.df.drop_duplicates(subset=['smi_std', 'logp'], keep='first')
        removed_count = before - len(self.df)
        logger.info("Stage 5: Removing duplicates:  {}".format(self.df.shape))
        logger.info(f"Удалено {removed_count} дубликатов")
        self._save_intermediate("06_dedup")

    def stage_6_non_unique_logp(self):
        grouped = self.df.groupby('smi_std').agg(set)
        grouped['logp_len'] = grouped['logp'].apply(len)
        valid = grouped[grouped['logp_len'] == 1]

        kept = self.df[self.df['smi_std'].isin(valid.index)]
        removed = self.df[~self.df['smi_std'].isin(valid.index)]
        self._save_removed(removed, "06_non_unique_logp", "Multiple logP for same smiles")

        self.df = kept
        logger.info("Stage 6: Removing smi_std with multiple logP values: {}".format(self.df.shape))
        self._save_intermediate("06_non_unique_logp")
        
    def stage_7_all(self, output_path):
        df_smiles_to_remove = pd.read_csv('../clean_data/filter_for_remove_SGNN_test.csv')
        self.df['container'] = self.df['smiles'].apply(self.normalize_smiles)
        df_smiles_to_remove['container'] = df_smiles_to_remove['smiles'].apply(self.normalize_smiles)

        to_remove = set(df_smiles_to_remove['container'].dropna())
        filtered_df = self.df[~self.df['container'].isin(to_remove)]
        filtered_df.drop(columns=['container'], inplace=True)
        self.df = filtered_df
        self.df[['smi_std', 'smiles', 'logp', 'source', 'units']].to_csv(output_path, index=False)
        logger.info("Stage 7: Saving the final dataset: {}".format(self.df.shape))
        self._save_intermediate("07_all")

    def clean(self, output_path):
        self._save_intermediate("01_raw")
        self.stage_2_prepare_smiles()
        self.stage_3_filter_molwt()
        self.stage_4_filter_logp()
        self.stage_5_remove_duplicates()
        self.stage_6_non_unique_logp()
        self.stage_7_all(output_path)
    
    @staticmethod
    def normalize_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return None
    

@click.command()
@click.option(
    "-i",
    "--data_dir_path",
    type=click.Path(),
    help="Path to dir with raw datasets.",
)
@click.option(
    "-o",
    "--output_path",
    type=click.Path(),
    help="Path to file with cleaned molecules"
)
@click.option(
    "-t",
    "--data_type",
    default='internal',
    help="Data type (internal/external)"
)
def main(data_dir_path, output_path, data_type):
    logger.add("dataset_info_jb_{}.log".format(data_type),  level="INFO")
    logger.add("dataset_errors_jb_{}.log".format(data_type), level="ERROR")
    raw_dataset_parser = RawDatasetsParser(data_dir_path, data_type)
    df = raw_dataset_parser.get_datasets()
    dataset_cleaner = DatasetCleaner(df)
    dataset_cleaner.clean(output_path)


if __name__ == "__main__":
    main()
