import click
from sklearn.model_selection import train_test_split
import os
import pandas as pd

seed = 2


def separate(input_name, output_name, train_ratio, validation_ratio, test_ratio):
    data_path = '../clean_data'
    output_path = '../separate_data'
    data_file = '{}/{}'.format(data_path, input_name)
    output_file = '{}/{}'.format(output_path, output_name)
    
    df = pd.read_csv(data_file)
    Yr = df['logp']
    all_smiles = df['smi_std']
    
    X_train, X_test, y_train, y_test = train_test_split(all_smiles, Yr, test_size=1 - train_ratio, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=seed)
    
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.rename(columns={'smi_std': 'molecule', 0: "logp"}, inplace=True)
    train_data.insert(2, "dataset_type", ["train" for _ in range(len(train_data))], True)
    
    val_data = pd.concat([X_val, y_val], axis=1)
    val_data.rename(columns={'smi_std': 'molecule', 0: "logp"}, inplace=True)
    val_data.insert(2, "dataset_type", ["validation" for _ in range(len(val_data))], True)
    
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.rename(columns={'smi_std': 'molecule', 0: "logp"}, inplace=True)
    test_data.insert(2, "dataset_type", ["test" for _ in range(len(test_data))], True)
    
    all_data = train_data
    all_data = all_data.merge(val_data, how="outer")
    all_data = all_data.merge(test_data, how="outer")
    all_data = all_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    all_data.to_csv(output_file, index=False)


@click.command()
@click.option(
    "-i",
    "--input_name",
    type=click.Path(),
    help="Name input file.csv without path"
)
@click.option(
    "-o",
    "--output_name",
    type=click.Path(),
    help="Name output file.csv without path"
)
@click.option(
    "-tr",
    "--train_ratio",
    default=0.7,
    help="Training sample share"
)
@click.option(
    "-v",
    "--validation_ratio",
    default=0.15,
    help="Validation sample share"
)
@click.option(
    "-te",
    "--test_ratio",
    default=0.15,
    help="Test sample share"
)

def main(input_name, output_name, train_ratio, validation_ratio, test_ratio):
    separate(input_name, output_name, train_ratio, validation_ratio, test_ratio)


if __name__ == "__main__":
    main()