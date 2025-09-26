import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataset import PandasData
from model_p import Model, early_stop_callback, FeatureExtractorFreezeUnfreeze

torch.manual_seed(42)

@click.command()
@click.option(
    "-d", "--path_to_csv", type=click.Path(), help="Path to csv file with data."
)
@click.option(
    "-i",
    "--path_to_interm_dataset",
    type=click.Path(),
    help="Path to pickle with intermediate data.",
)
def train(path_to_csv, path_to_interm_dataset):
    print('start')
    dataset = PandasData(
        csv=path_to_csv,
        structure="molecule",
        property1="logp",
        dataset_type="dataset_type",
        prepared_df_path=path_to_interm_dataset,
        batch_size=128,
    )
    print('load')
    dataset.prepare_data()
    print('prepare')
    dataset.setup()
    print('setup')
    modeler = Model(
        loss_function=nn.HuberLoss(),
        learning_rate=4e-4,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator='gpu', devices=1, min_epochs=50, max_epochs=1500,
                         callbacks=[FeatureExtractorFreezeUnfreeze(),
                                    early_stop_callback, lr_monitor],
                         check_val_every_n_epoch=5, log_every_n_steps=38)
    trainer.fit(modeler, dataset.train_dataloader(), dataset.validation_dataloader())
    trainer.validate(modeler, dataset.validation_dataloader())
    trainer.test(modeler, dataset.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    train()
