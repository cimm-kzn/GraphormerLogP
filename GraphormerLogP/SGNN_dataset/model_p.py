from typing import Optional, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from chytorch.nn import MoleculeEncoder, Slicer
from chytorch.optim.lr_scheduler import WarmUpCosine
from chytorch.zoo.rxnmap import Model as RXNmap
from torch.optim import AdamW
from pytorch_lightning.callbacks import BaseFinetuning, EarlyStopping

torch.manual_seed(42)




class Model(pl.LightningModule):
    def __init__(
        self,
        loss_function,
        learning_rate: Union[float, int],
    ):
        super().__init__()
        self.network = None
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.encoder = MoleculeEncoder()
        self.encoder.load_state_dict(RXNmap().molecule_encoder.state_dict())
        self.mlp = nn.Sequential(
            Slicer(slice(None), 0),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(in_features=2048, out_features=256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=2)
        )
       # self.linear_d = nn.Linear(in_features=512, out_features=1)
        #self.network_d = nn.Sequential(self.encoder, self.mlp, self.linear_d)
        self.network_p = nn.Sequential(self.encoder, self.mlp)
        self.mse = torchmetrics.MeanSquaredError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.r2 = torchmetrics.R2Score()
        self.training_step_outputs = {"preds": [], "y": [], "uncert": []}
        self.validation_step_outputs = {"preds": [], "y": [], "uncert": []}
        self.test_step_outputs = {"preds": [], "y": [], "uncert": []}

    def predict_values(self, batch):
        X, y = batch
        predictions = self.network_p(X)
        preds = predictions[:, 0:1]
        uncert = predictions[:, 1:2]
        return preds, uncert, y
    
    def predict_step(self, batch):
        X, y = batch
        predictions = self.network_p(X)
        preds = predictions[:, 0:1]
        uncert = predictions[:, 1:2]
        return preds, uncert
    
    # def predict_step(self, batch):
    #     X, y = batch
    #     predictions = self.network_p(X)
    #     preds = predictions[:, 0:1]
    #     uncert = predictions[:, 1:2]
    #     return preds

    # def calc_loss(self, preds_p, y_p):
    #     loss_p = self.loss_function(preds_p, y_p.nan_to_num())
    #     loss = loss_p
    #     return loss

    def calc_loss(self, preds, uncertainty, y, loss_lambda=1):
        logvar = uncertainty

        var = torch.exp(logvar)
        se = (preds - y) ** 2
        l = (1 - loss_lambda) * se + loss_lambda * (se / var + logvar)

        return l.mean()

    def calc_stats(self, preds, y, phase):

        rmse = self.rmse(preds, y)
        mse = self.mse(preds, y)
        r2 = self.r2(preds, y)
        return {f'{phase}_rmse': rmse,
                f'{phase}_mse': mse, f'{phase}_r2': r2}

    def training_step(self, batch, batch_idx):
        #print(self.trainer.lr_scheduler_configs)
        preds, uncert, y = self.predict_values(batch)
        loss = self.calc_loss( preds, uncert, y)
        results = {'train_loss': loss}
        self.training_step_outputs["preds"].append(preds)
        self.training_step_outputs["y"].append(y)
        # results.update(self.calc_stats(preds_p, y_p, preds_d, y_d, phase="train"))
        self.log_dict(results,
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        preds = torch.cat(self.training_step_outputs["preds"],dim=0)
        y = torch.cat(self.training_step_outputs["y"], dim=0)
        results = self.calc_stats(preds, y, phase="train")
        self.log_dict(results,
                      on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.training_step_outputs = {"preds":[], "y":[]}

    def test_step(self, batch, batch_idx):
        preds, uncert, y = self.predict_values(batch)
        loss = self.calc_loss(preds, uncert, y)
        #results = {'train_loss': loss}
        self.test_step_outputs["preds"].append(preds)
        self.test_step_outputs["y"].append(y)
        results = self.calc_stats(preds, y, phase="test")
        #results.update(self.calc_stats(preds_p, y_p, preds_d, y_d, phase="test"))
        self.log_dict(results,
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_step_outputs["preds"], dim=0)
        y = torch.cat(self.test_step_outputs["y"], dim=0)
        results = self.calc_stats(preds, y, phase="test")
        self.log_dict(results,
                      on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.test_step_outputs = {"preds": [], "y": []}

    def validation_step(self, batch, batch_idx):
        preds, uncert, y = self.predict_values(batch)
        loss = self.calc_loss(preds, uncert, y)
        #results = {'train_loss': loss}
        self.validation_step_outputs["preds"].append(preds)
        self.validation_step_outputs["y"].append(y)
        results = self.calc_stats(preds, y, phase="val")
        #results.update(self.calc_stats(preds_p, y_p, preds_d, y_d, phase="val"))
        self.log_dict(results,
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.validation_step_outputs["preds"],dim=0)
        y = torch.cat(self.validation_step_outputs["y"], dim=0)
        results = self.calc_stats(preds, y, phase="val")
        self.log_dict(results,
                      on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs = {"preds": [], "y": []}

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        scheduler = WarmUpCosine(optimizer, warmup=30, period=100, decrease_coef=0.01)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=30):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.encoder)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(modules=pl_module.encoder, optimizer=optimizer,
                                              train_bn=True, lr=4e-5
                        )
            #warmup_cosine = pl_module.trainer.lr_scheduler_configs[0].scheduler
            pl_module.trainer.lr_scheduler_configs[0].scheduler = WarmUpCosine(optimizer=optimizer,
                                                                               warmup=5,
                                                                               period=300,
                                                                               decrease_coef=
                                                                               0.01,
                                                                               last_epoch=-1)


early_stop_callback = EarlyStopping(monitor="val_rmse_epoch", min_delta=0.00, patience=20, verbose=False, mode="min")

