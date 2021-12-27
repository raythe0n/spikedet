
from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch


from spikedet.core.lookahead import Lookahead
import numpy as np
from sklearn.metrics import average_precision_score

from spikedet.core.focal_loss import FocalLoss, FocalLossDiffused
import pandas as pd

from spikedet.core.utils import fbeta_search, threshold_search, remove_duplicates


class CardioSystem(pl.LightningModule):
    def __init__(
        self, model, train_weight=1, val_weight = 1, lr=1e-3, alpha=0.5 ):
        super().__init__()

        self.save_hyperparameters(
            {
                "lr": lr,
                "alpha": alpha,
                "hp_metric": -1,
            }
        )


        self.model = model
        self.train_loss = FocalLoss(pos_weight=train_weight)
        self.val_loss = FocalLoss(pos_weight=val_weight)

        self.loss = dict(Train=self.train_loss,
                         Val=self.val_loss)

        self.thresh = None
        self.val_data = pd.DataFrame()

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_index):
        return self.common_step(batch, batch_index, "Train")

    def validation_step(self, batch, batch_index):
        return self.common_step(batch, batch_index, "Val")

    def test_step(self, batch, batch_index):
        return self.common_step(batch, batch_index, "Test")

    def validation_epoch_end(self, outputs) -> None:
        self.common_epoch_end(outputs, "Val")
        self.plot_location_loss(outputs)

    def test_epoch_end(self, outputs):
        self.hparams.hp_metric = self.common_epoch_end(outputs, "Test")
        self.log("hp_metric", self.hparams.hp_metric)

    def common_step(self, batch, batch_idx, mode):
        inpts = batch["x"]
        actual = batch["y"]
        proba = self.model(inpts)

        loss = self.loss[mode](proba, actual)

        self.log(f"{mode}/loss", loss)

        indices = torch.arange(actual.size(-1), device=actual.device) + batch['idx'].unsqueeze(-1)

        return {"loss": loss, "proba": proba.detach(), "actual": actual, 'indices': indices}

    def plot_location_loss(self, outputs):
        histo = torch.cat([out["proba"] for out in outputs], dim=0)
        histo = torch.sigmoid(histo)
        mask = torch.cat([out["actual"] for out in outputs], dim=0)
        histo = torch.sum(histo * mask, dim=0) / torch.sum(mask, dim=0)
        print(histo)

    def common_epoch_end(self, outputs, mode):
        actual = torch.cat([out["actual"].flatten() for out in outputs])
        proba = torch.cat([out["proba"].flatten() for out in outputs])
        indices = torch.cat([out["indices"].flatten() for out in outputs])

        #actual_reduced, proba_reduced  = remove_duplicates(actual, proba, indices)

        #pred = torch.sigmoid(proba_reduced)
        y_true = actual.cpu().numpy()
        y_score = torch.sigmoid(proba).cpu().numpy()

        #best_th, best_score = threshold_search(actual_reduced.cpu().numpy(), pred.cpu().numpy())
        #recall, precision, fbeta_score, th = fbeta_search(actual_reduced.cpu().numpy(), pred.cpu().numpy())
        #ap = average_precision_score(actual_reduced.cpu().numpy(), pred.cpu().numpy())

        best_th, best_score = threshold_search(y_true, y_score)
        recall, precision, fbeta_score, th = fbeta_search(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        self.thres = th
        #Store validation data
        #target = actual_reduced.cpu().numpy()

        #pred = pred.cpu().numpy()
        #dets = np.zeros_like(target)
        #dets[pred > th] = 1

        #self.val_data = pd.DataFrame({'dets':dets, 'pred':pred, 'target':target})

        self.log(f"{mode}/thresh", best_th.item())
        self.log(f"{mode}/f1_score", best_score.item())

        self.log(f"{mode}/fbeta_score", fbeta_score.item())

        self.log(f"{mode}/precision", precision.item())
        self.log(f"{mode}/recall", recall.item())

        self.log(f"{mode}/AP_score", ap.item())

        return best_score

    def configure_optimizers(self):
        #optimizer = Lookahead(
        #    torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr), self.hparams.alpha, self.hparams.step_ahead
        #)
        #return optimizer


        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, cooldown=5 ),
                "monitor": "Val/loss",
                "frequency": 1,

            },
        }



        #return optimizer

