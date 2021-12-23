import os
from dataclasses import dataclass
from uuid import uuid1

import hydra
import numpy as np
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed, dump, load
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import precision_recall_curve

import json
import torch
import random

from spikedet.dataset.dataset import CardioDataset
from spikedet.dataset.folds import split_dataframe

from spikedet.model.spike_net import SpikeNet
from spikedet import DATA_DIR, NUM_CORES, CHECKPOINTS_DIR, LOGS_DIR, TRAIN_DATA_PATH

from spikedet import CARDIO_RR_MEAN, CARDIO_RR_SCALE

from model.net import CardioSystem



def train(name, train_ds, val_ds, batch_size=1024, num_worker=0, pruning_callback=None):
    torch.autograd.set_detect_anomaly(True)
    train_dataloader = DataLoader(
        train_ds, num_workers=num_worker, pin_memory=False, shuffle=True, batch_size=batch_size
    )
    val_dataloader = DataLoader(
        val_ds, num_workers=num_worker, pin_memory=False, shuffle=False, batch_size=batch_size
    )

    model = SpikeNet(win_size=train_ds.win_size, padding=train_ds.padding)
    system = CardioSystem(model, train_weight=train_ds.pos_weight() * 0.25,
                          val_weight=val_ds.pos_weight())


    experiment_checkpoints_dir = f"{CHECKPOINTS_DIR}/{name}"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=experiment_checkpoints_dir,  monitor="Val/loss")
    early_stopping_callback = pl.callbacks.EarlyStopping(patience=50,  monitor="Val/loss")

    monitor_gpu_callback = pl.callbacks.GPUStatsMonitor()
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, early_stopping_callback, monitor_gpu_callback, lr_monitor_callback]

    if pruning_callback is not None:
        callbacks.append(pruning_callback)

    logger = TensorBoardLogger(save_dir=LOGS_DIR, name=f"{name}")


    trainer = pl.Trainer(logger=logger, callbacks=callbacks, gpus=1, max_epochs=200, log_every_n_steps=25)
    trainer.fit(system, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    #val_df = pd.concat([val_ds.df, system.val_data], axis=1)
    #os.makedirs(DATA_DIR / name)
    #val_df.to_csv(DATA_DIR / name / 'result.csv')

    logger.finalize(status="success")

    trainer.save_checkpoint(filepath=f"{experiment_checkpoints_dir}/best.ckpt")


def train_folds(df, win_size=32, padding=4, batch_size=1024, num_worker=0):
    folds = split_dataframe(df)
    uuid = uuid1()
    for i, fold in enumerate(folds):
        train_df = fold["train"]
        val_df = fold["val"]
        train_ds = CardioDataset(train_df['x'].values, train_df['y'].values, win_size, padding=padding, aug=0.5)
        val_ds = CardioDataset(val_df['x'].values, val_df['y'].values, win_size, padding=padding)

        name = f"{uuid}/FOLD_{i}"
        print(f'Fold {i}, pos_weight={train_ds.pos_weight()}')

        train(name, train_ds, val_ds, batch_size, num_worker)


def train_all(df, win_size = 32, padding=4, batch_size=1024, num_worker=NUM_CORES):
    dataset_ids = df["id"].unique()
    random.shuffle(dataset_ids)

    size = int(len(dataset_ids) * 0.05)
    val_ids = dataset_ids[:size]
    train_ids = dataset_ids[size:]

    train_df = df.loc[df["id"].isin(train_ids)].reset_index(drop=True).copy()
    val_df = df.loc[df["id"].isin(val_ids)].reset_index(drop=True).copy()

    train_ds = CardioDataset(train_df, win_size, padding=padding, aug=0.5)
    val_ds = CardioDataset(val_df, win_size, padding=padding)

    train(f"{uuid1()}/ALL", train_ds, val_ds, batch_size)

def update_scaler(x):
    scaler = RobustScaler(quantile_range=(15, 85))
    x = x.reshape(-1, 1)
    scaler.fit(x)
    print(f'Mean {scaler.center_}, Scale {scaler.scale_}')
    return scaler.transform(x)


def main():
    df = pd.read_csv(TRAIN_DATA_PATH)
    train_folds(df)


if __name__ == "__main__":
    main()