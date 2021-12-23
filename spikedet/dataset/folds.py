import os
from sklearn.model_selection import KFold
import json

from spikedet import TRAIN_SPLITTED_DATA_PATH, DATA_DIR, NUM_CORES, CHECKPOINTS_DIR, LOGS_DIR, FULL_TRAIN_DATA_PATH

def make_splits(ids, splits=10):
    kf = KFold(n_splits=splits, shuffle=True)

    val = []
    train = []

    for i, (tr_id, va_id) in enumerate(kf.split(ids)):
        val_ids = [ids[x] for x in va_id]
        train_ids = [ids[x] for x in tr_id]

        val.append(val_ids)
        train.append(train_ids)

    return dict(val=val, train=train)


def split_dataframe(df, folds_file='folds.json'):
    if not os.path.exists(DATA_DIR / folds_file):
        splits = make_splits(df["id"].unique())
        with open(DATA_DIR / folds_file, 'w') as f:
            json.dump(splits, f)
    else:
        with open(DATA_DIR / folds_file, 'r') as f:
            splits = json.load(f)

    folds = []
    for i, (tr, val) in enumerate(zip(splits['train'], splits['val'])):

        train_df = df.loc[df["id"].isin(tr)].sort_index().reset_index(drop=True)
        val_df = df.loc[df["id"].isin(val)].sort_index().reset_index(drop=True)

        folds.append(dict(train=train_df, val=val_df))
    return folds

