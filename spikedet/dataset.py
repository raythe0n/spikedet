
import numpy as np
from spikedet import DATA_DIR

import pandas as pd
import os
import glob

from datetime import datetime as dt
from spikedet.dataset.trainset import *


if __name__ == "__main__":

    df_train = load_from_trainset(DATA_DIR / 'labeled'/'train')
    df_val = load_from_trainset(DATA_DIR / 'labeled' / 'val')

    df_clear_train = load_cogninn(DATA_DIR / 'clear'/'train')
    df_clear_val = load_cogninn(DATA_DIR / 'clear' / 'val')

    df_clear_train['y'] = pd.Series([0]).repeat(df_clear_train.index.size).reset_index(drop=True)
    df_clear_val['y'] = pd.Series([0]).repeat(df_clear_val.index.size).reset_index(drop=True)


    #df_train = pd.concat([df_train[['id', 'x', 'y']], df_clear_train[['id', 'x', 'y']]], ignore_index=True)
    #df_val = pd.concat([df_val[['id', 'x', 'y']], df_clear_val[['id', 'x', 'y']]], ignore_index=True)

    df_train.to_csv(DATA_DIR / 'train.csv', index=False)
    df_val.to_csv(DATA_DIR / 'val.csv', index=False)







