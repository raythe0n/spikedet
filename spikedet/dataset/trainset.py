"""
TRAINSET format for labelling
Refer to https://trainset.geocene.com/

Example:
    series,timestamp,value,label
    series_a,2019-01-14T16:26:37.000Z,29.4375,
    series_b,2019-01-14T16:26:37.000Z,-0.5625,

"""

import numpy as np
from spikedet import DATA_DIR

import pandas as pd
import os
import glob

from datetime import datetime as dt

def save_to_traineset(df, path, value_name = 'x', aux_name = None, label_name = None, label_map={1:'spike'}):

    time = pd.to_datetime(pd.Series(df.index), unit='s')
    # time = time.map(lambda x: dt.strftime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    timestamp = time.map(lambda x: dt.isoformat(x, timespec='milliseconds') + 'Z')
    label = pd.Series(['']).repeat(time.size).reset_index(drop=True)
    series = pd.Series(['RR']).repeat(time.size).reset_index(drop=True)
    value = df[value_name]

    if label_name is not None:
        for k, v in label_map.items():
            label[df[label_name] == k] = v

    if aux_name is not None:
        serie_aux = pd.Series([aux_name]).repeat(time.size).reset_index(drop=True)
        value = pd.concat([value, df[aux_name]], axis=0, ignore_index=True)
        series = pd.concat([series, serie_aux], axis=0, ignore_index=True)
        timestamp = pd.concat([timestamp, timestamp], axis=0, ignore_index=True)
        label = pd.concat([label, label], axis=0, ignore_index=True)


    trainset_dict = dict(series=series, timestamp=timestamp, value=value, label=label)

    trainset_df = pd.DataFrame(trainset_dict)
    trainset_df.sort_values(['timestamp'], inplace=True)
    trainset_df.to_csv(path, sep=',', index=False)


def save_to_traineset_splitted(df, dir, split_by = 'id', value_name = 'x', aux_name = None, label_name = None, label_map={1:'spike'}):

    for id, grp in df.groupby(split_by):
        # gdf = grp.sort_values(["time"]).reset_index(drop=True).copy()
        gdf = grp.sort_index().reset_index(drop=True)
        save_to_traineset(gdf, dir/f'{id}.csv', value_name, aux_name, label_name, label_map)


def load_from_trainset(dir, label_map=dict(spike=1, extra=2)):
    dfs = []
    for f in glob.glob(os.path.join(dir, '*-labeled.csv')):
        id = os.path.splitext(os.path.basename(f))[0]
        id = '-'.join(id.split('-')[:-1])
        id_df = pd.read_csv(f)
        # remove markup
        id_df = id_df.loc[id_df['series'] == 'RR'].reset_index(drop=True).sort_values(['timestamp'])
        id_df.drop(['series'], axis=1, inplace=True)
        id_df['id'] = pd.Series([id]).repeat(id_df.index.size).reset_index(drop=True)
        id_df['time'] = id_df['value'].cumsum()
        dfs.append(id_df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    df['label'].replace(float('NaN'), 0, inplace=True)

    for k, v in label_map.items():
        df['label'].replace(k, v, inplace=True)

    count = np.count_nonzero(df['label'].to_numpy())
    print(f'Find {count} labelled samples')

    df.rename(columns=dict(value='x', label='y'), inplace=True)
    df.drop(['timestamp'], axis=1, inplace=True)
    return df



def load_cogninn(dir):
    data = []
    for path in sorted(glob.glob(os.path.join(dir, '*.txt'))):
        basename = os.path.basename(path)
        id = os.path.splitext(basename)[0]
        csv = pd.read_csv(path, sep='\t', header=None, names=['time', 'timestamp', 'x'])
        csv['id'] = pd.Series([id]).repeat(csv.index.size).reset_index(drop=True)
        data.append(csv)

    return pd.concat(data, ignore_index=True)



if __name__ == "__main__":
    #df = load_cogninn(DATA_DIR / '25_dec')
    #save_to_traineset_splitted(df, DATA_DIR / '25_dec/trainset')


    #df = merge_val_folds(DATA_DIR/'82786bf6-5929-11ec-bdd4-18c04d961554')
    #os.makedirs(DATA_DIR/'dataset_refined', exist_ok=True)
    #df.to_csv(DATA_DIR/'dataset_refined/covid19_dataframe.csv')

    #fix_multiple_ranged(DATA_DIR / 'dataset_refined/covid19_all_filtered_ranged.csv', DATA_DIR / 'dataset_refined/covid19_all_fix_filtered_ranged.csv')

    df = load_from_trainset(DATA_DIR / '21-12-2021')
    df.to_csv(DATA_DIR/'covid19_filtered.csv')

    df_negative = load_cogninn(DATA_DIR / 'TEST')
    df_negative['y'] = pd.Series([0]).repeat(df_negative.index.size).reset_index(drop=True)
    df_negative.drop(['timestamp'], axis=1, inplace=True)

    df_all = pd.concat([df, df_negative], ignore_index=True)
    df_all.to_csv(DATA_DIR / 'covid19_merged.csv')

    #df = pd.read_csv(DATA_DIR/'dataset_refined/covid19_refined4_dataframe.csv',index_col=0)
    #save_dataframe_as_traineset(df, DATA_DIR / 'trainset')





