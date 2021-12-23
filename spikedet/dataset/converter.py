
import numpy as np
from spikedet import RANGED_DATASET_DIR, FULL_TRAIN_DATA_PATH, DATA_DIR

import pandas as pd
import os
import glob
from splitter import split_spikes


def convert_from_ranged(data_dir, max_range=20):
    frames = []
    for f in glob.glob(str(data_dir / '*_ranged.txt')):
        name = f[:f.rfind('_')]
        id = os.path.basename(name)
        data_name = name + '.txt'

        if os.path.isfile(data_name):
            data = pd.read_csv(data_name, header=None)
            df = pd.DataFrame(0, index=np.arange(len(data)), columns=['id', 'time', 'x', 'y'])
            df['x'] = data
            df['id'][:] = id

            corrupted = []

            if  os.path.getsize(f) > 0:
                ranged = pd.read_csv(f, sep=' ', header=None)

                for index, ranges in ranged.iterrows():
                    df.loc[ranges[0]:ranges[1], 'y'] = 1
                    if ranges[1] - ranges[0] > max_range:
                        corrupted.append(ranges)

            for cor in corrupted:
                df.drop(range(cor[0], cor[1]+1), inplace=True)
            df.reset_index(drop=True, inplace=True)

            df['time'] = df['x'].cumsum()
            #if len(corrupted) == 0:
            frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    return df

def merge_val_folds(dir):
    folds = []
    for f in glob.glob(os.path.join(dir, '**/*.csv')):
        val_df = pd.read_csv(f)
        val_df.drop(val_df.columns[val_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        folds.append(val_df)

    df = pd.concat(folds, ignore_index=True)
    return df

def save_fails_as_ranged(df, dir, name):
    base_name = os.path.join(dir, name)

    df['x'].to_csv(base_name + '.csv', header=False, index=False)
    spikes = df['spike'].to_numpy().nonzero()[0]
    pd.DataFrame(dict(begin=spikes, end=spikes + 1)).to_csv(base_name + '_spike_ranged.csv', header=False, index=False, sep=' ')

    error = df['dets'] - df['target']

    fp = (error > 0).to_numpy().nonzero()[0]
    fn = (error < 0).to_numpy().nonzero()[0]
    all = (error != 0).to_numpy().nonzero()[0]
    pd.DataFrame(dict(begin=fp, end=fp+1)).to_csv(base_name + '_fp_ranged.csv', header=False, index=False, sep=' ')
    pd.DataFrame(dict(begin=fn, end=fn+1)).to_csv(base_name + '_fn_ranged.csv', header=False, index=False, sep=' ')
    pd.DataFrame(dict(begin=all, end=all + 1)).to_csv(base_name + '_all_ranged.csv', header=False, index=False, sep=' ')

def save_thresholded_fails_as_ranged(df, dir, name, threshold=0.4):
    base_name = os.path.join(dir, name)

    df['x'].to_csv(base_name + '.csv', header=False, index=False)
    #spikes = df['spike'].to_numpy().nonzero()[0]
    #pd.DataFrame(dict(begin=spikes, end=spikes + 1)).to_csv(base_name + '_spike_ranged.csv', header=False, index=False, sep=' ')

    dets = (df['pred'] > threshold).to_numpy().astype(int)

    error = dets - df['target'].to_numpy()

    fp = (error > 0).nonzero()[0]
    fn = (error < 0).nonzero()[0]
    all = (error != 0).nonzero()[0]
    pd.DataFrame(dict(begin=fp, end=fp+1)).to_csv(base_name + f'_th_{threshold}_fp_ranged.csv', header=False, index=False, sep=' ')
    pd.DataFrame(dict(begin=fn, end=fn+1)).to_csv(base_name + f'_th_{threshold}_fn_ranged.csv', header=False, index=False, sep=' ')
    pd.DataFrame(dict(begin=all, end=all + 1)).to_csv(base_name + f'_th_{threshold}_all_ranged.csv', header=False, index=False, sep=' ')

def save_splits_as_ranged(df, dir, name, num_spikes_per_file=0):

    indices = df['spike'].to_numpy().nonzero()[0]
    splits = np.array_split(indices, int(indices.size / num_spikes_per_file))
    prev_point = 0
    for i, split in enumerate(splits):
        base_name = os.path.join(dir, name + f'_{i}')
        last = split[-1] + 1
        df['x'].loc[prev_point:last].to_csv(base_name + '.csv', header=False, index=False)
        pd.DataFrame(dict(begin=split-prev_point, end=split - prev_point + 1)).to_csv(base_name + '_ranged.csv', header=False,
                                                        index=False, sep=' ')
        prev_point = last + 1



def update_labels_from_ranged(labels, ref_path, target = 1):
    ref_df = pd.read_csv(ref_path, sep=' ', header=None)
    #target = df['target'].copy()
    #error = df['dets'] - df['target']
    #target.loc[error != 0] = 0

    # Ranged MUST be size 1
    validate_ranged = (ref_df[1] - ref_df[0] != 1).to_numpy().nonzero()[0]
    assert validate_ranged.size == 0

    labels.loc[ref_df[0]] = target
    #err = (target - df['target']).to_numpy().nonzero()[0]
    return labels

def refine_labels_from_filtered(df, orig_paths, filtered_paths):
    df = df.copy()
    target = df['target']
    for orig in orig_paths:
        target = update_labels_from_ranged(target, orig, 0)

    for filt in filtered_paths:
        target = update_labels_from_ranged(target, filt, 1)

    df['refined'] = target
    return df



if __name__ == "__main__":
    df = merge_val_folds(DATA_DIR/'26deed5e-5fea-11ec-a89b-18c04d961554')
    os.makedirs(DATA_DIR/'dataset_refined2', exist_ok=True)
    df.to_csv(DATA_DIR/'dataset_refined2/covid19_dataframe.csv')
    save_fails_as_ranged(df, DATA_DIR/'dataset_refined2', 'covid20')

    df = pd.read_csv(DATA_DIR/'dataset_refined3/covid19_refined4_dataframe.csv',index_col=0)
    save_splits_as_ranged(df, DATA_DIR/'dataset_refined3', 'covid21', 50)

    #df = pd.read_csv(DATA_DIR/'dataset_refined/covid19_dataframe.csv',index_col=0)

    #df_refined = refine_labels_from_filtered(df, [DATA_DIR/'dataset_refined/covid19_all_ranged.csv'], [DATA_DIR / 'dataset_refined/covid19_all_filtered_ranged.csv'])
    #df_refined.to_csv(DATA_DIR / 'dataset_refined/covid19_refined_dataframe.csv')

    #save_thresholded_fails_as_ranged(df, DATA_DIR/'dataset_refined', 'covid19')

    #df = convert_from_ranged(RANGED_DATASET_DIR)
    #y = split_spikes(df['x'].values, df['y'].values)
    #df["spike"] = y
    #df.to_csv(FULL_TRAIN_DATA_PATH)






