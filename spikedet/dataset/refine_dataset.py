
import numpy as np
from spikedet import RANGED_DATASET_DIR, FULL_TRAIN_DATA_PATH, DATA_DIR

import pandas as pd
import os
import glob


def merge_val_folds(dir):
    folds = []
    for f in glob.glob(os.path.join(dir, '**/*.csv')):
        val_df = pd.read_csv(f)
        val_df.drop(val_df.columns[val_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        folds.append(val_df)

    df = pd.concat(folds, ignore_index=True)
    return df


def save_fails_as_ranged(df, dir, name, threshold=None):
    base_name = os.path.join(dir, name)

    df['x'].to_csv(base_name + '.csv', header=False, index=False)
    #spikes = df['spike'].to_numpy().nonzero()[0]
    #pd.DataFrame(dict(begin=spikes, end=spikes + 1)).to_csv(base_name + '_spike_ranged.csv', header=False, index=False, sep=' ')

    th_str = ''
    if threshold is None:
        dets = df['dets'].to_numpy().astype(int)
    else:
        th_str = f'_th_{threshold}'
        dets = (df['pred'] > threshold).to_numpy().astype(int)

    error = dets - df['target'].to_numpy()

    fp = (error > 0).nonzero()[0]
    fn = (error < 0).nonzero()[0]
    all = (error != 0).nonzero()[0]
    pd.DataFrame(dict(begin=fp, end=fp+1)).to_csv(base_name + f'{th_str}_fp_ranged.csv', header=False, index=False, sep=' ')
    pd.DataFrame(dict(begin=fn, end=fn+1)).to_csv(base_name + f'{th_str}_fn_ranged.csv', header=False, index=False, sep=' ')
    pd.DataFrame(dict(begin=all, end=all + 1)).to_csv(base_name + f'{th_str}_all_ranged.csv', header=False, index=False, sep=' ')



def update_labels_from_ranged(labels, ref_path, target = 1):
    ref_df = pd.read_csv(ref_path, sep=' ', header=None)

    labels=labels.copy()

    # Ranged MUST be size 1
    validate_ranged = (ref_df[1] - ref_df[0] != 1).to_numpy().nonzero()[0]
    assert validate_ranged.size == 0

    labels.loc[ref_df[0]] = target
    #err = (target - df['target']).to_numpy().nonzero()[0]
    return labels

def fix_multiple_ranged(ref_path, out_path):
    ref_df = pd.read_csv(ref_path, sep=' ', header=None)
    #target = df['target'].copy()
    #error = df['dets'] - df['target']
    #target.loc[error != 0] = 0
    unique = ref_df[0].value_counts(sort=False)
    max_count = unique.max()
    filtered = unique[unique == max_count]
    valid = filtered.index.sort_values()

    df = pd.DataFrame({0:valid, 1:valid+1})

    df.to_csv(out_path, sep=' ', header=False, index = False)


    print(unique)


def refine_labels_from_filtered(df, orig_paths, filtered_paths):
    df = df.copy()
    target = df['target'].copy()
    for orig in orig_paths:
        target = update_labels_from_ranged(target, orig, 0)

    for filt in filtered_paths:
        target = update_labels_from_ranged(target, filt, 1)

    df['spike'] = target
    return df



if __name__ == "__main__":
    #df = merge_val_folds(DATA_DIR/'82786bf6-5929-11ec-bdd4-18c04d961554')
    #os.makedirs(DATA_DIR/'dataset_refined', exist_ok=True)
    #df.to_csv(DATA_DIR/'dataset_refined/covid19_dataframe.csv')

    #fix_multiple_ranged(DATA_DIR / 'dataset_refined/covid19_all_filtered_ranged.csv', DATA_DIR / 'dataset_refined/covid19_all_fix_filtered_ranged.csv')

    df = pd.read_csv(DATA_DIR/'dataset_refined/covid19_dataframe.csv',index_col=0)

    df_refined_0_4 = refine_labels_from_filtered(df, [DATA_DIR / 'dataset_refined/covid19_th_0.4_all_ranged.csv'],
                                             [DATA_DIR / 'dataset_refined/covid19_th_0.4_all_filtered_ranged.csv'])

    df_refined_auto = refine_labels_from_filtered(df, [DATA_DIR / 'dataset_refined/covid19_all_ranged.csv'],
                                                 [DATA_DIR / 'dataset_refined/covid19_all_fix_filtered_ranged.csv'])

    empty = df_refined_auto['spike'].copy()
    empty[:] = 0
    empty_all = update_labels_from_ranged(empty, DATA_DIR / 'dataset_refined/covid19_all_ranged.csv', 1)
    empty_th_all = update_labels_from_ranged(empty, DATA_DIR / 'dataset_refined/covid19_th_0.4_all_ranged.csv', 1)


    delta = df_refined_auto['spike'] - df_refined_0_4['spike']
    err = delta.to_numpy().nonzero()[0]
    delta_err = delta[err].to_numpy()[:, None]

    changed_all = empty_all[err]
    changed_th_all = empty_th_all[err]

    view_df = pd.DataFrame()
    view_df['err'] = delta[err]
    view_df['changed_all'] = changed_all
    view_df['changed_th_all'] = changed_th_all
    view_df.reset_index(drop=True, inplace=True)

    ind = np.arange(-10, 10)
    ind = ind[None, :] + err[:, None]

    data  = df_refined_auto['x'].to_numpy()[ind]

    df_refined = refine_labels_from_filtered(df, [DATA_DIR/'dataset_refined/covid19_all_ranged.csv',
                                                  DATA_DIR/'dataset_refined/covid19_th_0.4_all_ranged.csv'],
                                             [DATA_DIR / 'dataset_refined/covid19_all_fix_filtered_ranged.csv',
                                              DATA_DIR / 'dataset_refined/covid19_th_0.4_all_filtered_ranged.csv'])

    df_refined = df_refined[['id','time', 'x', 'y', 'spike']]

    df_refined.to_csv(DATA_DIR / 'dataset_refined/covid19_refined4_dataframe.csv')

    #df_refined = refine_labels_from_filtered(df, [DATA_DIR / 'dataset_refined/covid19_th_0.4_all_ranged.csv'],
    #                                         [DATA_DIR / 'dataset_refined/covid19_th_0.4_all_filtered_ranged.csv'])
    #df_refined.to_csv(DATA_DIR / 'dataset_refined/covid19_refined3_dataframe.csv')

   # delta = df_refined['spike'] - df_refined['refined']
   # delta = delta.to_numpy().nonzero()

    #print(delta)

    #save_thresholded_fails_as_ranged(df, DATA_DIR/'dataset_refined', 'covid19')

    #df = convert_from_ranged(RANGED_DATASET_DIR)
    #y = split_spikes(df['x'].values, df['y'].values)
    #df["spike"] = y
    #df.to_csv(FULL_TRAIN_DATA_PATH)






