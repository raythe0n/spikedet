
from causal_net import CausalNet, TreeNet, AttnNet
import pandas as pd
from spikedet import TRAIN_SPLITTED_DATA_PATH, DATA_DIR, NUM_CORES, CHECKPOINTS_DIR, LOGS_DIR, FULL_TRAIN_DATA_PATH
import matplotlib.pyplot as plt
import numpy as np

def draw_detections(target, prob, dets, data, y):
    x = np.arange(target.shape[-1])
    plt.plot(x, data, color='b', label='RR')
    plt.plot(x, y, color='cyan', label='Markers')
    plt.plot(x, target, color='y', label='Target Spikes')

    plt.plot(x, prob, color='g', label="Probability")

    err = dets - target
    plt.plot(x, err, color='r', label='Errors')
    # show a legend on the plot
    plt.legend()
    plt.show()

def main():
    val = pd.read_csv(DATA_DIR / 'folds_data/val_9.csv')
    det = pd.read_csv(DATA_DIR / 'result.csv')

    data = val['x_norm'].values

    dets = det['dets']
    pred = det['pred']
    target = det['target']

    draw_detections(target, pred, dets, data, val['y'].values)


if __name__ == "__main__":
    main()
