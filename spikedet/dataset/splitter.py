import numpy as np
import pandas as pd

from spikedet import TRAIN_DATA_PATH, TRAIN_SPLITTED_DATA_PATH

def recursive_detect_spikes(data, range=0, dist_to_min=3, margin=0.5):
    max = np.argmax(data)
    result = []

    if max < len(data) - dist_to_min:
        min = np.argmin(data[max:max+dist_to_min]) + max

        max_val = data[max]
        min_val = data[min]
        delta = max_val - min_val
        if range <=0:
            range = delta
        elif delta < range * (1-margin):
            return []
        result.append(max)

        right = recursive_detect_spikes(data[max + 1:], range, dist_to_min, margin)
        result = result + [r + max + 1 for r in right]


    if max > dist_to_min:
        result = recursive_detect_spikes(data[:max], range, dist_to_min, margin) + result

    return result

def split_spikes(x, y):

    start = -1
    y_spikes = np.zeros_like(y)

    spikes_count = 0

    for i, label in enumerate(y):
        if label == 0:
            if start+1 < i:
                data = x[start+1:i]
                spikes = recursive_detect_spikes(data)
                spikes = [s + start + 1 for s in spikes]

                y_spikes[spikes] = 1
                spikes_count += len(spikes)


            start = i

    print(spikes_count)
    return y_spikes



def main():
    #test_false_covid()
    df = pd.read_csv(TRAIN_DATA_PATH)
    y = split_spikes(df['x'].values, df['y'].values)
    df["spike"] = y
    df.to_csv(TRAIN_SPLITTED_DATA_PATH)




if __name__ == "__main__":

    main()
