
import torch
import glob
import pandas as pd
import numpy as np

import os
import time

from spikedet.model.spike_net import SpikeNet
from spikedet import DATA_DIR

from spikedet.dataset.trainset import save_to_traineset
from spikedet import CARDIO_RR_MEAN, CARDIO_RR_SCALE

'''
SpikeDetector - Base detector of spikes 
'''
class SpikeDetector:

    def __init__(self, path_to_model: str, win_size : int = 32, threshold = 0.5, batch_size : int = 64, device=None):
        self.threshold = threshold
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.win_size = win_size
        self.batch_size = batch_size

        self.model = torch.load(path_to_model, map_location=self.device)
        if not isinstance(self.model, SpikeNet):
            pretrained_dict = self.model['state_dict']
            pretrained_dict = {key.replace("model.", ""): value for key, value in pretrained_dict.items() if 'model.' in key}
            self.model = SpikeNet()
            self.model.load_state_dict(pretrained_dict)
        self.model.to(self.device)
        self.model.eval()

    def save_model(self, path):
        torch.save(self.model, path)

    def _run_model(self, x):
        while torch.no_grad():
            x = (x - CARDIO_RR_MEAN) / CARDIO_RR_SCALE
            #x = torch.tensor(x, dtype=torch.float32, device=self.device)
            y = self.model(x)
            return torch.sigmoid(y)


    def predict(self, x, return_prob=False):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        win_det = self.win_size - 2*self.model.padding
        #win_count = (x.shape[0] - self.model.padding - self.win_size) // win_det + 1
        win_count = (x.shape[0] - 2*self.model.padding) // win_det
        sliding = torch.arange(win_count) * win_det #+ self.model.padding
        win_batches = torch.tensor_split(sliding, win_count // self.batch_size)
        indices = torch.arange(self.win_size)

        y = torch.zeros_like(x)

        for batch_indices in win_batches:
            batch_indices = batch_indices.unsqueeze(-1) + indices
            batch_data = x[batch_indices]
            pred = self._run_model(batch_data)
            batch_indices = batch_indices[...,self.model.padding : -self.model.padding]
            y[batch_indices.flatten()] = pred.flatten()
        dets = (y >= self.threshold).int()
        if return_prob:
            return dets.detach().numpy(), y.detach().numpy()
        return dets.detach().numpy()

'''
SpikeDetectorOnline - Buffered detector for online data. Slightly slow. 
'''

class SpikeDetectorOnline(SpikeDetector):

    def __init__(self, path_to_model: str, win_size : int = 32, threshold = 0.5, buffer_size : int = 64, device=None):
        assert buffer_size >= win_size
        super(SpikeDetectorOnline, self).__init__(path_to_model, win_size, threshold, 1, device)
        self.ring = torch.zeros(2*win_size, dtype=torch.float32, device=self.device)
        self.underflow_index = 0
        self.data_index = 0
        self.win_indices = torch.arange(self.win_size, device=self.device)

    # x is sequenced
    def predict(self, x, return_prob=False):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        emplace = min(self.ring.shape[0] - self.underflow_index, x.shape[0])
        self.ring[self.underflow_index : self.underflow_index + emplace] = x[:emplace]
        self.underflow_index += emplace
        win_det = self.win_size - 2 * self.model.padding
        batch_size = (self.underflow_index  - 2 * self.model.padding) // win_det

        if batch_size > 0:
            indices = torch.arange(batch_size, device=self.device).unsqueeze(-1) * win_det + self.win_indices
            batch = self.ring[indices]

            pred = self._run_model(batch)
            pred_indices = indices[..., self.model.padding: -self.model.padding] + self.data_index

            shifts = batch_size * win_det + self.model.padding
            self.ring = torch.roll(self.ring, -shifts, dims=0)
            self.underflow_index -= shifts
            self.data_index += shifts
            dets = (pred >= self.threshold).int()
            return (dets.flatten().detach().numpy(), pred_indices.flatten().detach().numpy(), pred.flatten().detach().numpy()) \
                if return_prob  else  (dets.flatten().detach().numpy(), pred_indices.flatten().detach().numpy())

        return (None, None, None) if return_prob else (None, None)


def load_test():
    data = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, 'TEST/*.txt'))):
        basename = os.path.basename(path)
        id = os.path.splitext(basename)[0]
        csv = pd.read_csv(path, sep='\t', header=None, names=['time', 'timestamp', 'value'])
        csv['id'] = pd.Series([id]).repeat(csv.index.size).reset_index(drop=True)
        data.append(csv)

    return pd.concat(data, ignore_index=True)

def test_online_detector():
    data = load_test()
    det = SpikeDetectorOnline(DATA_DIR / 'spikedet.pkl', device='cpu', threshold=0.5)
    x = data['value'].to_numpy()
    splits = np.array_split(x, 1000)

    for split in splits:
        pred, indices = det.predict(split)
        if pred is not None:
            detections = indices[pred > 0]
            if len(detections) > 0:
                print(detections)

def test_detector():
    data = load_test()
    det = SpikeDetector(DATA_DIR / 'spikedet.pkl', device='cpu', threshold=0.5)
    x = data['value'].to_numpy()
    tic = time.time()
    pred, prob = det.predict(x, return_prob=True)
    # pred, prob = detector.predict(d, return_prob=True)
    elapsed = time.time() - tic
    print(f'Elapsed time {elapsed} s')
    wrongs = pred.nonzero()[0]

    data['pred'] = pred
    data['prob'] = prob

    save_to_traineset(data, DATA_DIR / "detector_result.csv", value_name='value', aux_name='prob', label_name='pred')
    print(wrongs)


def main():

    test_online_detector()
    test_detector()



if __name__ == "__main__":
    main()



