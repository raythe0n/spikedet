
import torch
import glob
import pandas as pd

import os
import time

from spikedet.model.spike_net import SpikeNet
from spikedet import DATA_DIR

from spikedet.dataset.trainset import save_to_traineset
from spikedet import CARDIO_RR_MEAN, CARDIO_RR_SCALE

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
            #x = torch.tensor(x, dtype=torch.float32, device=self.device)
            y = self.model(x)
            return torch.sigmoid(y)


    def predict(self, x, return_prob=False):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        win_det = self.win_size - 2*self.model.padding
        win_count = (x.shape[0] - self.model.padding - self.win_size) // win_det + 1
        sliding = torch.arange(win_count) * win_det  + self.model.padding
        win_batches = torch.tensor_split(sliding, win_count // self.batch_size)
        indices = torch.arange(self.win_size)

        y = torch.zeros_like(x)

        #Normalization
        x = (x - CARDIO_RR_MEAN) / CARDIO_RR_SCALE

        for batch_indices in win_batches:
            batch_indices = batch_indices.unsqueeze(-1) + indices
            #np.expand_dims(batch_indices, axis=-1) + indices

            batch_data = x[batch_indices]
            pred = self._run_model(batch_data)
            batch_indices = batch_indices[...,self.model.padding : -self.model.padding]
            y[batch_indices.flatten()] = pred.flatten()
        dets = (y >= self.threshold).int()
        if return_prob:
            return dets.detach().numpy(), y.detach().numpy()
        return dets.detach().numpy()


def load_false_covid():
    data = []
    for path in glob.glob(os.path.join(DATA_DIR, 'FALSE COVID/*.txt')):
        basename = os.path.basename(path)
        id = os.path.splitext(basename)[0]
        csv = pd.read_csv(path, sep='\t', header=None, names=['time', 'timestamp', 'value'])
        csv['id'] = pd.Series([id]).repeat(csv.index.size).reset_index(drop=True)
        data.append(csv)

    return pd.concat(data, ignore_index=True)

def main():

    #detector = SpikeDetector(DATA_DIR / 'checkpoints/b22982f8-6361-11ec-a2b7-18c04d961554/FOLD_0/best.ckpt', device='cpu')
    detector = SpikeDetector(DATA_DIR / 'spikedet.pkl', device='cpu', threshold=0.5)
    detector.save_model(DATA_DIR / 'spikedet.pkl')
    data = load_false_covid()

    tic = time.time()
    pred, prob = detector.predict(data['value'].to_numpy(), return_prob=True)
    #pred, prob = detector.predict(d, return_prob=True)
    elapsed = time.time() - tic
    print(f'Elapsed time {elapsed} s')
    wrongs = pred.nonzero()[0]

    data['pred'] = pred
    data['prob'] = prob

    save_to_traineset(data, DATA_DIR / "detector_result.csv", value_name='value', aux_name='prob', label_name='pred')
    print(wrongs)


if __name__ == "__main__":
    main()



