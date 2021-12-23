import numpy as np
from numpy.random import rand
from torch.utils.data import Dataset
from spikedet import CARDIO_RR_MEAN, CARDIO_RR_SCALE

class CardioDataset(Dataset):
    def __init__(self, x, y, win_size=32, padding=4, aug=0):
        self.win_size = win_size
        self.padding = padding
        self.aug = aug
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

        self.x = (self.x - CARDIO_RR_MEAN) / CARDIO_RR_SCALE

        assert len(x) == len(y)


    def __len__(self):
        return len(self.x) - self.win_size

    def __getitem__(self, idx):
        x = self.x[idx:idx + self.win_size]
        y = self.y[idx+self.padding:idx + self.win_size - self.padding]

        if self.aug > 0:
            x = self._augment(x, apply_prob=self.aug)

        return {'x': x, 'y': y, 'idx': idx + self.padding}


    @staticmethod
    def _augment(cardio, perm=10 / 400, shift=150 / 400, scale=0.2, apply_prob=0.5):
        apply_rand = np.random.rand(3)
        # Linear mutation
        # if apply_rand[0] > apply_prob:
        # linspace = torch.linspace(-0.5, 0.5, cardio.shape[0])
        #    linspace = np.linspace(-1, 1, cardio.shape[0], dtype=np.float32)
        #    cardio = cardio + linspace * (rand(1) - 0.5) * perm * cardio.shape[0]

        # Mean shift
        if apply_rand[1] > apply_prob:
            value = shift * (rand(1) - 0.5)
            cardio = cardio + value

        # Scale
        if apply_rand[2] > apply_prob:
            factor = 1 + scale * (rand(1) - 0.5)
            cardio = cardio * factor
        return cardio.astype(np.float32)


    def pos_weight(self):
        #label = self.data[:, 1]
        pos_count = np.count_nonzero(self.y)
        return self.y.size / (pos_count*(self.win_size-2*self.padding)) - 1

        #idx = label.nonzero()[0]
        #win = np.arange(1-self.win_size, self.win_size)
        #idx = idx[:, None] + win

        #idx = idx[np.logical_and(idx >= 0, idx < label.size) ]

        #nonzero_label = np.zeros_like(label)
        #nonzero_label[idx] = 1
        #pos_count = nonzero_label.nonzero()[0].size
        #return label.size / pos_count - 1


    @property
    def win(self):
        return self.win_size
