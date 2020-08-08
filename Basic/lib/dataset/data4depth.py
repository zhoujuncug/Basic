import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self,data, is_train=True, is_augment=True):
        super().__init__()

        self.data = np.load(data, allow_pickle=True)
        self.is_train = is_train
        self.is_augment = is_augment

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x, y = self.data[0, idx], self.data[1, idx]
        x = cv2.resize(x, (28, 28))[None, ...]
        x[x>6000] = 6000
        if y > 6000:
            y = 6000

        x[x < 400] = 400
        if y < 500:
            y = 500
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.tensor(y)

        return x, y