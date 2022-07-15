from PIL import Image
import torch
import numpy as np


class NihDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, transforms=None):
        self.data_frame = data_frame
        self.transforms = transforms
        self.len = data_frame.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        address = row['path']
        x = Image.open(address).convert('RGB')

        vec = np.array(row['disease_vec'], dtype=np.float)
        y = torch.FloatTensor(vec)

        if self.transforms:
            x = self.transforms(x)
        return x, y
