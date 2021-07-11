"""
Pytorch 를 이용해 Data loader 를 만든다.
"""
from torch.utils.data import *
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import torch

class AudioDataset(Dataset):
    def __init__(self, dataset_name, transforms=None):
        self.data = pd.read_csv(dataset_name)

        self.length = 1500 if dataset_name == "GTZAN" else 250
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        values = np.load(entry['values'])
        values = values.reshape(-1, 128, self.length)
        values = torch.Tensor(values)
        if self.transforms:
            values = self.transforms(values)
        target = torch.LongTensor([entry["target"]])

        return values, target


def fetch_dataloader(pkl_dir, dataset_name, batch_size, num_workers):
    dataset = AudioDataset(pkl_dir, dataset_name)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader


def get_data_loader(args):

    csv_path = '/Users/selectstar_namkill/PycharmProjects/Audio_classification/data/files/total_audio_list.csv'
    custom_dataset = AudioDataset(csv_path)

    val_csv_path = '/Users/selectstar_namkill/PycharmProjects/Audio_classification/data/files/total_audio_list.csv'
    val_custom_dataset = AudioDataset(val_csv_path)

    train_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_custom_dataset, batch_size=args.batch_size,shuffle=False)

    return train_loader, val_loader

