import csv
import requests
import shutil
import time
from pathlib import Path
from os import path
from PIL import Image
import pandas as pd
import matplotlib as plt
from torch.utils.data import Dataset


class PaintingDataset(Dataset):
    def __init__(self):
        self.samples = []
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def grab_labels():
	dataset = pd.read_csv('data_clean_full.csv', encoding="latin-1")

	X = dataset.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]].values
	Y = dataset.iloc[10].values

	return dataset


if __name__ == '__main__':
    dataset = PaintingDataset()
    print(len(dataset))
    #print(grab_labels())






