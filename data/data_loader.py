import csv
import requests
import shutil
import time
import torch
import numpy as np
from pathlib import Path
from os import path
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
import pandas as pd
import matplotlib as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class PaintingDataset(Dataset):
	def __init__(self, transform=None):
		self.dataframe = grab_frame('data_clean_full.csv')
		self.transform = transform
		self.to_tensor = ToTensor()
		self.to_pil = ToPILImage()
		self.images = np.asarray(self.dataframe["PATH"])
		self.labels = np.asarray(self.dataframe["TIMELINE"])
		self.len = len(self.dataframe)


	def __len__(self):
		return self.len

	def __getitem__(self, index):
		img_name = self.images[index]
		img_obj = Image.open(img_name)
		img_tensor = self.to_tensor(img_obj)
		img_label = self.labels[index]


		return (img_tensor, img_label)


def grab_frame(frame_path):
	dataset = pd.read_csv(frame_path, encoding="latin-1")
	return dataset


if __name__ == '__main__':
	dataset = PaintingDataset()
	print(len(dataset))
	print(dataset[20])
	#print(grab_labels())






