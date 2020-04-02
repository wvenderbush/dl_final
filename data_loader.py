import csv
import requests
import shutil
import time
import torch
import statistics as stats
import numpy as np
from pathlib import Path
from os import path
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
import pandas as pd
import matplotlib as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
#from scipy.misc import imsave, imresize
from matplotlib.pyplot import imread


class PaintingDataset(Dataset):
	def __init__(self, transform=None):
		self.dataframe = grab_frame('data/data_clean_full.csv')
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
		img_obj = Image.open("data/" + img_name)
		# if self.transform != None: # TODO: DOING THIS ON THE FLY ALL THE TIME MIGHT BE COSTLY, WINSTON, CAN YOU FIND A WAY TO PREPARE THIS IN INIT?
		img_tensor = self.transform(img_obj)
		# img_tensor = self.to_tensor(img_obj)
		img_label = self.labels[index]

		return (img_tensor, img_label)


def grab_frame(frame_path):
	dataset = pd.read_csv(frame_path, encoding="latin-1")
	return dataset


def exp_stats(frame):
	print(frame.describe())
	times = frame["TIMELINE"]
	t_set = set(times)
	t_dict = {}
	print(t_set)
	for i in t_set:
		t_dict[i] = 0
	aps = frame["PATH"]
	t_w = []
	t_h = []
	for p, t in zip(aps, times):
		#image = Image.open("data/" + p)
		#width, height = image.size
		#t_w.append(width)
		#t_h.append(height)
		t_dict[t] += 1
		#image.close()
	#print("Median Width: " + str(stats.median(t_w)))
	#print("Median Height: " + str(stats.median(t_h)))
	for i in sorted(t_set):
		print(str(i) + ": " + str(t_dict[i]))

	objects = sorted(t_set)
	y_pos = objects
	print(y_pos)
	#y_pos = np.arange(len(objects))
	performance = []
	for i in y_pos:
		performance.append(t_dict[i])

	print("Total Data Points: " + str(sum(performance)))

	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, y_pos)
	plt.ylabel('Data Points')
	plt.title('Data Sparsity')

	plt.show()

def p_bw(frame):
	aps = frame["PATH"]
	t = len(aps)
	bw = 0
	c = 0
	count = 1
	for p in aps:
		print("Picture: " + str(count) + "/" + str(t))
		image = imread("data/" + p)
		if(len(image.shape)<3):
		      bw += 1
		else:
		      c += 1
		count += 1
	print("B/W: " + str(bw / t * 100) + "%")
	print("Color: " + str(c / t * 100) + "%")








#if __name__ == '__main__':
	#dataset = PaintingDataset()
	#print(len(dataset))
	#print(dataset[20])
	#exp_stats(grab_frame('data/data_clean_full.csv'))
	#p_bw(grab_frame('data/data_clean_full.csv'))
	#print(grab_labels())






