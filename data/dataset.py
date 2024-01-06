import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset
from PIL import Image
import pdb
from numpy import random
import glob
from skimage.transform import resize
import numpy as np
from skimage.exposure import equalize_hist as equalize
from skimage.draw import random_shapes
from skimage.filters import gaussian

	
def pad(data):
	
	pImg = torch.zeros((1,256,256))
	h = (int((256-data.shape[1])/2) )
	w = int((256-data.shape[2])/2)
	if w == 0: 
		pImg[0,np.abs(h):(h+data.shape[1]),:] = (data[0]) 
	else: 
		pImg[0,:,np.abs(w):(w+data.shape[2])] = (data[0])
	return pImg

class lungData(Dataset):
	def __init__(self,data_dir = '', process=False,
					hflip=False,vflip=False,rot=0,p=0.5,rMask=0,block=False,
					blur=True,transform=None):

		super().__init__()
		self.h = 256
		self.w = 256
		self.data_dir = data_dir
		self.hflip = hflip
		self.vflip = vflip
		self.rot = rot
		self.rMask = rMask
		self.blur = blur  # Use blurry masks
		self.block = block
#		pdb.set_trace()
		if process:
			self.process()
		self.data, self.targets,self.edges = torch.load('lungData2000n.pt')

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, index):

		image, label,edg = self.data[index], self.targets[index],self.edges[index]
		image, label,edg = TF.to_pil_image(image), TF.to_pil_image(label),TF.to_pil_image(edg)
		image, label,edg = TF.to_tensor(image), TF.to_tensor(label),TF.to_tensor(edg)

		return image, label,edg

	def process(self):
			mask = sorted(glob.glob(self.data_dir+'datasets\\labelsT\\*.png'))
			edge = sorted(glob.glob(self.data_dir+'datasets\\edgeT\\*.png'))
			data = sorted(glob.glob(self.data_dir+'datasets\\images\\*.jpg'))
			N = 6000
			images = torch.zeros((N,1,self.h,self.w))
			labels = torch.zeros((N,1,self.h,self.w))
			edges = torch.zeros((N,1,self.h,self.w))
			for index in range(N):
				image = Image.open(data[index])
				label = Image.open(mask[index])
				edg=Image.open(edge[index])
				h = int(image.height/(image.width/self.w))
				if h > self.h:
					self.w = int(image.width/(image.height/self.h))
				image, label, edg = TF.resize(image,(self.h,self.w)), TF.resize(label,(self.h,self.w)),TF.resize(edg,(self.h,self.w))
				image, label,edg = TF.to_tensor(image), TF.to_tensor(label),TF.to_tensor(edg)

				image, label,edg = pad(image), pad(label),pad(edg)
				images[index],labels[index],edges[index] = image, label,edg
			torch.save((images,labels,edges),self.data_dir+'lungData256n.pt')




