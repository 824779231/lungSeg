import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data.dataset import lungData
import torch.nn as nn
from models.GMSCU-Net import U_net
from models.loss import get_loss
import time
from utils.tools import makeLogFileU, writeLogU,dice_loss
from torchvision.utils import save_image
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--data_path', type=str, default='',help='Path to data.')
parser.add_argument('--aug',action='store_true', default=False,help='Use data aug.')
parser.add_argument('--blur',action='store_true', default=False,help='Use blurry masks.')
parser.add_argument('--block',action='store_true', default=False,help='Use block masks')
parser.add_argument('--hidden', type=int, default=32, help='Number of filters')

fName = time.strftime("%Y%m%d_%H_%M")
logFile = 'logs/'+fName+'.txt'

args = parser.parse_args()

print("Using U-Net without VAE")
fName = fName+'_unet'

device = torch.device('cuda:0')


### Choose augmentation method
if args.aug:
	dataset = lungData(data_dir=args.data_path,blur=args.blur,
					block=args.block,hflip=False,vflip=False,rot=15,p=0.1,rMask=50)
	print("Using data augmentation....")
	if args.block:
		print("Using block masks in inputs")
		fName = fName+'_block'
	elif args.blur:
		print("Using diffuse noise in inputs")
		fName = fName+'_diff'
	else:
		print("Using both masks in inputs")
		fName = fName+'_both'
else:
	dataset = lungData(data_dir=args.data_path,
						hflip=True,vflip=True,rot=15,p=0.1,rMask=0)
	print("Standard augmentation...")
	fName = fName+'_noAug'

fName = fName+'_'+repr(args.hidden)+'_hid'

# Location to save validation visualization at each epoch
if not os.path.exists('vis/'+fName):
	os.mkdir('vis/'+fName)
# Location to save current best model
if not os.path.exists('saved_models/'+fName):
	os.mkdir('saved_models/'+fName)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.25 * dataset_size))

np.random.shuffle(indices)
valid_indices, train_indices = indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)

print("Number of train/valid patches:",(len(train_indices),len(valid_indices)))

net = U_net(nhid=32)
nParam = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Model"+fName+"Number of parameters:%d"%(nParam))
with open(logFile,"a") as f:
    print("Model:"+fName+"Number of parameters:%d"%(nParam),file=f)
makeLogFileU(logFile)

#criterion, criterion_val = get_loss()
criterion = nn.BCELoss(reduction='mean')

net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

nTrain = len(train_loader)
nValid = len(valid_loader)

minLoss = 1e5
convIter=0
convCheck = 20

iter_num = 0
for epoch in range(args.epochs):
	trLoss = []
	vlLoss = []

	t = time.time()
	for step, (patch, mask,edg) in enumerate(train_loader):
		patch = patch.to(device)
		mask = mask.to(device)
		edg =edg.to(device)
		pred, ed = net(patch)
		pred = torch.sigmoid(pred)
		rec_loss = criterion(target=mask, input=pred)
		#edg_loss = criterion(target=edg, input=ed)
		edgLoss = dice_loss(ed, edg)
		diceLoss = dice_loss(pred, mask)
		loss = rec_loss+diceLoss+0.1*edgLoss
		'''loss=criterion((pred,ed),(mask,edg)).mean()'''
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		'''lr_ = args.lr * (1.0 - iter_num / (args.epochs * len(train_loader))) ** 0.9
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr_'''
		iter_num = iter_num + 1
		trLoss.append(loss.item())
		if (step+1) % 5 == 0:
			with torch.no_grad():
				for idx, (patch, mask,edg) in enumerate(valid_loader):
					patch = patch.to(device)
					mask = mask.to(device)
					edg = edg.to(device)
					pred,ed = net(patch)
					pred = torch.sigmoid(pred)

					rec_loss = criterion(target=mask, input=pred)
					#edg_loss = criterion(target=edg, input=ed)
					edgLoss = dice_loss(ed, edg)
					diceLoss = dice_loss(pred, mask)
					loss = rec_loss+diceLoss+0.1*edgLoss
					'''loss = criterion_val((pred, ed), (mask, edg)).mean()'''
					vlLoss.append(loss.item())

					break
				print ('Epoch [{}/{}], Step [{}/{}], TrLoss: {:.4f}, VlLoss: {:.4f}'
					.format(epoch+1, args.epochs, step+1,
							nTrain, trLoss[-1], vlLoss[-1]))
	epValidLoss =  np.mean(vlLoss)

	if (epoch+1) % 1 == 0 and epValidLoss < minLoss:
		convIter = 0
		minLoss = epValidLoss
		print("New max: %.4f\nSaving model..."%(minLoss))
		torch.save(net.state_dict(),'saved_models/'+fName+'/epoch_%03d.pt'%(epoch+1))
		img = torch.zeros((2*mask.shape[0],3,mask.shape[2],mask.shape[3]))
		img[::2] = patch
		pred =  ((pred.detach() > 0.5).float() + 2*mask).squeeze()
		img[1::2,0][pred == 1] = 0.55
		img[1::2,2][pred == 2] = 0.6
		img[1::2,1][pred == 3] = 0.75
		save_image(img,'vis/'+fName+'/epoch_%03d.jpg'%(epoch+1))
	else:
		convIter += 1
	writeLogU(logFile, epoch, np.mean(trLoss),
					   epValidLoss, time.time()-t)
	if convIter == convCheck:
		print("Converged at epoch %d"%(epoch+1-convCheck))
		break
	elif np.isnan(epValidLoss):
		print("Nan error!")
		break