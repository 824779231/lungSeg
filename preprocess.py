### raghav@di.ku.dk
### Convert Dicom to PNG
### Perform histogram equalization

from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import argparse
import pdb
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import glob
import pydicom
from skimage.io import imread, imsave
from skimage.exposure import equalize_hist as equalize
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../testData/image/',help='Path to dicom files.')
parser.add_argument('--no_equalize',action='store_true',
                        default=False,help='Do not histogram equalize images')
parser.add_argument('--ext', type=str,default='jpg',
					help='File extension: DCM/png/JPG (case-sensitive)')

args = parser.parse_args()
t = time.strftime("%Y%m%d_%H_%M")

save_dir = 'preprocessed_'+t+'/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
dicom=False
files = sorted(glob.glob(args.data+'*.'+args.ext))
N = len(files)
print("Found %d files with "%N+args.ext+' extension')
if args.ext == 'dcm':
	dicom = True

for fIdx in range(N):
	f = files[fIdx]
	if dicom:
		dcm = pydicom.dcmread(f)
		img = dcm.pixel_array
		img = img/img.max()
		if dcm.PhotometricInterpretation == 'MONOCHROME1':
			img = 1-img ### DICOM and PNG have inverted intensities!
	else:
		img = imread(f)
		#img = rgb2gray(img)
		img = img/img.max()
	if not args.no_equalize:
		img = equalize(img)

	imsave(save_dir+f.split('\\')[-1],img_as_ubyte(img))
	print("Processed %d/%d"%(fIdx+1,N))


