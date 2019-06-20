from tqdm import tqdm
import struct
import numpy as np
import os.path
import sys
import random
import math
import string
from multiprocessing import Pool
from random import randint

import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

np.set_printoptions(threshold=np.nan)

IMZ_SZ1 = 320.0 
IMZ_SZ2 = 240.0 

SZ = 128  

srt_gbl = 0 

BLOCK_SIZE1 = 32 # sqrt(NUM_POINT / POOL_SIZE)
BLOCK_SIZE2 = 32 # sqrt(NUM_POINT / POOL_SIZE)

directory = "datasets/"
import numpy.matlib

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def centeredCrop(img, sft_h, sft_w, output_side_length):
    height, width, depth = img.shape
    height_offset = (height - output_side_length) / 2
    height_offset = int(height_offset) + sft_h
    width_offset = (width - output_side_length) / 2
    width_offset = int(width_offset) + sft_w
    #print(height_offset, sft_h, width_offset, sft_w)
    cropped_img = img[height_offset:height_offset + output_side_length, width_offset:width_offset + output_side_length]
    cropped_img = cropped_img/127.5 - 1.
    return cropped_img

def add_noise(images_des):
    all_des = images_des.astype(float)+np.multiply(10.0, np.random.randn(images_des.shape[0], 128))  
    np.clip(all_des, 0.0, 255.0) 
    return all_des 

def preprocess(images):

    filename, file_extension = os.path.splitext(images) 
    with open(filename+'.sift', mode='rb') as f: # b is important -> binary
        nameNversion = f.read(8)
        
        frmt = (f.read(12)) #the number of features
        num, nLocDim, nDesDim = struct.unpack("3i", frmt)
        
         #if nDesDim != 128: #should be 128 in this case
            #raise RuntimeError, 'Keypoint descriptor length invalid (should be 128).' 
        
        strlocs = f.read(num*nLocDim*4)
        locs = struct.unpack(str(num*nLocDim)+"f", strlocs)
        strdescriptors = f.read(num*nDesDim)  

        descriptors = struct.unpack(str(num*nDesDim)+"B", strdescriptors)
        pos = 0

        data_loc = np.zeros((num, (nLocDim)))
        data_des = np.zeros((num, (nDesDim))).astype('uint8') 

        for point in range(num):
            data_loc[point, :] = locs[point*nLocDim:point*nLocDim+nLocDim] 
            data_des[point, :] = descriptors[point*nDesDim:point*nDesDim+nDesDim] 

    inds = indices(data_des[:, 9], lambda x: x != 45) 
    all_points = data_loc[inds, :].astype(float)
    all_des = data_des[inds, :].astype(float) 


    # Read and crop the image 
    data = cv2.imread(filename+'.png') 
    sft_h = randint(-111, 111)
    sft_w = randint(-191, 191)
    #if srt_gbl == 1:
    #sft_h = 0
    #sft_w = 0
    

    data = centeredCrop(data, sft_h, sft_w, 256) 

    #print all_points[100:150, 0:2]   
    tmp0 = all_points[:, 0] - IMZ_SZ1 - sft_w
    all_points[:, 0] = all_points[:, 1] - IMZ_SZ2 - sft_h
    all_points[:, 1] = tmp0 
    #print all_points[100:150, 0:2]  

    all_points[:, 2] = np.log(1+all_points[:, 3])  
    all_points[:, 3] = np.sin(all_points[:, 4]) 
    all_points[:, 4] = np.cos(all_points[:, 4])   
    all_points[:, 5:8] = all_points[:, 5:8]/127.5 - 1. 
    

    idd = [ii for ii, e in enumerate(range(all_points.shape[0])) if (np.abs(all_points[e, 0]) <= SZ) & (np.abs(all_points[e, 1]) <= SZ)]
    
    all_points = all_points[idd, :] 
    all_des = all_des[idd, :] 

    all_des = add_noise(all_des) 

    idxr = np.argsort(all_points[:, 0])
    all_points = all_points[idxr, :] 
    all_des = all_des[idxr, :]

    xedges = np.linspace(-SZ, SZ, num=BLOCK_SIZE1+1) 
    yedges = np.linspace(-SZ, SZ, num=BLOCK_SIZE2+1) 

    # print xedges, yedges
    hist_points_VL = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=(xedges, yedges))
    
    hist_points_L = np.cumsum(np.sum(hist_points_VL[0], axis=1)).astype(int)  
    hist_points_cum = np.zeros((BLOCK_SIZE1*BLOCK_SIZE2+1)).astype(int)   
    hist_points_cum[1:] = np.cumsum(hist_points_VL[0])[:BLOCK_SIZE1*BLOCK_SIZE2]
    hist_points =  np.reshape(hist_points_VL[0], (BLOCK_SIZE1*BLOCK_SIZE2)) 
    #print hist_points_VL[0]  
    hist_points_L = hist_points_L.astype(int)
    all_points_tmp = all_points[:hist_points_L[0] , :]

    idxc = np.argsort(all_points_tmp[:, 1])
    all_points[:hist_points_L[0], :] = all_points_tmp[idxc , :]

    all_points[:hist_points_L[0], 0] = all_points[:hist_points_L[0], 0] - xedges[0]
    all_des[:hist_points_L[0], :] = all_des[idxc , :]
    for j in range(BLOCK_SIZE2):
        all_points[hist_points_cum[j]:hist_points_cum[j+1], 1] = all_points[hist_points_cum[j]:hist_points_cum[j+1], 1] - yedges[j]

    #print hist_points_cum
    for k in range(BLOCK_SIZE1-1):
        all_points_tmp = all_points[hist_points_L[k]:hist_points_L[k+1], :]
        idxc = np.argsort(all_points_tmp[:, 1])
        all_points[hist_points_L[k]:hist_points_L[k+1], :] = all_points_tmp[idxc, :]
        all_des[hist_points_L[k]:hist_points_L[k+1], :] = all_des[hist_points_L[k]+idxc, :]  

        # print all_points[hist_points_L[k]:hist_points_L[k+1], 0], hist_points_VL[1][k]
        all_points[hist_points_L[k]:hist_points_L[k+1], 0] = all_points[hist_points_L[k]:hist_points_L[k+1], 0] - xedges[k+1]
        # print 100*all_points[hist_points_L[k]:hist_points_L[k+1], 0]
        for j in range(BLOCK_SIZE2):   
            all_points[hist_points_cum[(k+1)*BLOCK_SIZE1+j]:hist_points_cum[(k+1)*BLOCK_SIZE1+j+1], 1] = all_points[hist_points_cum[(k+1)*BLOCK_SIZE1+j]:hist_points_cum[(k+1)*BLOCK_SIZE1+j+1], 1] - yedges[j] 
    #print all_points[150:155, :2] 
    data_loc = all_points

    data_des = all_des
    all_points = np.concatenate((data_loc, np.divide(data_des.astype(float), 512.0)), axis=1)
        
    no_pts, no_ftr = all_points.shape
    idx = np.random.rand(BLOCK_SIZE1*BLOCK_SIZE2,)
    idx = np.multiply(idx, hist_points)
    #print hist_points  
    idx2 = [ii for ii, e in enumerate(idx) if e <= 10e-10]
    idx = idx.astype(int) 
    # print hist_points_cum, idx, no_pts 
    idx = hist_points_cum[:BLOCK_SIZE1*BLOCK_SIZE2] + idx
    idx3 = [ii for ii, e in enumerate(idx) if e == no_pts]
    idx[idx3] = no_pts - 1 

    #print no_pts, no_ftr, idx.shape
    points = all_points[idx, :]
    points[idx2, :] = 0.0
    points[idx3, :] = 0.0

    point_set = np.reshape(points, (BLOCK_SIZE1, BLOCK_SIZE2, all_points.shape[1]))

    #print point_set[:, :, 1] 

    #print data.shape
    #print point_set.shape 
    return data, point_set 


def get_data(dataset, dataset_train, arr, srt):
    poses = []
    camerapara = []
    images = []
    MX = 650000
    count = 0 
    with open(directory+dataset+dataset_train) as f:
        # print(directory+dataset)
        
        next(f)  # skip the 3 header lines
        next(f)
        next(f)
        for line in f:
            # print(line)
            fname,fl,rd,p0,p1,p2,p3,p4,p5,p6 = line.split()
            p0 = float(p0)
            p1 = float(p1)
            fl = float(fl)
            rd = float(rd)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)

            poses.append((p0,p1,p2,p3,p4,p5,p6))
            camerapara.append((fl,rd))
            images.append(directory+dataset+fname) 
            count = count + 1
            if count > MX:
                break
    poses_sample = [] 
    camerapara_sample =[]
    images_sample = [] 
    no_el = arr.shape[0] 
    #print arr 
    for i in range(no_el): 
        poses_sample.append(poses[arr[i]]) 
        camerapara_sample.append(camerapara[arr[i]]) 
        images_sample.append(images[arr[i]]) 
    if srt == 1:    
        images_sample = sorted(images_sample)
    srt_gbl = srt 
    pool = Pool(32) 
    images = []
    points = []
    #images, points = preprocess(images_sample[0]) 
    #print images_sample 
    for data, des in pool.map(preprocess, images_sample, chunksize=1): 
        images.append(data) 
        points.append(des)
    pool.close() 

    return images, points #, poses_sample, camerapara_sample



