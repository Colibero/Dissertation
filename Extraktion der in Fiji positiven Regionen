# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:41:15 2020

@author: chris

This is the code used for the extraction of the regions, that were recognized in FIJI using parametric image segementation.
The script was executet for the stains on HO-1, IBA1 and HO-1 using their own directories. 

"""
import os
import csv
import numpy as np
from PIL import Image
import random
from shutil import copyfile


#directory with all the ROIs detected as positive using FIJI image segmentation
dir_roi = 'd:/D_lw/APromotion/Bilder/iba1 main/rois_nachtrag/'

#list all detectec ROI that are in the directory
list_roi_files = os.listdir(dir_roi)


#Directory with raw images
dir_pics = 'd:/D_lw/APromotion/Bilder/iba1 main/raw_nachtrag/'

#create list with all the images in the directory
list_pics = os.listdir(dir_pics)

#extract the ROI as single images containing one cell and save them in one output directory
for rois in list_roi_files:
    fname_img = rois[:-8]+ '.tif'
    print(fname_img)
    with open(dir_roi + rois) as csvdatei:
        data = csv.reader(csvdatei, delimiter = ',') 
        
        im = Image.open(dir_pics +fname_img)
        size = im.size
                
        i = 0
        for line in data:
            if i>0:
                x = int(line[4])
                y = int(line[5])
                left = x - 15
                top = y + 55
                right = x +55
                bottom = y - 15
                if left <= 0:
                    left = 0
                if top > size[1]:
                    top =size[1] 
                if bottom <= 0:
                    bottom = 0
                if right > size[0]:
                    right =size[0] 
                
                im1 = im.crop((left, bottom, right, top, ))
                
                output = ('d:/D_lw/APromotion/Bilder/iba1 main/cropped_nachtrag/' +fname_img[:-4]+ str(i) + '.tif')
                print(output)
                im1.save(output)
                
            i += 1
            
#correct filenames
dir_cropped = 'd:/D_lw/APromotion/Bilder/iba1 main/cropped_nachtrag/'

list_cropped = os.listdir(dir_cropped)

for fname in list_cropped:
    fname2 = fname.lower()
    if fname2[5:18] != 'pgl iba1 musk':
        src = 'd:/D_lw/APromotion/Bilder/iba1 main/cropped_nachtrag/' + fname
        trg = 'd:/D_lw/APromotion/Bilder/iba1 main/cropped_nopglmuco_nachtrag/'+ fname
        copyfile(src, trg)

