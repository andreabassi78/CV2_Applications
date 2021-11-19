'''
Created on 6 apr 2020

@author: Andrea Bassi (Politecnico di Milano)
'''

import cv2
import time
from PIL import Image
import numpy as np



def create_contours(img8bit, cnt, area):        
    """ Parameters: 
    img8bit: monochrome image, previously converted to 8bit
    cnt: list of the contours.
        Return:
    img: RGB image with contours
    """  
    real_area = 0
    l = img8bit.shape
    roi = []
    img = cv2.cvtColor(img8bit,cv2.COLOR_GRAY2RGB)

    if len(area)>1:
        print(22222)
      
    for indx, val in enumerate(area): 
        img = cv2.drawContours(img, [cnt[indx]], 0, (0,255,0), 2) 
        real_area += val
    return img, real_area
        


def find_contour(img8bit, min_size): 
    """ 
        Parameters:
    img8bit: monochrome image, previously converted to 8bit (img8bit)
    min_size: minimum area of the object
        Return:
    area: list of the detected object'area (no child contours are detected)        
    contour: list of contours of the detected object (no child contours are detected).  
    """ 
    ret, thresh_pre = cv2.threshold(img8bit,0,255,cv2.THRESH_BINARY )
    kernel  = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh_pre,cv2.MORPH_OPEN, kernel, iterations = 2)
    
    contours, _hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour =[]
    area = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] >  min_size:
            area.append(M['m00']) # (M['m00'] gives the contour area, also as cv2.contourArea(cnt)     
            contour.append(cnt)
   
    
    
    ret,thresh_gut = cv2.threshold(img8bit,0,255,cv2.THRESH_OTSU)
    return area, contour, img8bit


MIN_STRUCT_SIZE = 50*50 # the area must be at least MIN_STRUCT_SIZE (px^2) to be detected 
SCALING = 1

filename = 'drosophila3.tif'

im = Image.open(filename)

try:
    volume = 0
    for i in range(im.n_frames):
        im.seek(i)
        im_in = np.array(im)
        im_in = (im_in).astype('uint8') 
        t0 = time.time()
        areas, cnts, im_p = find_contour(im_in, MIN_STRUCT_SIZE)
        im_out, area = create_contours(im_p, cnts, areas)
        volume +=area
        dim = ( int(SCALING*im_out.shape[1]) , int(SCALING*im_out.shape[0]) )
        im_resized = cv2.resize(im_out, dim, interpolation = cv2.INTER_AREA)        
        cv2.imshow('Acquired data', im_resized)
        cv2.waitKey(10) # waits the specified ms. Value 0 would stop until key is hitten    
    
    
    pixel_width = 0.65*1e-3 #mm
    pixel_height = 0.65*1e-3 #mm
    voxel_depth = 4*1e-3 #mm
    print("Measured volume (mm^3):", volume*pixel_width*pixel_height*voxel_depth)
finally:    
    cv2.destroyAllWindows()