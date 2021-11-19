# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:59:21 2021

@author: luigi
"""

# importing the module
import cv2
import numpy as np
import matplotlib as plt
 

global positions_list
global half_size
global img

import time



def select_point_on_click(event, x, y, flags, params):
    """
    gets the coordinates of the point and shows a rectangle around it
    """
    
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # print('x1 = ' + str(x), ' ','y1 = ' + str(y))
        print(f'Selected point: x = {x} , y = {y}')
        positions_list.append([x,y])
        
        
        startpoint = (x-half_size, y+half_size)
        endpoint =   (x+half_size, y-half_size)
        
        cv2.rectangle(img, startpoint, endpoint,
                      color = 255,
                      thickness = 3)
        
        cv2.imshow('image', img)
        cv2.waitKey(0)
    
def select_rois(img, half_size = 100):
    
    rois = []
    
    for pos in positions_list:
        x = pos[0]
        y = pos[1]
        rois.append(img[y-half_size:y+half_size,
                        x-half_size:x+half_size]
                    )
    return rois

def show_rois(rois):
    
    for roi_idx, roi in enumerate(rois):
        cv2.imshow(f'Roi {roi_idx}',roi)
    
    # cv2.waitKey(0)
       


if __name__=="__main__":
 
    # reading the image
    original_img = cv2.imread('WT_1_3_t0000_z0000_c0.tif', 0)
    img = original_img.copy()
    positions_list = []
    half_size = 100
    img_size = original_img.shape    
 
    # display the rescaled image 
    cv2.namedWindow("image", cv2.WINDOW_NORMAL) 
    rescale = 0.3
    cv2.resizeWindow('image', (int(img_size[1]*rescale), int(img_size[0]*rescale)) )
    cv2.imshow('image', img)
    
    # get the positions on the image
    cv2.setMouseCallback('image', select_point_on_click)
    cv2.waitKey(0)
    
    # select the rois
    rois = select_rois(original_img, half_size = 100)
    
    show_rois(rois)
    

    #template matching
    next_img = cv2.imread('WT_1_3_t0100_z0000_c0.tif',0)
    
    template = rois[0]
    h,w = template.shape
    print(h,w)
    
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    
    meth = 'cv2.TM_CCOEFF'
    t0 = time.time()
    #next_img = _next_img.copy()
    method = eval(meth)
    res = cv2.matchTemplate(next_img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    x_new = top_left[0] + half_size
    y_new = top_left[1] + half_size
    
    print('Found ROI center:', x_new , y_new)
    print(f'Execution time using method {meth}: {time.time()-t0: .2f} s')
  
    found_roi = next_img[y_new-half_size:y_new+half_size,
                         x_new-half_size:x_new+half_size]
    
    cv2.imshow('Found ROI', found_roi)
    
    cv2.waitKey(0)
   
   

    