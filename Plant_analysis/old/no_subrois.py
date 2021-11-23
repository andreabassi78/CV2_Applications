# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 12:31:56 2021

@author:
"""

import cv2
import time
import numpy as np

global positions_list
global half_size
global img


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
       
    
def align_with_template_match(image_to_align, template):
    
    h,w = template.shape
    
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    
    meth = 'cv2.TM_CCOEFF'
    t0 = time.time()
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
  
    aligned_roi = next_img[y_new-half_size:y_new+half_size,
                         x_new-half_size:x_new+half_size]
    
    return aligned_roi, template
    
    
def align_with_registration(next_rois, previous_rois, half_size):  
    
    sx,sy = previous_rois.shape
    
    # define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000;
    termination_eps = 1e-10;
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     
    # run the ECC algorithm
    (cc, warp_matrix) = cv2.findTransformECC (previous_rois,next_rois,warp_matrix, warp_mode, criteria, None, 1 )
    next_rois_aligned = cv2.warpAffine(next_rois, warp_matrix, (sy,sx), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        
    # #obtain subrois
    # original_rois = previous_rois[sy//2-half_size:sy//2+half_size,
    #                                sx//2-half_size:sx//2+half_size ]
    
    
    # aligned_rois =  next_rois_aligned[sy//2-half_size:sy//2+half_size,
    #                                sx//2-half_size:sx//2+half_size ]

    
    # return aligned_rois, original_rois
    
    print(warp_matrix)
    return next_rois_aligned

if __name__=="__main__":
 
    # reading the image
    previous_img = cv2.imread('WT_1_3_t0000_z0000_c0.tif', 0)
    next_img = cv2.imread('WT_1_3_t0100_z0000_c0.tif',0)
    
    img = previous_img.copy()
    positions_list = []
    half_size = 100
    
    img_size = previous_img.shape    
    
    # display the rescaled image 
    cv2.namedWindow("image", cv2.WINDOW_NORMAL) 
    rescale = 0.3
    cv2.resizeWindow('image', (int(img_size[1]*rescale), int(img_size[0]*rescale)) )
    cv2.imshow('image', img)
    
    # get the positions on the image
    cv2.setMouseCallback('image', select_point_on_click)
    cv2.waitKey(0)
    
    
    
    
    # # %% Template matching
    
    # # select the rois
    # rois = select_rois(previous_img, half_size)
    
    # # show_rois(rois)

    # # template matching
    # aligned, original = align_with_template_match(image_to_align = next_img,
    #                                           template = rois[0])
    
    # cv2.imshow('Aligned ROI', aligned)
    # cv2.imshow('Original ROI', original)
    # cv2.waitKey(0)
   
    # Registration
   
    # select the rois
    previous_rois = select_rois(previous_img, half_size)
    
    next_rois = select_rois(next_img, half_size)
    
    #registration
    # aligned, original = align_with_registration(next_rois[0],
    #                                             previous_rois[0],
    #                                             half_size) 
    aligned = align_with_registration(next_rois[0],
                                                previous_rois[0],
                                                half_size) 
    cv2.imshow('Aligned ROI', aligned)
    cv2.waitKey(0)