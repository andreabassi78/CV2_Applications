# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:35:46 2021

@author: luigi
"""

import cv2
import time
import numpy as np
import os
import matplotlib.pyplot as plt

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
                      color = (10, 255, 10),
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
       
    
def align_with_template_match(next_image, templates):
    
    aligned_rois = []
    
    for pos in positions_list:
        x = pos[0]
        y = pos[1]
    
    for template in templates: 
    
        h,w = template.shape
        
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        
        meth = 'cv2.TM_CCOEFF'
        # t0 = time.time()
        method = eval(meth)
        res = cv2.matchTemplate(next_image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        
        x_new = top_left[0] + half_size
        y_new = top_left[1] + half_size
        
        
        aligned_roi = next_image[y_new-half_size:y_new+half_size,
                             x_new-half_size:x_new+half_size]
        
        aligned_rois.append(aligned_roi)
        print('Found ROI center:', x_new , y_new)
        
    
        displacement = np.sqrt( (x_new - x)**2 + (y_new - y)**2 )
        # print(f'Execution time using method {meth}: {time.time()-t0: .2f} s')
    
    return aligned_rois, templates, displacement, x_new, y_new
    
    
def align_with_registration(next_rois, previous_rois, half_size):  
    
    original_rois = []
    aligned_rois = []
    
    for pos in positions_list:
        x = pos[0]
        y = pos[1]
    
    for previous_roi, next_roi in zip(previous_rois, next_rois):
    
        sx,sy = previous_roi.shape
        
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        number_of_iterations = 5000
        termination_eps = 1e-10
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
         
        # run the ECC algorithm
        _, warp_matrix = cv2.findTransformECC (previous_roi, next_roi,
                                                  warp_matrix, warp_mode, criteria)
        
        next_roi_aligned = cv2.warpAffine(next_roi, warp_matrix, (sx,sy),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
        
        original_roi = previous_roi[sy//2-half_size:sy//2+half_size,
                                        sx//2-half_size:sx//2+half_size ]
        
        
        aligned_roi =  next_roi_aligned[sy//2-half_size:sy//2+half_size,
                                        sx//2-half_size:sx//2+half_size ]
    
        original_rois.append(original_roi)
        aligned_rois.append(aligned_roi)
        
        x_new = x + int(warp_matrix[0,2])
        y_new = y + int(warp_matrix[1,2])
        
        distance = np.sqrt( warp_matrix[0,2] **2 + warp_matrix[1,2] **2 )
        print('Distance:', distance )
    
    
    return aligned_rois, original_rois, distance, x_new, y_new


def pixel_intensity_mean (aligned_rois, x_new, y_new, half_increment):
    
    areas_of_interest = []
    intensity_means = []
    
    for roi in aligned_rois:
                area_of_interest = roi [y_new - half_increment: y_new + half_increment,
                        x_new - half_increment: x_new + half_increment]
                areas_of_interest.append(area_of_interest)
            
                cv2.threshold(area_of_interest, 127, 255, cv2.THRESH_TOZERO)
                intensity_mean = np.mean(area_of_interest)
                intensity_means.append(intensity_mean)

    return areas_of_interest, intensity_means


if __name__=="__main__":
    
    half_size = 100
    half_increment = 10
    mode = 1 # 0: 'template_matching', 1: 'registration'
    t_indexes = []
    distances = []
    displacements = []
 
    #folder = os.getcwd()
    folder = 'C:\\Users\\luigi\\Desktop\\Giorgia\\WT_1'

    # file_names = os.listdir(folder)
    
    file_names = [ x for x in os.listdir(folder) if 'c0.tif' in x ]
    
    first_img = cv2.imread(folder + '\\' + file_names[0])
    
    # first_img = cv2.imread([ x for x in file_names if 't0000' in x][0])
    
    
    
    
    
    ## %% open file and select points
    
    positions_list = []
     
    img = first_img.copy()
    img_size = img.shape    
 
    # display the rescaled image 
    cv2.namedWindow("image", cv2.WINDOW_NORMAL) 
    rescale = 0.3
    cv2.resizeWindow('image', (int(img_size[1]*rescale), int(img_size[0]*rescale)) )
    cv2.imshow('image', img)
    
    # get the positions on the image
    cv2.setMouseCallback('image', select_point_on_click)
    cv2.waitKey(0)
      


    for t_index in range(len(file_names)-1):
        
        print(t_index)
        t_indexes.append(t_index)

        # reading the image
        previous_img = cv2.imread(folder + '\\' + file_names[t_index], 0)
        next_img = cv2.imread(folder + '\\' + file_names[t_index+1], 0)
            
        # # %% Template matching
        if mode == 0:
            # select the rois
            rois = select_rois(previous_img, half_size)
            # template matching
            aligned, original, displacement, x_new, y_new = align_with_template_match(next_img,
                                                      templates = rois)
            displacements.append(displacement)
        
    
        ## %% Registration
        if mode == 1:
                
            # select the rois
            previous_rois = select_rois(previous_img, int(half_size*1.5))
            next_rois = select_rois(next_img, int(half_size*1.5))
            
            # registration
            aligned, original, distance, x_new, y_new = align_with_registration(next_rois,
                                                        previous_rois,
                                                        half_size) 
            distances.append(distance)
            
        areas_of_interest, intensity_means = pixel_intensity_mean (aligned, x_new, y_new, half_increment)
            
        roi_to_show = 0
        cv2.imshow('Aligned ROI', aligned[roi_to_show])
        #cv2.imshow('Original ROI', original[roi_to_show])
        # cv2.imshow('Difference', (aligned-original)**2)
        cv2.waitKey(10)
        
        
if mode == 0:
        xs1 = t_indexes
        ys1 = displacements
        plt.plot(xs1, ys1), plt.xlabel("t_index"), plt.ylabel("Displacement")
        plt.show()
        
        xs1 = t_indexes
        ys2 = intensity_means
        plt.plot(xs1, ys2), plt.xlabel("t_index"), plt.ylabel("Intensity mean")
        plt.show()
            
else:     
        xs1 = t_indexes
        ys1 = distances
        plt.plot(xs1, ys1), plt.xlabel("t_index"), plt.ylabel("Displacement")
        plt.show()

        xs1 = t_indexes
        ys2 = intensity_means
        plt.plot(xs1, ys2), plt.xlabel("t_index"), plt.ylabel("Intensity mean")
        plt.show()

cv2.destroyAllWindows()