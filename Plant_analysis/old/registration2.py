# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:35:46 2021

@author: Andrea Bassi, Giorgia Tortora @Polimi
"""

import cv2
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import filters


def select_points(input_img, half_size):
    """
    opens the input image and allows the user to select points 
    returns a list with the cohordinates of the selected points
    makes use of the nested function select_point_on_click    
    """ 
    positions_list = []
    RESCALE = 0.3 # scaling factor for the image to show
    
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

     
    img = input_img.copy()
    img_size = img.shape    
    # display the rescaled image 
    cv2.namedWindow("image", cv2.WINDOW_NORMAL) 
    
    cv2.resizeWindow('image', (int(img_size[1]*RESCALE), int(img_size[0]*RESCALE)) )
    cv2.imshow('image', img)
    
    # get the positions on the image
    cv2.setMouseCallback('image', select_point_on_click)
    cv2.waitKey(0)
    
    return positions_list
 

def open_and_filter_image(path, sigma = 2):
    '''
    opens an image in 8 bit and filters it with a gaussian blur
    '''
    img = cv2.imread(path, -1)
    img = np.float32(img)
    img = filters.gaussian(img, sigma)
    img_max = np.amax(img)
    img = (img/256).astype('uint8') 
    return img

    
def select_rois(input_image, positions, half_size):
    
    rois = []
    
    for pos in positions:
        x = pos[0]
        y = pos[1]
        rois.append(input_image[y-half_size:y+half_size,
                              x-half_size:x+half_size])
    return rois


def show_rois(rois):
    
    for roi_idx, roi in enumerate(rois):
        cv2.imshow(f'Roi {roi_idx}',roi)
    # cv2.waitKey(0)
       
    
def align_with_template_match(next_image, templates):
    
    aligned_rois = []
    updated_positions_list = []
    
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    meth = 'cv2.TM_CCOEFF'
    method = eval(meth)    
    
    for template in templates: 
    
        h,w = template.shape
        
        
        res = cv2.matchTemplate(next_image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        
        x_new = top_left[0] + h//2
        y_new = top_left[1] + w//2
         
    
        aligned_roi = next_image[y_new - w//2: y_new + w//2,
                                 x_new - h//2: x_new + h//2]
        
        aligned_rois.append(aligned_roi)
        # print('Found ROI center:', x_new , y_new)
    
        updated_positions_list.append([x_new,y_new])
        
    return aligned_rois, templates, updated_positions_list
    
def align_with_registration(next_rois, previous_rois, previous_positions, half_size):  
    
    original_rois = []
    aligned_rois = []
    updated_positions_list = []

    warp_mode = cv2.MOTION_TRANSLATION
    number_of_iterations = 2000
    termination_eps = 1e-6
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    number_of_iterations,  termination_eps)
    
    for previous_roi, next_roi, previous_pos in zip(previous_rois, next_rois, previous_positions):
    
        x = previous_pos[0] 
        y = previous_pos[1] 
    
        sx,sy = previous_roi.shape
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)

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
        
        updated_positions_list.append([x_new,y_new])
    
    return aligned_rois, original_rois, updated_positions_list

def find_displacements(new_positions, old_positions):
    """
    function determine the displacements of the rois 
    """
    
    distances_x = []
    distances_y = []
    for new_pos, old_pos in zip(new_positions, old_positions):
        distances_x.append( new_pos[0]-old_pos[0] )
        distances_y.append( new_pos[1]-old_pos[1] )
        
    return distances_x,distances_y

def calculate_mean_intensity(imgs):
    """
    function to calculate the mean intensity in the ROIs
    """
    mean_intensities = []
    sx,sy = imgs[0].shape 
    
    for img in imgs:
        # _ret, thresh_pre = cv2.threshold(img,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
        # mean_intensity = np.mean(thresh_pre[thresh_pre>0])
        #cv2.imshow('segmented', thresh_pre)
        #cv2.waitKey(10)
        
        subroi_size = sx//4
        
        mean_intensity = np.mean(img[sx//2-subroi_size:sx//2+subroi_size,
                                     sy//2-subroi_size:sy//2+subroi_size])
        mean_intensities.append(mean_intensity)
        
        
    return mean_intensities


if __name__=="__main__":
    
    HALF_SIZE = 80
    mode = 1 # 0: 'template_matching', 1: 'registration'
    sigma = 1 # of the gaussian blur applied to the images
 
    #folder = os.getcwd()
    folder = 'D:\\DATA\\SPIM\\211014\\aca2-2_2'
    # file_names = os.listdir(folder)

    try:     
         
        file_names = [ x for x in os.listdir(folder) if 'c0.tif' in x ]
        
        first_img = cv2.imread(folder + '\\' + file_names[0])
        
        initial_list = select_points(first_img, HALF_SIZE) # list of the selected points
        previous_list = initial_list
        
        time_frames = len(file_names)-1
        
        displacements_x = []
        displacements_y = []
        intensities = []
        
        previous_img = open_and_filter_image(folder + '\\' + file_names[0],sigma)
    
        for t_index in range(0, time_frames, 10): # TODO leave time_frame only
            
            print('processing image:', t_index )
            # previous_img = open_and_filter_image(folder + '\\' + file_names[t_index], sigma)
            next_img = open_and_filter_image(folder + '\\' + file_names[t_index+1], sigma)
             
            # Template matching
            if mode == 0:
                # select the rois
                rois = select_rois(previous_img, initial_list, HALF_SIZE)
                # template matching
                aligned, original, next_list = align_with_template_match(next_img, rois)              
                 
            # Registration
            if mode == 1:
                # select the rois
                previous_rois = select_rois(previous_img, initial_list, int(HALF_SIZE*1.2))
                next_rois = select_rois(next_img, previous_list, int(HALF_SIZE*1.2))
                
                # registration
                aligned, original, next_list = align_with_registration(next_rois,
                                                                       previous_rois,
                                                                       previous_list,
                                                                       HALF_SIZE)
            
            dx, dy = find_displacements(next_list, previous_list)
            displacements_x.append(dx) 
            displacements_y.append(dy)
                    
            intensity = calculate_mean_intensity(aligned)    
            intensities.append(intensity)
            
            previous_list = next_list    
                      
            roi_to_show = 0
            cv2.imshow('Aligned ROI', aligned[roi_to_show])
            #cv2.imshow('Original ROI', original[roi_to_show])
            cv2.waitKey(10)
         
        
        #calculate fft power spectrum
        intensities_array = np.array(intensities) 
        ft = np.fft.fftshift(np.fft.fft(intensities_array, axis = 0))
        intensities_ft = (np.abs(ft))**2   
    
        plt.plot(displacements_x)
        plt.xlabel("time index") 
        plt.ylabel("displacement (px)")
        plt.show()
        
        plt.plot(intensities)
        plt.xlabel("time index")
        plt.ylabel("mean intensity")
        plt.show()
        
        plt.plot(np.log10(intensities_ft) + 1e-8)
        plt.xlabel("frequency index")
        plt.ylabel("power spectrum")
        plt.show()              
                
    finally:    
            
        cv2.destroyAllWindows()