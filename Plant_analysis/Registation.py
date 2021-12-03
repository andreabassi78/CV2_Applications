# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:35:46 2021

@author: Andrea Bassi, Giorgia Tortora @Polimi
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def select_points(input_img, size, rescale = 0.3):
    """
    opens the input image and allows the user to select points 
    returns a list with the cohordinates of the selected points
    makes use of the nested function select_point_on_click
    - size: side of the selected rectangle
    - rescale: scaling factor applied to the shown image
        
    """ 
    positions_list = []
    half_size = size//2
    
    def select_point_on_click(event, x, y, flags, params):
        """
        gets the coordinates of the point and shows a rectangle around it
        """     
        if event == cv2.EVENT_LBUTTONDOWN:

            print(f'Selected point: x = {x} , y = {y}')
            positions_list.append([x,y])
            
            startpoint = (x-half_size, y+half_size)
            endpoint =   (x+half_size, y-half_size)
            
            cv2.rectangle(img, startpoint, endpoint,
                          color = 255,
                          thickness = 3)
            
            cv2.imshow('image', img)
            cv2.waitKey(0)
 
    img = input_img.copy()
    img_size = img.shape    
    # display the rescaled image 
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)     
    cv2.resizeWindow('image', (int(img_size[1]*rescale), int(img_size[0]*rescale)) )
    cv2.imshow('image', img)
    
    # get the positions on the image
    cv2.setMouseCallback('image', select_point_on_click)
    cv2.waitKey(0)
    
    return positions_list
 

def open_image(path, *args):
    '''
    -opens an image, 
    -normalizes it to its maximum and minimum values,
    unless normalization values are provided in args,
    -casts the image in 8 bit
    '''
    img = cv2.imread(path, -1)
    img = np.float32(img)
    if len(args) == 0:
        vmin = np.amin(img)
        vmax = np.amax(img)
    else:
        vmin = args[0]
        vmax = args[1]
    contrast = 1   
    img = contrast * (img-vmin) / (vmax-vmin)
    img = (img*255).astype('uint8')
      
    return img, vmin, vmax


def filter_image(img, sigma):
    if sigma >0:
        sigma = (sigma//2)*2+1 # sigma must be odd in cv2
        #filtered = cv2.GaussianBlur(img,(sigma,sigma),cv2.BORDER_DEFAULT)
        #_ret, img = cv2.threshold(img,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)  
        #_ret, img = cv2.threshold(img,31,255,cv2.THRESH_TOZERO)  
        filtered = cv2.medianBlur(img,sigma)
        return filtered
    else:
        return img
        
    
def select_rois(input_image, positions, roi_size):
    
    rois = []
    half_size = roi_size//2
    for pos in positions:
        x = int(pos[0])
        y = int(pos[1])
        rois.append(input_image[y-half_size:y+half_size,
                                x-half_size:x+half_size])
    return rois


def show_rois(rois, title, rect_fraction, zoom=3):
    
    for roi_idx, roi in enumerate(rois):
        sx,_ = roi.shape
        s = sx//2
        delta= int(s/rect_fraction)
        startpoint = (s-delta, s+delta)
        endpoint =   (s+delta, s-delta)
        #_ret, roi = cv2.threshold(roi,33,255,cv2.THRESH_TOZERO) # TODO test different cases
        cv2.rectangle(roi, startpoint, endpoint,
                  color = 255,
                  thickness = 1)
        roi = cv2.resize(roi, [zoom*sx,zoom*sx])
        roi = cv2.applyColorMap(roi, cv2.COLORMAP_PINK   )
        cv2.imshow(f'{title} ROI{roi_idx}', roi )
        cv2.waitKey(3)
         
    
def align_with_registration(next_rois, previous_rois, filter_size, roi_size):  
    
    original_rois = []
    aligned_rois = []
    dx_list = []
    dy_list = []
    
    half_size = roi_size//2
    
    warp_mode = cv2.MOTION_TRANSLATION 
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    number_of_iterations,  termination_eps)
    
    for previous_roi, next_roi in zip(previous_rois, next_rois):
      
        previous_roi = filter_image(previous_roi, filter_size)
        next_roi = filter_image(next_roi, filter_size)
        
        sx,sy = previous_roi.shape
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        try:
            _, warp_matrix = cv2.findTransformECC (previous_roi, next_roi,
                                                      warp_matrix, warp_mode, criteria)
            
            next_roi_aligned = cv2.warpAffine(next_roi, warp_matrix, (sx,sy),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except:
            next_roi_aligned = next_roi
        
        original_roi = previous_roi[sy//2-half_size:sy//2+half_size,
                                        sx//2-half_size:sx//2+half_size ]
        
        aligned_roi =  next_roi_aligned[sy//2-half_size:sy//2+half_size,
                                        sx//2-half_size:sx//2+half_size ]
    
        original_rois.append(original_roi)
        aligned_rois.append(aligned_roi)
        
        dx = warp_matrix[0,2]
        dy = warp_matrix[1,2]
        
        dx_list.append(dx)
        dy_list.append(dy)
    
    return aligned_rois, original_rois, dx_list, dy_list


def update_position(old_positions, initial_lengths, dx_list, dy_list ):
    
    new_positions = []
    new_lengths = []
    roi_idx = 0
    for pos, dx, dy, init_length in zip(old_positions, dx_list, dy_list, initial_lengths):
        x = pos[0] + dx
        y = pos[1] + dy
        disp = np.sqrt(dx**2+dy**2)
        new_positions.append([x,y])
        new_lengths.append(np.sqrt(x**2+y**2)-init_length)
        print(f'Displacement for ROI{roi_idx}:', disp)    
        roi_idx +=1
        
    return new_positions, new_lengths


def calculate_mean_intensity(imgs, fraction=2):
    """
    function to calculate the mean intensity in the ROIs.
    the intensity is the mean calculated in a subROI with a side
    calculated as the original side divided by fraction   
    """
    mean_intensities = []
    sx,sy = imgs[0].shape 
    
    for img in imgs:
        # _ret, img = cv2.threshold(img,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
        # mean_intensity = np.mean(thresh_pre[thresh_pre>0])
        #cv2.imshow('segmented', thresh_pre)
        #cv2.waitKey(10)       
        subroi_halfsize = int(sx/2/fraction)    # TODO use circular subroi
        mean_intensity = np.mean(img[sx//2-subroi_halfsize:sx//2+subroi_halfsize,
                                     sy//2-subroi_halfsize:sy//2+subroi_halfsize])
        mean_intensities.append(mean_intensity)
              
    return mean_intensities


def plot_data(data, xlabel, ylabel, plot_type='lin'):
    char_size = 10
    linewidth = 0.85
    fig = plt.figure(figsize=(4,3), dpi=300)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_title(title, size=char_size)   
    ax.set_xlabel(xlabel, size=char_size)
    ax.set_ylabel(ylabel, size=char_size)
    ax.plot(data, linewidth=linewidth)
    ax.xaxis.set_tick_params(labelsize=char_size*0.75)
    ax.yaxis.set_tick_params(labelsize=char_size*0.75)
    if plot_type == 'log':
        ax.set_yscale('log')
    ax.grid(True, which='major',axis='both',alpha=0.2)


if __name__== "__main__":
    
    ROI_SIZE = 100
    FILTER_SIZE = 3 # size of the blur applied to the images 

    folder = 'D:\\DATA\\SPIM\\211014\\aca2-2_2'
    # folder = os.curdir
    #folder = 'D:\\DATA\\SPIM\\211014\\WT_1'

    try:     
         
        file_names = [ x for x in os.listdir(folder) if 'c0.tif' in x ]
        
        selection_img, vmin, vmax = open_image(folder + '\\' + file_names[0])
        initial_positions_list = select_points(selection_img, ROI_SIZE) # list of the selected points
        positions_list = initial_positions_list
        initial_length = [np.sqrt(x**2+y**2) for [x,y] in positions_list] #TODO use the real length
        
        time_frames = len(file_names)-1
        
        displacements_x = []
        displacements_y = []
        intensities = []
        lengths = []
        
        initial_img,_,_ = open_image(folder + '\\' + file_names[0], vmin, vmax)
        
        for t_index in range(0, time_frames, 1): # TODO leave time_frame only
            
            # print('processing image:', t_index )
            
            #previous_img = open_image(folder + '\\' + file_names[t_index], vmin, vmax)
            next_img,_,_ = open_image(folder + '\\' + file_names[t_index+1], vmin, vmax)
            
            previous_rois = select_rois(initial_img, initial_positions_list, ROI_SIZE)
            next_rois = select_rois(next_img, positions_list, ROI_SIZE)
            
            # registration
            aligned, original, dx, dy = align_with_registration(next_rois,
                                                                previous_rois,
                                                                FILTER_SIZE,
                                                                ROI_SIZE)
        
            positions_list, length = update_position(positions_list, initial_length, dx, dy)
            rect_fraction = 3
            intensity = calculate_mean_intensity(aligned, rect_fraction)    
            
            displacements_x.append(dx)
            displacements_y.append(dy)
            lengths.append(length)
            intensities.append(intensity)
            
            show_rois(aligned, 'Aligned', rect_fraction, zoom=4)
 
        
        lengths_array =np.array(lengths) 
        intensities_array = np.array(intensities) 
        
        #calculate power spectrum
        ft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(intensities_array), axis=0))
        spectra_array = (np.abs(ft))**2   
        
        
        # show data with plt
        char_size = 10
        plt.rc('font', family='calibri', size=char_size)
        plot_data(lengths_array,
                "time index",
                "displacement (px)")
        plot_data(intensities_array,
                "time index",
                "mean intensity")
        plot_data(spectra_array, 
                "frequency index",
                "power spectrum", plot_type = 'log')
        plt.rcParams.update(plt.rcParamsDefault)

    finally:    
            
        cv2.destroyAllWindows()