# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:35:46 2021

@author: Andrea Bassi, Giorgia Tortora @Polimi
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

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


def update_position(old_pos, initial_pos, dx_list, dy_list ):
    
    new_positions = []
    new_lengths = []
    roi_idx = 0
    for pos, pos0, dx, dy in zip(old_pos, initial_pos, dx_list, dy_list, ):
        x1 = pos[0] + dx
        y1 = pos[1] + dy
        x0 = pos0[0]
        y0 = pos0[1]
        dr = np.sqrt(dx**2+dy**2)
        new_positions.append([x1,y1])
        new_lengths.append(np.sqrt((x1-x0)**2+(y1-y0)**2))
        print(f'Displacement for ROI{roi_idx}: {dr:.3f}')    
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
        subroi_halfsize = int(sx/2/fraction)    # TODO use circular subroi. Update show_rois too
        mean_intensity = np.mean(img[sx//2-subroi_halfsize:sx//2+subroi_halfsize,
                                     sy//2-subroi_halfsize:sy//2+subroi_halfsize])
        mean_intensities.append(mean_intensity)
              
    return mean_intensities


def correct_decay(data):
    '''
    corrects decay fitting data with a polynomial and subtracting it
    data are organized as a list (time) of list (roi)
    returns a 2D numpy array
    '''
    data = np.array(data) # shape is raws:time, cols:roi
    rows, cols = data.shape 
    order = 2
    corrected = np.zeros_like(data)
    for col_idx, column in enumerate(data.transpose()):
        t_data = range(rows)
        coeff = np.polyfit(t_data, column, order) 
        fit_function = np.poly1d(coeff)
        corrected_value = column - fit_function(t_data) 
        corrected[:,col_idx]= corrected_value
    
    return corrected


def calculate_spectrum(data):
    '''
    calculates power spectrum with fft
    data are organized as a list (time) of list (roi), or as a 2D numpy array
    returns a 2D numpy array
    '''
    
    data = np.array(data) # shape is raws:time, cols:roi
    ft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data), axis=0))
    spectra = (np.abs(ft))**2 
    
    return spectra
    

def plot_data(data, xlabel, ylabel, plot_type='lin'):
    '''
    data are organized as a list (time) of list (roi), or as a 2D numpy array
    '''
    data = np.array(data)
    char_size = 10
    linewidth = 0.85
    plt.rc('font', family='calibri', size=char_size)
    fig = plt.figure(figsize=(4,3), dpi=300)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_title(title, size=char_size)   
    ax.set_xlabel(xlabel, size=char_size)
    ax.set_ylabel(ylabel, size=char_size)
    if plot_type == 'log':
        # data=data+np.mean(data) # in case of log plot, the mean is added to avoid zeros
        ax.set_yscale('log')
    ax.plot(data, linewidth=linewidth)
    ax.xaxis.set_tick_params(labelsize=char_size*0.75)
    ax.yaxis.set_tick_params(labelsize=char_size*0.75)
    
    ax.grid(True, which='major',axis='both',alpha=0.2)
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)
    
    
def save_in_excel (filename_xls, sheets_number, **kwargs):
    """
    Creates or overwrite an an excel file with a number of sheets.
    The data to save (in columns) are passed as kwargs as 2D lists
    """    
    writer = pd.ExcelWriter(filename_xls +'.xlsx')
        
    for sheet_idx in range(sheets_number):  
 
        data = []
        headers = []    
        
        for key, val in kwargs.items():
            val_array = np.array(val)
            data.append(val_array[:,sheet_idx])
            headers.append(key)
            
        df = pd.DataFrame(data, index = headers).transpose()
    
        df.index.name = 't_index'
        df.to_excel(writer, f'ROI_{sheet_idx}')
    writer.save()
    

if __name__== "__main__":
    
    ROI_SIZE = 100
    FILTER_SIZE = 3 # size of the blur applied to the images 
    rect_fraction = 3
    # folder = 'D:\\DATA\\SPIM\\211014\\aca2-2_2'
    # folder = os.curdir
    # folder = 'D:\\DATA\\SPIM\\211014\\WT_1'
    # folder = 'C:\\Users\\luigi\\Desktop\\Giorgia\\WT_1'
    folder = 'C:\\Users\\andrea\\OneDrive - Politecnico di Milano\\Documenti\\PythonProjects\\CV2Apps\\Plant_analysis\\images'
    
    SAVEDATA = False

    try:     
         
        file_names = [ x for x in os.listdir(folder) if 'c0.tif' in x ]
        
        selection_img, vmin, vmax = open_image(folder + '\\' + file_names[0])
        initial_position_list = select_points(selection_img, ROI_SIZE) # list of the selected points
        position_list = initial_position_list
        initial_img,_,_ = open_image(folder + '\\' + file_names[0], vmin, vmax)        
        initial_roi = select_rois(initial_img, initial_position_list, ROI_SIZE)
        initial_intensity = calculate_mean_intensity(initial_roi, rect_fraction)  
        
        
        # initialize time depended lists 
        roi_num = len(initial_position_list)
        time_frames_num = len(file_names)-1
        intensities = [initial_intensity]
        lengths = [[0.0]*roi_num]
        positions_x = [[val[0] for val in position_list]]
        positions_y = [[val[1] for val in position_list]]
        displacements_x = [[0.0]*roi_num] # ! create a list of zeros with a number of elements equals to the ROIs number 
        displacements_y = [[0.0]*roi_num]
        
        initial_img,_,_ = open_image(folder + '\\' + file_names[0], vmin, vmax)
        
        for t_index in range(0, time_frames_num, 1):
            
            # print('processing image:', t_index )
            
            #previous_img = open_image(folder + '\\' + file_names[t_index], vmin, vmax)
            next_img,_,_ = open_image(folder + '\\' + file_names[t_index+1], vmin, vmax)
            
            previous_rois = select_rois(initial_img, initial_position_list, ROI_SIZE)
            next_rois = select_rois(next_img, position_list, ROI_SIZE)
            
            # registration
            aligned, original, dx, dy = align_with_registration(next_rois,
                                                                previous_rois,
                                                                FILTER_SIZE,
                                                                ROI_SIZE)
        
            position_list, length = update_position(position_list, initial_position_list, dx, dy)
            
            displacements_x.append(dx)
            displacements_y.append(dy)
            lengths.append(length)
            positions_x.append([val[0] for val in position_list])
            positions_y.append([val[1] for val in position_list])
            
            intensity = calculate_mean_intensity(aligned, rect_fraction)
            intensities.append(intensity)
            
            show_rois(aligned, 'Aligned', rect_fraction, zoom=4)
 
        # calculate power spectrum
        # intensities = correct_decay(intensities)
        spectra = calculate_spectrum(intensities)
        
        #save data in excel
        if SAVEDATA:
            save_in_excel (filename_xls = folder, 
                            sheets_number = len(position_list),
                            x = positions_x,
                            y = positions_y,
                            dx = displacements_x,
                            dy = displacements_y,
                            length = lengths,
                            intensity = intensities,
                            spectrum = spectra
                            )
        
        # show data with plt
        char_size = 10
        plt.rc('font', family='calibri', size=char_size)
        plot_data(lengths,
                "time index",
                "lenght px)")
        plot_data(intensities,
                "time index",
                "mean intensity")
        plot_data(spectra, 
                "frequency index",
                "power spectrum", plot_type = 'log')
        
        plt.rcParams.update(plt.rcParamsDefault)

    finally:    
            
        cv2.destroyAllWindows()