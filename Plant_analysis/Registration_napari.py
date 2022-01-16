# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:35:46 2021

@author: Andrea Bassi, Giorgia Tortora @Polimi
"""

import numpy as np
from Registration import plot_data, select_rois, align_with_registration
from Registration import update_position, correct_decay, calculate_spectrum
from magicgui import magicgui
import napari
from napari.layers import Image, Points
import pathlib
import cv2
import os
import pandas as pd


def normalize_stack(stack, **kwargs):
    '''
    -normalizes n-dimensional stack it to its maximum and minimum values,
    unless normalization values are provided in kwargs,
    -casts the image to 8 bit for fast processing with cv2
    '''
    img = np.float32(stack)
    if 'vmin' in kwargs:
        vmin = kwargs['vmin']
    else:    
        vmin = np.amin(img)
   
    if 'vmax' in kwargs:
        vmax = kwargs['vmax']
    else:    
        vmax = np.amax(img)
    saturation = 1   
    img = saturation * (img-vmin) / (vmax-vmin)
    img = (img*255).astype('uint8') 
    return img, vmin, vmax


def calculate_intensity(image, roi_num, points, roi_size):
    """
    Calculates the mean intensity,
    of Roi designed around each point in points
    """
    stack = image.data
    locations = points.data
    st, _sy, _sx = stack.shape
    subroi_halfsize = roi_size//2
    intensities = np.zeros([st,roi_num])
    
    for time_idx in range(st):
        
        img = stack[time_idx,...]
    
        for roi_idx in range(roi_num):
            global_index = roi_idx+(time_idx*roi_num)
            y = int(locations[global_index,1])
            x = int(locations[global_index,2])
            intensity = np.mean(img[y-subroi_halfsize:y+subroi_halfsize,
                                    x-subroi_halfsize:x+subroi_halfsize])
            intensities[time_idx, roi_idx] = intensity
    return intensities


def measure_displacement(image, roi_num, points):
    """
    Measure the displacement of each roi:
    dr: relative to its position in the previous time frame 
    deltar: relative to the initial position.
    """
    stack = image.data
    locations = points.data
    st, sy, sx = stack.shape
    
    reshaped = locations.reshape((st, roi_num, stack.ndim))
    reshaped = reshaped[...,1:] # remove the time index value
    
    reshaped0 = reshaped[0,...] # take the y,x cohordinates of the rois in the first time frame
    deltar = np.sqrt( np.sum( (reshaped-reshaped0)**2, axis=2) )
    rolled = np.roll(reshaped, 1, axis=0) #roll one roi    
    rolled[0,...] = reshaped0
    dxy = reshaped-rolled
    print('reshaped:\n',reshaped)
    print('rolled:\n',rolled)
    print('dxy\n',dxy)
    
    dr = np.sqrt( np.sum( (dxy)**2, axis=2) )
    
    return deltar, dr

    
def save_in_excel (filename_xls, sheets_number, **kwargs):
    """
    Creates or overwrite an an excel file with a number of sheets.
    The data to save (in columns) are passed as kwargs as 2D lists or 2D numpy array
    """    
    writer = pd.ExcelWriter(filename_xls)
        
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

    
@magicgui(call_button="Register ROIs",
          roi_size = {'min':2},
          time_series_undersampling = {'min':1})
def register_images(image: Image, initial_points: Points, 
                    normalize: bool = True,
                    roi_size: int = 100,  
                    median_filter_size:int = 3, 
                    time_series_undersampling:int =1
                    )-> Points:
    
    assert hasattr(image, 'data'), 'No image stack defined'
    assert hasattr(initial_points, 'data'), 'No registration ROIs defined'
    assert len(initial_points.data) > 1, 'At least one registration ROI must be chosen'
    for point in initial_points.data:
        assert point[0] == 0.0, 'Registration ROIs must be selected in the first time frame of the image stack.'

    roi_num = len(initial_points.data)
    
    stack = image.data
    if normalize:
        stack, _vmin, _vmax = normalize_stack(stack)
    
    time_frames_num, sy, sx = stack.shape
    
    registered_points = np.zeros([roi_num*time_frames_num, stack.ndim])
    registered_points[0:roi_num, :] = initial_points.data
    
    initial_position_list = np.flip(initial_points.data).tolist() #TODO change position list using napari points notation
    position_list = initial_position_list
    initial_img = stack[0, ...]        
    
    for t_index in range(1, time_frames_num, time_series_undersampling):
        
        #previous_img = select_image(t_index, vmin, vmax)
        next_img = stack[t_index, ...]
        
        previous_rois = select_rois(initial_img, initial_position_list, roi_size)
        next_rois = select_rois(next_img, position_list, roi_size)
        
        # registration
        aligned, original, dx, dy = align_with_registration(next_rois,
                                                            previous_rois,
                                                            median_filter_size,
                                                            roi_size)
    
        position_list, _length = update_position(position_list, initial_position_list, dx, dy)
        
        #show_rois(aligned, 'Aligned', rect_fraction=3, zoom=4) # TODO: substitute witn napari visualization
        
        next_points_data = np.flip(np.array(position_list))
        registered_points[roi_num*t_index:roi_num*t_index+roi_num, 1:] = next_points_data
        registered_points[roi_num*t_index:roi_num*t_index+roi_num, 0] = t_index
    
    return Points(registered_points, size=sx//30, face_color='red',  name='registered points')    


@magicgui(call_button="Process registered ROIs")
def process_rois(image: Image, 
                 registered_points: Points,
                 correct_photobleaching: bool,
                 subroi_size:int = 50,
                 plot_results:bool = True,
                 save_results:bool = False,
                 path = pathlib.Path(os.getcwd()+"\\test.xlsx"),
                 ):
    
    assert hasattr(image, 'data'), 'No image stack defined'
    time_frames_num, sy, sx = image.data.shape
    assert hasattr(registered_points, 'data'), 'No registration ROIs defined'
    locations = registered_points.data
    assert len(locations) > time_frames_num, 'Please select a layer of registered ROIs'
    assert len(locations) % time_frames_num == 0, 'Check that roi_num is int, report if you get assertion error'
    roi_num = len(locations) // time_frames_num
    
    intensities = calculate_intensity(image, roi_num, registered_points, subroi_size)
    deltar, dr = measure_displacement(image, roi_num, registered_points)
    spectra = calculate_spectrum(intensities)
    
    if correct_photobleaching:
        intensities = correct_decay(intensities)
        
    if plot_results:   
        plot_data(deltar, "time index", "lenght (px)")
        plot_data(intensities, "time index", "mean intensity")
        plot_data(spectra, "frequency index", "power spectrum", plot_type = 'log')
       
    if save_results:
        save_in_excel(filename_xls = path, 
                      sheets_number = roi_num,
                      dr = dr,
                      length = deltar,
                      intensity = intensities,
                      )    

  
if __name__ == '__main__':
    
    viewer = napari.Viewer()
    
    folder = os.getcwd()+"\\images"
    
    viewer.open(folder)
    
    points = np.array([[0,1076, 829], [0,1378, 636]])
    points_layer = viewer.add_points(
        points,
        size=50,
        name= 'selected points'
    )
        
    
    viewer.window.add_dock_widget(register_images, name = 'Registration', area='right',add_vertical_stretch=True)
    viewer.window.add_dock_widget(process_rois, name = 'Processing', area='right')
    napari.run() 
    