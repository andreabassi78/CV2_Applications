'''
Created on 6 apr 2020

@author: Andrea Bassi (Politecnico di Milano)
'''

import cv2
import time
from PIL import Image
import numpy as np


def create_contours(cx,cy, img8bit, cnt, rect_size):        
    """ Input: 
    img8bit: monochrome image, previously converted to 8bit
    cx,cy: list of the coordinates of the centroids  
    cnt: list of the contours.
    rect_size: side of the square to be displayed/extracted  
        Output:
    img: RGB image with annotations
    roi: list of the extracted ROIs  
    
    Note: ROIs are not registered and this might be a problem if one wants to save the stack directly  
    """  
    
    l = img8bit.shape
    roi = []
    img = cv2.cvtColor(img8bit,cv2.COLOR_GRAY2RGB)      
    
    for indx, val in enumerate(cx):
        
    #x,y,w,h = cv2.boundingRect(cnt)
        x = int(cx[indx] - rect_size/2) 
        y = int(cy[indx] - rect_size/2)
     
        w = h = rect_size 
        
        img = cv2.drawContours(img, [cnt[indx]], 0, (0,256,0), 2) 
        
        if indx == 0:
            color = (0,0,256)
        else: 
            color = (256,0,128)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,1)    
        
        if x>0 and y>0 and x+w<l[1]-1 and y+h<l[0]-1:
            detail = img8bit [y:y+h, x:x+w]
            roi.append(detail)
       
    return img, roi


def find_gut(img8bit, gut_size): 
    """ 
        Input:
    img8bit: monochrome image, previously converted to 8bit (img8bit)
    cell_size: minimum area of the object to be detected.
        Output:
    cx,cy : list of the coordinates of the centroids of the detected objects 
    selected_contours: list of contours of the detected object (no child contours are detected).  
    """               
    _ret,thresh_pre = cv2.threshold(img8bit,0,255,cv2.THRESH_OTSU)
    # ret is the threshold that was used, thresh is the thresholded image.        
    kernel  = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh_pre,cv2.MORPH_CLOSE, kernel, iterations = 2)
    # morphological opening
    contours, _hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cx = []
    cy = []            
    area = []
    selected_contours =[]
    #print(len(contours))
    
    for cnt in contours:
    #   print(len(cnt))           
        M = cv2.moments(cnt)
        if M['m00'] >  gut_size:   # (M['m00'] gives the contour area, also as cv2.contourArea(cnt)     
            #extracts image center
            cx.append(int(M['m10']/M['m00']))
            cy.append(int(M['m01']/M['m00']))
            area.append(M['m00']) 
            selected_contours.append(cnt)
    im_process = thresh_pre
    return cx, cy, selected_contours, im_process

RECT_SIZE = 300 #side of ROI that are extracted
MIN_CELL_SIZE = 300*100 #cell area must be at least MIN_CELL_SIZE (px^2) to be detected as a cell
SCALING = 0.3
#path = 'C:\\Users\\Andrea Bassi\\Documents\\Data\\SPIMonCHIP_Drosophila\\Figure articolo Misure marzo 2020\\Stage 123\\'
#filename = 'dual_color_stack'
path = ''
filename = 'drosophila1'

im = Image.open(path+filename+'.tif')
h,w = np.shape(im)
tiffarray = np.zeros((h,w,im.n_frames))
shown_rois = 0
print(im.size)

try:
    
    for i in range(im.n_frames):  
        print(i)
        im.seek(i)
        im_in = np.array(im)
        im_in = (im_in/256).astype('uint8') 
        t0 = time.time()
        cx, cy, cnts, im_p = find_gut(im_in, MIN_CELL_SIZE)
        #print('x positions of the centroids', cx)
        print('elapsed time: ', time.time()-t0)      
        #im_out = im_in
        im_out, rois = create_contours(cx, cy, im_in, cnts, RECT_SIZE)
                
        im_v = im_p
        
        dim = ( int(SCALING*im_v.shape[1]) , int(SCALING*im_v.shape[0]) )
        print(dim)
        im_resized = cv2.resize(im_v, dim, interpolation = cv2.INTER_AREA)        
                

        
        cv2.imshow('Acquired data', im_resized)
        #This would save the annotated images in the subfolder \Annotated, that must be created
        #cv2.imwrite(path+'Annotated'+'\\'+filename+'\\out\\out'+ str(i)+'.tif', im_out)
        cv2.waitKey(10) # waits the specified ms. Value 0 would stop until key is hitten    
    
finally:    
    cv2.destroyAllWindows()