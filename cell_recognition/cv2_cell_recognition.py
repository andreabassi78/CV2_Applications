'''
Created on 6 apr 2020

@author: Andrea Bassi (Politecnico di Milano)
'''

import cv2
import time
from PIL import Image
import numpy as np
from pystackreg import StackReg

def find_cell(img8bit, cell_size): 
    """ 
        Input:
    img8bit: monochrome image, previously converted to 8bit (img8bit)
    cell_size: minimum area of the object to be detected.
        Output:
    cx,cy : list of the coordinates of the centroids of the detected objects 
    selected_contours: list of contours of the detected object (no child contours are detected).  
    """               
    _ret,thresh_pre = cv2.threshold(img8bit,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret is the threshold that was used, thresh is the thresholded image.        
    kernel  = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh_pre,cv2.MORPH_OPEN, kernel, iterations = 2)
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
        if M['m00'] >  cell_size:   # (M['m00'] gives the contour area, also as cv2.contourArea(cnt)     
            #extracts image center
            cx.append(int(M['m10']/M['m00']))
            cy.append(int(M['m01']/M['m00']))
            area.append(M['m00']) 
            selected_contours.append(cnt)
        
                               
    return cx, cy, selected_contours
                
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



if __name__ == '__main__':

    RECT_SIZE = 150 #must be even
    MIN_CELL_SIZE = 40*40#cell area must be at least MIN_CELL_SIZE (px^2) to be detected as a cell
    ROI_SCALING = 2
    path = 'C:\\Users\\Andrea Bassi\\Documents\\Data\\PROCHIP\\Throughput_video\\'
    #filename = 'dual_color_stack'
    filename = 'selected_stack'
    
    
    
    im = Image.open(path+filename+'.tif')
    
    h,w = np.shape(im)
    tiffarray = np.zeros((h,w,im.n_frames))
    #cell_counter = 0
    shown_rois = 0
    mip = mip0 = np.zeros((RECT_SIZE,RECT_SIZE), dtype ='uint8') # Maximum Intensity Projection of roi[0] 
    
    sr = StackReg(StackReg.TRANSLATION)
    prev = curr = prev0 = np.zeros((RECT_SIZE,RECT_SIZE), dtype ='uint8')
    
    try:
        
        for i in range(im.n_frames-1):        
            im.seek(i)
            im_in = np.array(im)
            im_in = (im_in/256).astype('uint8') 
            t0 = time.time()
            #im_in=im_in/np.amax(im_in)*256
            cx, cy, cnts = find_cell(im_in, MIN_CELL_SIZE)
            print('x positions of the centroids', cx)
            #print('elapsed time: ', time.time()-t0)      
            
            im_out = im_in
            
            im_out, rois = create_contours(cx, cy, im_in, cnts, RECT_SIZE)
                
            cv2.imshow('Acquired data', im_out )
            
            #cv2.imwrite(path+'Annotated'+'\\'+filename+'\\out\\out'+ str(i)+'.tif', im_out)
            
            if len(rois) == 0:
                roi_resized = np.zeros((ROI_SCALING*RECT_SIZE,ROI_SCALING*RECT_SIZE), dtype ='uint8') # for saving only
                
                    
            for index, roi in enumerate(rois):
                dim = (int(ROI_SCALING*RECT_SIZE) , int(ROI_SCALING*RECT_SIZE))
                roi_rescaled = np.clip(roi.astype('float')*3, 20, 255).astype('uint8') #multiplication with 3 is to increase the contrast only
                #roi_rescaled = (np.clip(roi, 10, 255)/np.amax(roi)*255).astype('uint8')
                
                roi_resized = cv2.resize(roi_rescaled, dim, interpolation = cv2.INTER_AREA)        
                win_name = 'roi_' + str(index)
                
                cv2.imshow(win_name,roi_resized)
   
            #cv2.imwrite(path+'Annotated'+'\\'+filename+'\\roi0\\roi0'+ str(i)+'.tif', roi_resized)
                 
            num_rois = len(rois)
            
            if num_rois <= shown_rois:
                for index in range(num_rois, shown_rois):
                    if index > 0: 
                        cv2.destroyWindow('roi_' + str(index))    
            #print('number of detected ROIs: ', num_rois) 
            shown_rois = num_rois
                  
            #cell_counter +=1 
            
            if len(rois) > 0:
                t0=time.time()
                curr = rois[0]
                out = sr.register_transform(prev,curr)
                #out = out.astype('uint8') 
                #print (time.time()-t0)
                #mip = np.maximum(mip,rois[0])
                
                #print(np.amax(out))
                mip = np.maximum(mip,out)
                prev = out
             
            mip_rescaled = np.clip(mip.astype('float')*3, 20, 255).astype('uint8') #multiplication with 3 is to increase the contrast only
            #mip_rescaled = np.clip((mip/np.amax(mip)*255),20,255).astype('uint8')
            
            mip_resized = cv2.resize(mip_rescaled, dim, interpolation = cv2.INTER_AREA)    
            if num_rois < 1:
                mip = mip0
                prev = prev0   
            
            cv2.imshow('ROI_0 MIP', mip_resized)           
                
            cv2.waitKey(25)
            
        
    finally:
        
        cv2.destroyAllWindows()