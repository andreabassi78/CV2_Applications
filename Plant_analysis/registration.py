# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:47:22 2021

@author: luigi
"""
# importing the module
import cv2
import numpy as np
  
# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print('x1 = ' + str(x), ' ','y1 = ' + str(y))
 
        # displaying the coordinates
        # on the image window
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, str(x) + ',' +
        #            str(y), (x,y), font,
        #            1, (255, 0, 0), 2)
        #cv2.imshow('image', img)
        
        global posList
        posList = []
        if event == cv2.EVENT_LBUTTONDOWN:
            posList.append(x)
            posList.append(y)


# driver function
if __name__=="__main__":
 
    # reading the image
    img = cv2.imread('WT_1_3_t0000_z0000_c0.tif', 0)
 
    # displaying the image
    cv2.imshow('image', img)
 
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
    
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
 
    # close the window
    cv2.destroyAllWindows() 
    
    #adding coordinates to list
    posNp = np.array(posList)
    
#Drawing a rectangle
start_point = (posList[0] - 100, posList[1] + 100) 
end_point = (posList[0] + 100, posList[1] - 100) 
# Blue color in BGR 
color = (255, 255, 255)   
# Line thickness of 2 px 
thickness = 2 
# Using cv2.rectangle() method 
# Draw a rectangle with white line borders of thickness of 2 px 
img_1 = cv2.rectangle(img, start_point, end_point, color, thickness)
# displaying the image
cv2.imshow('image', img)
    
# wait for a key to be pressed to exit
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows() 

#crop image
y=posList[1] - 100
x=posList[0] - 100
t=200
r=200
crop = img[y:y+t, x:x+r]
    
cv2.imshow('Image', crop)
cv2.waitKey(0)

#adding the crop image to a list
global cropList
cropList = []
cropList.append(crop)

# Read the images to be aligned

im1 =  cropList[0]
im2 =  cv2.imread("WT_1_3_t0399_z0000_c0.tif");

im2 =  im2[y:y+t, x:x+r]

# Convert images to grayscale
im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
# Find size of image1
sz = im1.shape

# Define the motion model
warp_mode = cv2.MOTION_TRANSLATION

# Define 2x3 matrix and initialize the matrix to identity
warp_matrix = np.eye(2, 3, dtype=np.float32)

# Specify the number of iterations.
number_of_iterations = 5000;
 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
 
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
 
# Use warpAffine for Translation, Euclidean and Affine
im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
 
# Show final results

cv2.imshow("Image 1", im1)
cv2.imshow("Image 2", im2)
cv2.imshow("Aligned Image 2", im2_aligned)
cv2.waitKey(0)