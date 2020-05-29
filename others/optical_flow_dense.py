# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:28:30 2020

@author: Andrea Bassi
"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

cap = cv.VideoCapture(cv.samples.findFile("plant.avi"))

ret, frame1 = cap.read()

#print(ret)




prvs_image = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

#print ((prvs_image.shape))

#cv.imshow('Prvs',prvs)
#cv.waitKey(0)
#cv.destroyAllWindows()

hsv = np.zeros_like(frame1)
hsv[:,:,1] = 255

#print(hsv.shape)

mip = np.zeros([512,512],dtype='float')

i = 0

while(1):
    i += 1
    #if i%3!=0: continue
    
    ret, frame2 = cap.read()
    next_image = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY) #+ prvs_image
    
    #flow = cv.calcOpticalFlowFarneback(prvs_image,next_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    flow = cv.calcOpticalFlowFarneback(prvs_image,next_image, None, 0.5, 1, 45, 9, 5, 1.1, 1)
    
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    
    
    mip = np.maximum(mip,mag)
    #mip[0,0] = 255
    #print(np.amin(mip))
    
    
    print(mag.shape)
    
    #cv.imshow('Flow',mip)
    #cv.waitKey(10)
    plt.imshow (mag)
    plt.pause(0.01)
    prev_mag = mag
    prvs_image = next_image
    print(i) 
    
cv.destroyAllWindows()