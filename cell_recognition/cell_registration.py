# Import numpy and OpenCV
import numpy as np
import cv2
import time
from PIL import Image
from pystackreg import StackReg
from skimage import io
import matplotlib.pyplot as plt


#load reference and "moved" image
#ref = io.imread('some_original_image.tif')
#mov = io.imread('some_changed_image.tif')
# The larger the more stable the video, but less reactive to sudden panning

path = 'C:\\Users\\Andrea Bassi\\Documents\\Data\\PROCHIP\\Throughput_video\\registration_test\\'
    #filename = 'dual_color_stack'
filename = 'roi_small'   
    
im = Image.open(path+filename+'.tif')

img = np.array(im)    
h,w = np.shape(img)

tiffarray = np.zeros((h,w,im.n_frames))            

# Set frame count
n_frames = im.n_frames

im.seek(0)

img = np.array(im)
#img = img/np.amax(img)*255
prev = (img).astype('uint8')

sr = StackReg(StackReg.RIGID_BODY)


for i in range(n_frames-1):
  # Detect feature points in previous frame
  
  im.seek(i)
  img = np.array(im)
  #img = im#/np.amax(im)*255
  curr = (img).astype('uint8') 
  
  t0=time.time()
  #Translational transformation
  
  sr.register(prev, curr)
   
  out = sr.transform(curr)
  out = out.astype('uint8') 
  print(np.amax(curr))
  print(np.amax(out))
  
  
  #frame_out = cv2.hconcat([curr, out])
  #out = out/np.amax(out)*255
  #print(np.amax(out))
  #cv2.imshow("Before", curr)
  cv2.imshow("After", out)
  
  cv2.waitKey(200)
  
  prev = out
#  imgplot = plt.imshow(out)
#  plt.show()

  
#cv2.destroyAllWindows()
