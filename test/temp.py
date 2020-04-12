from pystackreg import StackReg
from skimage import io
import cv2
import matplotlib.pyplot as plt

#load reference and "moved" image

path = 'C:\\Users\\Andrea Bassi\\Documents\\Data\\PROCHIP\\Throughput_video\\registration_test\\'
    #filename = 'dual_color_stack'
filename = 'PSF'   
    
ref = io.imread(path+'im0.tif')
mov = io.imread(path+'im1.tif')


#Translational transformation
sr = StackReg(StackReg.RIGID_BODY)
out_tra = sr.register_transform(ref, mov)

imgplot = plt.imshow(out_tra)
plt.show()

#import cv2

#cv2.imshow("Out", out_tra/256)
  
#cv2.waitKey(0)