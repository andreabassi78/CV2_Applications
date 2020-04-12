import cv2 
from PIL import Image
import numpy as np 

path = 'C:\\Users\\Andrea Bassi\\Documents\\Data\\PROCHIP\\Throughput_video\\ImagesToRegister\\'
  
im1 = Image.open(path + 'image1.tif')
im1.seek(0)

im2 = Image.open(path + 'image2.tif')
im2.seek(0)

img1 = np.array(im1)
img2 = np.array(im2)


img1 = (img1/256).astype('uint8')
img2 = (img2/256).astype('uint8')

print(np.shape(img2))

s = np.shape(img2)
#img1_color = cv2.imread('im1.jpg')  # Image to be aligned. 
#img2_color = cv2.imread('im2.jpg')    # Reference image. 

 
# Convert to grayscale. 
#img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
#img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 

#cv2.imshow('im1', img1)           
#cv2.waitKey(0)

height = s[0]
width = s[1]

print (s)

  
# Create ORB detector with 5000 features. 
orb_detector = cv2.ORB_create(5000) 
  
# Find keypoints and descriptors. 
# The first arg is the image, second arg is the mask 
#  (which is not reqiured in this case). 
kp1, d1 = orb_detector.detectAndCompute(img1, None) 
kp2, d2 = orb_detector.detectAndCompute(img2, None) 
  
# Match features between the two images. 
# We create a Brute Force matcher with  
# Hamming distance as measurement mode. 
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
  
# Match the two sets of descriptors. 
matches = matcher.match(d1, d2) 
  
# Sort matches on the basis of their Hamming distance. 
matches.sort(key = lambda x: x.distance) 
  
# Take the top 90 % matches forward. 
matches = matches[:int(len(matches)*90)] 
no_of_matches = len(matches) 
  
# Define empty matrices of shape no_of_matches * 2. 
p1 = np.zeros((no_of_matches, 2)) 
p2 = np.zeros((no_of_matches, 2)) 
  
for i in range(len(matches)): 
  p1[i, :] = kp1[matches[i].queryIdx].pt 
  p2[i, :] = kp2[matches[i].trainIdx].pt 
  
# Find the homography matrix. 
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
  
# Use this matrix to transform the 
# colored image wrt the reference image. 
transformed_img = cv2.warpPerspective(img1, 
                    homography, (width, height))

#show the output
cv2.imshow('output', transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output. 
#cv2.imwrite('output.jpg', transformed_img) 