import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
  
image = cv2.imread('thresh4.png') 
  
#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
  
plt.imshow(image)
pixel_vals = image.reshape((-1,3)) 
  
pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image.shape))

plt.imshow(segmented_image)
cv2.imshow("Original image",image)

cv2.imshow("Clustring image", segmented_image)
k = cv2.waitKey(0)   # otherwise the program would end far too quickly
if k == ord("s"):
    cv2.imwrite("starry_night.png", segmented_image)



