import cv2 
import numpy as np
from scipy import signal

def find_dft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    edged = cv2.Canny(gray, 30, 200) 
    contours, hierarchy = cv2.findContours(edged,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    if len(contours)==0:
        return False
    contours = np.reshape(contours,[-1,2])

    u_hat = contours[:,0]+1j*contours[:,1]
    u_hat[~np.isnan(u_hat).any(axis=0)]

    template = np.fft.fft(u_hat)
    template_unit = template/np.absolute(template)

    return template_unit

image = cv2.imread('./sample.png') 
targetImage = cv2.imread('./a3.png') 

sheight,swidth = image.shape[0],image.shape[1]
height,width = targetImage.shape[0],targetImage.shape[1]

cv2.waitKey(0) 
  
# Grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# Find Canny edges 
edged = cv2.Canny(gray, 30, 200) 
cv2.waitKey(0) 
  
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

template_unit = find_dft(image)
cv2.imshow('Canny Edges After Contouring', edged) 
cv2.waitKey(0) 

print("Number of Contours found = " + str(len(contours))) 
  
# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
  
cv2.imshow('Contours', image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 


for h in range(sheight,height):
    for w in range(swidth,width):
        crop = targetImage[h-sheight:h,w-swidth:w]
        stemplate = find_dft(crop)
        if stemplate:
            print(np.shape(stemplate))
        # #     continue
        # corr = signal.correlate(template_unit, stemplate, mode='same') 
        # print(corr)


