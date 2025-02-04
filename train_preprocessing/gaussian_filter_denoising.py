import cv2
import numpy as np
import math
import os
import glob
from PIL import Image

def gausskernel(size):
    sigma = 1.0 # σ=1.0
    gausskernel = np.zeros((size, size), np.float32) 
    for i in range(size): 
        for j in range(size):
            norm = math.pow(i - 1, 2) + pow(j - 1, 2) 
            gausskernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2)))  
    sum = np.sum(gausskernel)  
    kernel=gausskernel/sum # Normalization
    return kernel


# Custom Gaussian filter function
def Gaussian(img, size):
    num = int((size - 1) / 2)  # The padding size required for the input image
    img = cv2.copyMakeBorder(img, num, num, num, num, cv2.BORDER_REPLICATE)  # Expand the original image to handle edge effects
    h, w = img.shape[0:2]  # Get the width and height of the input image
    # Gaussian filtering
    img1 = np.zeros((h, w, 3), dtype="uint8")
    kernel = gausskernel(size)  # Compute the Gaussian convolution kernel
    for i in range(num, h - num):
        for j in range(num, w - num):
            sum = 0
            q = 0
            p = 0
            for k in range(i - num, i + num + 1):
                for l in range(j - num, j + num + 1):
                    sum = sum + img[k, l] * kernel[q, p]  # Gaussian filtering
                    p = p + 1  # Count columns in the Gaussian kernel
                q = q + 1  # Count rows in the Gaussian kernel
                p = 0  # Reset column count after each inner loop iteration
            img1[i, j] = sum
    img1 = img1[(0 + num):(h - num), (0 + num):(h - num)]  # Crop the image back to its original size
    return img1



'''# Read the image
img = cv2.imread("E:\matting\Background-Matting-master-train\input/0001_img.png")
# Get image attributes
h, w = img.shape[0:2]'''
# Add noise
'''for i in range(3000):    # Add 3000 noise points
    x = np.random.randint(0, h)
    y = np.random.randint(0, w)
    img[x, y, :] = 255'''
# Call the mean filter function from the OpenCV library
# result = Gaussian(img, 5) # Pass the read image and kernel size
'''cv2.imshow("src", img)
cv2.imshow("Gaussian", result)
cv2.waitKey(0)'''

i = 0
path = "/content/drive/My Drive/mine_train_0113/data_train/adobe/fg_train"
for a in glob.glob(path + '/*.jpg'):
    # img = cv2.imread(a)
    img = Image.open(a)
    # h, w = img.shape[0:2]
    img = Gaussian(img, 5)
    # cv2.imwrite(a, img)
    img.save(a)
    i = i + 1
    print(i)