from PIL import Image
import glob
import cv2

# Resolved an issue with incorrect data attributes during training (800,800,3)
# The Image read by cv2 contains information (length, width, channel number), and the image read by image contains information (length, width).
path = "D:/keke/matting/mine-matting_0823/train_preprocessing/data_adobe_denoise/bg"
i=0
for a in glob.glob(path + '/*.jpg'):
    img = Image.open(a)
    img = img.convert('RGB')
    img.save(a)
    i=i+1
    print(i)

