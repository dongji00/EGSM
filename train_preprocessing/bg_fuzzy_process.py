import os
import cv2
from PIL import ImageFilter
from PIL import Image

path = 'D:/keke/matting/mine-matting_0823/train_preprocessing/data_adobe_fuzzy/mine_bg/'

all = os.walk(path)
for path, dir, filelist in all:
    for filename in filelist:
        if filename.endswith('/*.jpg'):
            filepath = os.path.join(path, filename)
            #img = cv2.imread(filepath)
            img = Image.open(filepath)
            img2 = img.filter(ImageFilter.BLUR)
            #img2.save("d:/keke/matting/mine_background_matting/bg/"+filename)
            img2.save(filepath)
