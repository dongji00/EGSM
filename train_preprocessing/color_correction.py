import cv2
import numpy as np
import glob
# Local_Color_correction
path = '/content/drive/My Drive/mine_train_0113/data_train/adobe/merged_train/'

iter=0
for a in glob.glob(path + '/*.png'):
    img = cv2.imread(a)
    h, w = img.shape[:2]
    I = np.zeros((h, w), dtype=np.float64)
    Out_img = img.copy()

    B = np.array(img[:, :, 0], dtype=np.float64)
    G = np.array(img[:, :, 1], dtype=np.float64)
    R = np.array(img[:, :, 2], dtype=np.float64)

    #print(B.dtype)
    for i in range(0, h):
        for j in range(0, w):
            I[i, j] = ((B[i, j] + G[i, j] + R[i, j])) / 3.0

    Revert_I = np.full((h, w), 255.0, dtype=np.float64)
    Mast = Revert_I - I

    Mast_Blur = cv2.GaussianBlur(Mast, (41, 41), 0)

    for k in range(0, 3):
        for i in range(0, h):
            for j in range(0, w):
                Exp = 2 ** ((128 - Mast_Blur[i, j]) / 128.0)
                Value = np.uint8(255 * ((img.item(i, j, k) / 255.0) ** Exp))
                Out_img.itemset((i, j, k), Value)
    cv2.imwrite(a, Out_img)
    iter +=1
    print(iter)
'''cv2.namedWindow("original image")
cv2.namedWindow("output image")

cv2.imshow("original image", img)
cv2.imshow("output image", Out_img)'''


#cv2.waitKey(0)  
