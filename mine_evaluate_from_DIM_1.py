import numpy as np
import cv2
import glob
import os
from sklearn.preprocessing import normalize
from sklearn import preprocessing
unknown_code = 128

def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    # print('unknown: ' + str(unknown))
    loss = np.sum(np.square(error_map) * mask) / np.sum(mask)
    # print('mse_loss: ' + str(loss))
    return loss


# compute the SAD error given a prediction, a ground truth and a trimap.
#
def compute_sad_loss(pred, target, trimap):
    error_map = np.abs(pred - target) / 255.
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    loss = np.sum(error_map * mask)

    # the loss is scaled by 1000 due to the large images used in our experiment.
    loss = loss / 1000
    # print('sad_loss: ' + str(loss))
    return loss

def mse(pred, target):
    mse = np.sqrt(np.mean( np.sum((pred - target) ** 2 )))
    return mse

# the loss is scaled by 1000 due to the large images used in our experiment.
def sad(pred, target):
    sad = np.sum(np.abs(pred - target))/1000
    return sad


def AutoNorm(mat):
    n = len(mat)
    m = mat.shape[2]
    MinNum = [9999999999] * m
    MaxNum = [0] * m
    for i in mat:
        for j in range(0, m):
            if i[j] > MaxNum[j]:
                MaxNum[j] = i[j]

    for p in mat:
        for q in range(0, m):
            if p[q] <= MinNum[q]:
                MinNum[q] = p[q]

    section = list(map(lambda x: x[0] - x[1], zip(MaxNum, MinNum)))
    print(section)
    NormMat = []

    for k in mat:
        distance = list(map(lambda x: x[0] - x[1], zip(k, MinNum)))
        value = list(map(lambda x: x[0] / x[1], zip(distance, section)))
        NormMat.append(value)
    return NormMat

def gradient_error(pred,target):

    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    target = (target - np.min(target)) / (np.max(target) - np.min(target))

    pred = cv2.GaussianBlur(pred,(3,3),1.4)
    #print(pred)
    [pred_x, pred_y] = pred.shape[:2]
    target = cv2.GaussianBlur(target,(3,3),1.4)
    [target_x, target_y] = target.shape[:2]
    pred_amp = np.sqrt(pred_x**2 + pred_y**2)
    target_amp = np.sqrt(target_x**2 + target_y**2)
    gradient_error = np.sum(pred_amp-target_amp)
    return(gradient_error)


img_path = 'D:/keke/matting/mine-matting_0823/test_code/mine_test_yy/input/'
img_pred_path = 'D:/keke/matting/mine-matting_0823/test_code/mine_test_yy/output_mine_0119_26/'
trimap_path = 'D:/keke/matting/mine-matting_0823/test_code/mine_test_yy/trimap/'
total_sad_loss = 0
total_mse_loss = 0
total_grad_loss = 0
#total_conn_loss = 0
img_list = []
img_pred_list = []
trimap_list = []
for a in glob.glob(img_path + '/*_maskDL.png'):
    img_list.append(a)
print(len(img_list))
for b in glob.glob(img_pred_path + '/*_out.png'):
    img_pred_list.append(b)
print(len(img_pred_list))
for c in glob.glob(trimap_path + '/*_trimap.png'):
    trimap_list.append(c)

for i in range(0,len(img_list)):

    img = cv2.imread(os.path.join(img_path, img_list[i]))
    #print(img)
    img_pred = cv2.imread(os.path.join(img_pred_path, img_pred_list[i]))
    trimap = cv2.imread(os.path.join(trimap_path, trimap_list[i]))
    sad_value = compute_sad_loss(img_pred,img,trimap)
    mse_value = compute_mse_loss(img_pred,img,trimap)
    #grad_value = gradient_error(img_pred,img)
    #print(grad_value)
    total_sad_loss += sad_value
    total_mse_loss += mse_value
    #total_grad_loss += grad_value
    #total_conn_loss += conn

print(total_sad_loss, total_mse_loss, total_grad_loss)
