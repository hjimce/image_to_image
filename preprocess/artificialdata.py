import cv2
import os
#制作模糊数据
def choose_method(image,method):
    if method='avg_smooth':
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(img,-1,kernel)
    elif method=


    return dst

def create_noise_data():
    return
def create_blur_data(dataroot):
    files=os.listdir
    for f in files:
        image=cv2.i


