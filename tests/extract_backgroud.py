
"""
处理数据集
    目标分割，
输入网络前处理数据集
    对马的图像进行目标分割 => (能不能获取到轮廓像素点，位置点) 前景输入到网络GA
    对斑马图像进行目标分割 => (获取轮廓像素点，位置点) 前景输入网络 GB

    网络间传输的就是前景
    对前景做处理，处理结果=> 和原背景进行融合

    Python 图像掩码

    https://juejin.cn/post/6886732020085424136

目标分割
获取前景，坐标
生成背景
根据坐标贴回原图片

"""


import cv2
import numpy as np

img = cv2.imread('hand.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,binary = cv2.threshold(gray,60,255,0)#阈值处理
contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#查找轮廓
print(len(contours))
x = 0
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area>10000:
        print(area)
        x = i
cnt = contours[x]
img1 = img.copy()
approx1 = cv2.approxPolyDP(cnt,3,True)#拟合精确度
img1  =cv2.polylines(img1,[approx1],True,(255,255,0),2)
cv2.imwrite('approxPolyDP1.jpg',img1)

img2 = img.copy()
approx2 = cv2.approxPolyDP(cnt,5,True)#拟合精确度
img2  =cv2.polylines(img2,[approx2],True,(255,255,0),2)
cv2.imwrite('approxPolyDP2.jpg',img2)

img3 = img.copy()
approx3 = cv2.approxPolyDP(cnt,7,True)#拟合精确度
img3  =cv2.polylines(img3,[approx3],True,(255,255,0),2)
cv2.imwrite('approxPolyDP3.jpg',img3)


cv2.imwrite("dst.png",img1)
print(len(approx1))
