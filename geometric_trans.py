# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 22:54:21 2018

@author: yzj
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
#---------------------zoom function-------------------------------#
'''
img = cv2.imread('IMG_2345.JPG')
#方法一：None处是输出图像的尺寸，但是有缩放因子fx,fy就写None
res1 = cv2.resize(img,None,fx=3,fy=2,interpolation=cv2.INTER_CUBIC)

#方法二：使用图像的输出尺寸
height,width = img.shape[:2]
res2 = cv2.resize(img,(3*width,2*height),interpolation=cv2.INTER_CUBIC)

while(1):
	cv2.imshow('res1',res1)
	cv2.imshow('res2',res2)
	cv2.imshow('img',img)

	if cv2.waitKey(1)&0xFF == ord('t'):
		break

cv2.destroyAllWindows()
'''
#---------------------hsv capture function-------------------------------#
'''
#打开0号摄像头
cap = cv2.VideoCapture(0)
while cap.isOpened() != True:
    cap.open()

while (1):
	ret, frame = cap.read()

	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	lower_blue = np.array([10,50,50])
	upper_blue = np.array([130,255,255])

	mask = cv2.inRange(hsv,lower_blue,upper_blue)
	res = cv2.bitwise_and(frame,frame,mask = mask)

	cv2.imshow('frame',frame)
	cv2.imshow('mask',mask)
	cv2.imshow('res',res)

	if cv2.waitKey(1)&0xFF == ord('t'):
		break

cap.release()
cv2.destroyAllWindows()
'''

#---------------------offset function-------------------------------#
'''
img1 = cv2.imread('IMG_2345.JPG')
#设置平移矩阵，100和50是可调参数
H = np.float32([[1,0,10],[0,1,5]])
rows,cols = img1.shape[:2]
#图像、变换矩阵、变换后大小
res1 = cv2.warpAffine(img1,H,(rows,cols))
cv2.imshow('img1',img1)
cv2.imshow('res1',res1)
while(1):
    if cv2.waitKey(1)&0xFF == ord('t'):
        cv2.destroyAllWindows()
'''

#---------------------rotate function-------------------------------#
'''
img2 = cv2.imread('IMG_2345.JPG')
rows,cols = img2.shape[:2]
#设置旋转矩阵，旋转中心元组、逆时针旋转角度、缩放因子
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1.2)
#图像、变换矩阵、输出图像大小
res2 = cv2.warpAffine(img2,M,(2*cols,2*rows))
cv2.imshow('img1',img2)
cv2.imshow('res1',res2)
while(1):
    if cv2.waitKey(1)&0xFF == ord('t'):
        cv2.destroyAllWindows()
'''

#--------------------- affine function-------------------------------#
'''
在仿射变换中，原图中所有的平行线在结果图像中同样平行。为了创建这
个矩阵我们需要从原图像中找到三个点以及他们在输出图像中的位置。
'''
'''
img3 = cv2.imread('IMG_2345.JPG')
rows3,cols3,ch3 = img3.shape
#输入的三个点
pts1 = np.float32([[50,50],[200,50],[50,200]])
#输出的三个点
pts2 = np.float32([[10,100],[200,50],[100,250]])
#创建2*3矩阵
M3 = cv2.getAffineTransform(pts1,pts2)
dst3 = cv2.warpAffine(img3,M3,(cols3,rows3))
cv2.imshow('img3',img3)
cv2.imshow('dst3',dst3)
while(1):
    if cv2.waitKey(1)&0xFF == ord('t'):
        cv2.destroyAllWindows()
'''


#--------------------- 透视变换 function-------------------------------#
'''
对于视角变换，我们需要一个 3x3 变换矩阵。在变换前后直线还是直线。
要构建这个变换矩阵，需要在输入图像上找 4 个点，以及他们在输出图
像上对应的位置。这四个点中的任意三个都不能共线。
'''
img4 = cv2.imread('IMG_2345.JPG')
rows4,cols4 = img4.shape[:2]
col_change,row_change = 300,300
pts1_4 = np.float32([[0,0],[cols4,0],[0,rows4],[cols4,rows4]])
pts2_4 = np.float32([[0,0],[col_change,40],[50,row_change],[col_change,row_change]])
M4 = cv2.getPerspectiveTransform(pts1_4,pts2_4)
dst4 = cv2.warpPerspective(img4,M4,(cols4,rows4))
cv2.imshow('img4',img4)
cv2.imshow('dst4',dst4)
while(1):
    if cv2.waitKey(1)&0xFF == ord('t'):
        cv2.destroyAllWindows()
