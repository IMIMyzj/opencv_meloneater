# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:37:58 2018

@author: yzj
"""

'''
首先将logo转成灰度图像，然后使用threshold函数将灰阶像素与纯白像素分开得到一个2值图像并用其作为掩码图像。
使用带掩码的位操作扣出logo形状，反向掩码的位操作得到logo图像，两者相加得到带logo的背景图。 
注：如果直接使用权重加，必然会有透明度，或者就是logo对背景进行全遮挡。
'''

import cv2
import numpy as np

#-------------获取图像信息基本操作-----------#
img = cv2.imread('IMG_2005.JPG',1)
#获得图像的行数、列数和通道数的元组
print(img.shape)

#返回像素的数目
print(img.size)

#返回像素的类型
print(img.dtype)

#获得某个点的R/G/B值，并修改
print(img.item(100,100,0))
img.itemset((100,100,2),0)
print(img.item(100,100,0))

#-------------合成图像----------------------#
img1 = img                              #大图
img2 = cv2.imread('IMG_2345.JPG')		#小图

rows,cols,channels = img2.shape
roi = img1[800:800+rows,800:800+cols]

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)  #阈值,阈值化后的图=cv2.threshold(图像,阈值,使用值上限,使用的阈值类型)
#print(ret)

#一堆 bitwise_not/and/or/xor
#cv2.bitwise_and(第一个对象，第二个对象，输出值dst，掩膜（黑的地方不处理做零，即蒙版）)
mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

dst = cv2.add(img1_bg,img2_fg)
img1[800:800+rows,800:800+cols] = dst

cv2.namedWindow('res',cv2.WINDOW_NORMAL)
cv2.imshow('res',img1_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()

