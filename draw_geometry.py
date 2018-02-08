# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:56:42 2018

@author: yzj
"""

import numpy as np
import cv2
#建立画布,一个是创建的图片矩阵大小，另一个是数据类型 
#512*512的像素，3是BGR三色; uint8指的是每种颜色是0-255
img=np.zeros((512,512,3),np.uint8)

#直线是图片名称、起点坐标、终点坐标、颜色、线宽（-1为填满） 
cv2.line(img,(0,0),(511,511),(255,0,0),5)
#矩形是图片名称、两个对角点坐标、颜色数组、线宽 
cv2.rectangle(img,(38,0),(510,128),(0,255,0),-1)
#圆是图片名称、圆心坐标、半径、颜色数组、线宽 
cv2.circle(img,(447,63),63,(0,0,255),-1)
#椭圆是图像、中心坐标、长轴短轴、旋转角度（顺时针）、显示的部分、颜色数组、线宽 
cv2.ellipse(img,(256,256),(100,50),90,0,270,(0,0,255),-1)
#多边形是图像、顶点集、是否闭合、颜色数组、线宽 
pts=np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
pts=pts.reshape((-1,1,2))
img=cv2.polylines(img,[pts],False,(0,255,255),3)
#写字，图片、字符串、坐标、字体、字号、颜色数组、线宽、线条种类 
font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(img,'YZJ',(10,500),font,4,(255,255,255),2,cv2.LINE_4)

cv2.namedWindow('YZJ')
cv2.imshow('YZJ',img)
cv2.waitKey(0)
cv.destroyWindow('YZJ')