# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:50:20 2018

@author: yzj
"""

import cv2
import numpy as np

'''
#可以把event都给打印出来
events=[i for i in dir(cv2) if 'EVENT' in i]
print(events)
'''

drawing = False
mode = True
ix,iy = -1,-1
img=cv2.imread('IMG_2005.JPG')

#第一个参数是回调函数来的值，xy是返回的位置，flags有六种可以配合EVENT_MOUSEMOVE使用，para是setMouseCallback第三个参数
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),3,(255,0,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing == False

cv2.namedWindow('image',cv2.WINDOW_NORMAL)

#鼠标回调函数,其对应的draw_circle函数格式必须像上面一样
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1)&0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
    
cv2.destroyAllWindows()