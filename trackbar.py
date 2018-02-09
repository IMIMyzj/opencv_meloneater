# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:22:07 2018

@author: yzj
"""

import cv2
import numpy as np

drawing = False
mode = True
ix,iy = -1,-1
s = 0
img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#createTrackbar函数有五个参数，bar name、图name、默认起始点、最大值、回调函数（含有一个默认参数表示滑动条的位置）

def nothing(x):
    pass

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,s

    r=cv2.getTrackbarPos('R','image')
    g=cv2.getTrackbarPos('G','image')
    b=cv2.getTrackbarPos('B','image')
    s=cv2.getTrackbarPos(switch,'image')
    color = (b,g,r)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),color,-1)
            else:
                cv2.circle(img,(x,y),3,color,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing == False

cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch,'image',0,1,nothing)
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)&0xFF
    if k == 27:
        break
    elif k == ord('m'):
        mode = not mode
    if s == 0:
        img[:] = 255

cv2.destroyAllWindows()