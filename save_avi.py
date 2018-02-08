# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:46:13 2018

@author: yzj
"""

import numpy as np 
import cv2

#确定视频的编码格式
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#确定输出的名称、编码格式、播放频率、帧的大小
#第二个参数如果为-1，那就可以从系统中有的选出来
#第三个参数5.0为正常，小于为慢镜头
out = cv2.VideoWriter('output.avi',fourcc,5,(640,480))

#建立一个摄像头对象，0为笔记本内置，1 2 3 4。。。为其他外接
cap = cv2.VideoCapture(0)
while cap.isOpened() != True:
    cap.open()

while(1):
    success, frame = cap.read()
    
    if success == True:
        #第二个参数为整数，小于1为反的图形
        frame=cv2.flip(frame,1)
        #frame = cv2.cvtColor(frame, -2)
        out.write(frame)
        
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord ('Q'):
            break
    
    else:
        break

#释放摄像头
cap.release()
out.release()
cv2.destroyAllWindows()