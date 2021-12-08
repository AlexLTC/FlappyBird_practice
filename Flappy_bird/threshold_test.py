import cv2
import numpy as np

img = cv2.imread('/home/xuus/圖片/桌布/stretched-1920-1080-1190662.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

print(img.shape)  # (1080, 1920)
 
s = np.stack((img, img, img, img), axis=0)

print('s shape:', s.shape)  # (4, 1080, 1920)
