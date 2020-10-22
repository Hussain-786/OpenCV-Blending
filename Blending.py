import cv2
import numpy as np

apple = cv2.imread('apple.jpg')
orange = cv2.imread('orange.jpg')
apple_orange = np.hstack((apple[:, :256], orange[:,256:]))
apple_layer = apple.copy()
apple_gp =[apple_layer]
for i in range(6):
    apple_layer = cv2.pyrDown(apple_layer)
    apple_gp.append(apple_layer)

orange_layer = orange.copy()
orange_gp =[orange_layer]
for i in range(6):
    orange_layer = cv2.pyrDown(orange_layer)
    orange_gp.append(orange_layer)

apple_layer = apple_gp[5]
apple_lp = [apple_layer]
for i in range(5, 0, -1):
    apple_extended = cv2.pyrUp(apple_gp[i])
    laplacian = cv2.subtract(apple_gp[i-1], apple_extended)
    apple_lp.append(laplacian)

orange_layer = orange_gp[5]
orange_lp = [orange_layer]
for i in range(5, 0, -1):
    orange_extended = cv2.pyrUp(orange_gp[i])
    laplacian = cv2.subtract(orange_gp[i-1], orange_extended)
    orange_lp.append(laplacian)

n=0
apple_orange_pyramid = []
for apple_bel,orange_bel in zip(apple_lp,orange_lp):
    n+=1
    col,row,chl = apple_bel.shape
    laplacian = np.hstack((apple_bel[:, :int(col/2)], orange_bel[:,int(col/2):]))
    apple_orange_pyramid.append(laplacian)

apple_orange_reconstruct = apple_orange_pyramid[0]
for i in range(1,6):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i],apple_orange_reconstruct)

cv2.imshow('apple', apple)
cv2.imshow('orange', orange)
cv2.imshow('apple_orange', apple_orange)
cv2.imshow('apple_orange_reconstruct', apple_orange_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()
