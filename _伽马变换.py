import cv2
import numpy as np
import sys
# 当γ值等于1时，图像不变。如果图像整体或感兴趣区域较暗，则令 0 < γ < 1 可以增加图像对比度
image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)

# 图像归一化
fI = image / 255.0
# 伽马变换
gamma = 0.5
O = np.power(fI, gamma)
# 显示原图和伽马变换后的效果
cv2.imshow('image', image)
cv2.imshow('O', O)

cv2.waitKey(0)