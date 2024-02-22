import cv2
import numpy as np
import matplotlib.pyplot as plt

# 获得灰度直方图
def calcGrayHist(image):
    # 获取灰度图像的高、宽
    h, w = image.shape[:2]
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(h):
        for c in range(w):
            grayHist[image[r][c]] += 1
    return grayHist

'''


grayHist = calcGrayHist(image)
# 画出灰度直方图
x_range = range(256)
plt.plot(x_range, grayHist, 'r', linewidth=2, c='black')
# 设置坐标轴的范围
y_maxValue = np.max(grayHist)
plt.axis([0, 255, 0, y_maxValue])
# 设置坐标轴的标签
plt.xlabel('gray Level')
plt.ylabel('number of pixels')
# 显示灰度直方图
plt.show()
'''

image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)

rows, cols = image.shape
# 将二维的图像矩阵，变为一维的数组，便于计算灰度直方图
pixelSequence = image.reshape([rows * cols, ])
# 组数
numberBins = 256
# 计算灰度直方图  Matplotlib 本身提供了计算直方图的函数hist
histogram, bins, patch = plt.hist(pixelSequence, numberBins, facecolor='black', histtype='bar')
# 设置坐标轴的标签
plt.xlabel(u"gray Level")
plt.ylabel(u"number of pixels")
# 设置坐标轴的范围
y_maxValue = np.max(histogram)
plt.axis([0, 255, 0, y_maxValue])
# plt.show()


# OPENCV中Python API：normalize(src, alpha, beta, norm_type, dtype)
# alpha：OMax  beta：Omin  norm_type：边界扩充类型：1-范数（绝对值之和）2-范数(平方和的开方) ∞-范数：(绝对值的最大值)
dst = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow('normalizeImage', dst)


# 直方图正规化
# 获取原图的最大值、最小值
Imax = np.max(image)
Imin = np.min(image)
# 设置输出的最小最大灰度级
Omax, Omin = 255, 0
# 计算a的值
a = float((Omax - Omin) / (Imax - Imin))
b = Omin - a * Imin
# 矩阵的线性变换
O = a * image + b
# 数据类型转换
O = O.astype(np.uint8)
# 显示原图和直方图正规化的效果
cv2.imshow("image", image)
cv2.imshow("O", O)

cv2.waitKey(0)

















