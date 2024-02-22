
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal
# 获得高斯滤波核



def getGaussKernel(simga, H, W):
    # 第一步：构建高斯矩阵
    gaussMatrix = np.zeros([H, W], np.float32)
    # 得到中心点的位置
    cH = (H - 1) / 2
    cW = (W - 1) / 2
    # 计算gauss(sigma, r, c)
    for r in range(H):
        for c in range(W):
            norm2 = math.pow(r - cH, 2) + math.pow(c - cW, 2)
            gaussMatrix[r][c] = math.exp(-norm2 / (2 * math.pow(simga, 2)))
    # 第二步：计算高斯矩阵的和
    sumGM = np.sum(gaussMatrix)
    # 第三步：归一化
    gaussKernel = gaussMatrix / sumGM
    return gaussKernel


# 进行高斯滤波
def gaussBlur(image, sigma, H, W, _boundary='fill', _fillvalue=0):
    # 构建水平方向上的高斯卷积核
    gaussKenrnel_x = cv2.getGaussianKernel(sigma, W, cv2.CV_64F)
    # 转置
    gaussKenrnel_x = np.transpose(gaussKenrnel_x)
    # 图像矩阵与水平高斯核卷积
    gaussBlur_x = signal.convolve2d(image, gaussKenrnel_x, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    # 构建垂直方向上的高斯卷积核
    gaussKenrnel_y = cv2.getGaussianKernel(sigma, H, cv2.CV_64F)
    # 与垂直方向上的高斯核进行卷积
    gaussKenrnel_xy = signal.convolve2d(gaussBlur_x, gaussKenrnel_y, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    return gaussKenrnel_xy

if __name__ == '__main__':
    image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)
    # 高斯平滑
    blurImage = gaussBlur(image, 5, 5, 5, 'symm')
    # 对blurImage进行灰度级显示：round（取整，四舍五入）
    blurImage = np.round(blurImage)
    blurImage = blurImage.astype(np.uint8)



    gaussianBlurImage = cv2.GaussianBlur(image, (5, 5), sigmaX=5, sigmaY=5)
    gaussianBlurImage = np.round(gaussianBlurImage)
    gaussianBlurImage = gaussianBlurImage.astype(np.uint8)


    cv2.imshow('gaussianBlurImage', gaussianBlurImage)
    cv2.imshow('GuassBlur', blurImage)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()













