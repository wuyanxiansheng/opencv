import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

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

def equalHist(image):
    # 灰度图像矩阵的高、宽
    rows, cols = image.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(image)
    # 第二步，计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    output_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (rows * cols)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            output_q[p] = math.floor(q)
        else:
            output_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(image.shape, np.uint8)
    for r in range(rows):
        for c in range(cols):
            equalHistImage[r][c] = output_q[image[r][c]]
    return equalHistImage


if __name__ == "__main__":
    image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)
    grayHist = calcGrayHist(image)

    equalImage = equalHist(image)
    equalGrayHist = calcGrayHist(equalImage)

    # 画出灰度直方图
    x_range = range(256)
    plt.plot(x_range, grayHist, 'r', linewidth=2, c='black')
    # 设置坐标轴的范围
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue])
    # 设置坐标轴的标签
    plt.xlabel('gray Level')
    plt.ylabel('number of pixels')

    # x_range = range(256)
    plt.plot(x_range, equalGrayHist, 'r', linewidth=2, c='black')
    new_Y_maxValue = np.max(equalGrayHist)
    plt.axis([0, 255, 0, new_Y_maxValue])
    plt.xlabel('equal gray level')
    plt.ylabel('number of pixels')

    # 显示灰度直方图
    plt.show()

    cv2.imshow('image', image)
    cv2.imshow('equalImage', equalImage)
    cv2.waitKey(0)













