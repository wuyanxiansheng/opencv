import cv2
import numpy as np
import math

def getClosenessWeight(sigma_g, H, W):
    r, c = np.mgrid[0:H:1, 0:W:1]
    r = r - (H - 1) / 2
    c = c - (W - 1) / 2
    closeWeight = np.exp(-0.5 * (np.power(r, 2) + np.power(c, 2)) / math.pow(sigma_g, 2))
    return closeWeight

def bfltGray(I, H, W, sigma_g, sigma_d):
    # 构建空间距离权重模板
    closenessWeight = getClosenessWeight(sigma_g, H, W)
    # 模板的中心点所在位置
    cH = int((H - 1) / 2)
    cW = int((W - 1) / 2)
    # 图像矩阵的行数和列数
    rows, cols = I.shape
    # 双边滤波的结果
    bfltGrayImage = np.zeros(I.shape, np.float32)
    for r in range(rows):
        for c in range(cols):
            pixel = I[r][c]
            # 判断边界
            rTop = 0 if r - cH < 0 else r - cH
            rBottem = rows - 1 if r + cH > rows - 1 else r + cH
            cLeft = 0 if c - cW < 0 else c - cW
            cRight = cols - 1 if c + cW > cols - 1 else c + cW
            # 权重模板作用的区域
            region = I[rTop : rBottem + 1, cLeft: cRight + 1]
            # 构建灰度值相似形的权重因子
            similarityWeightTemp = np.exp(-0.5 * np.power(region - pixel, 2.0) / math.pow(sigma_d, 2))
            closenessWeightTemp = closenessWeight[rTop - r + cH: rBottem - r + cH + 1, cLeft - c + cW: cRight - c + cW + 1]
            # 两个权重模板相乘
            weightTemp = similarityWeightTemp * closenessWeightTemp
            # 归一化权重模板
            weightTemp = weightTemp / np.sum(weightTemp)
            # 权重模板和对应的邻域值相乘求和
            bfltGrayImage[r][c] = np.sum(region * weightTemp)

    return bfltGrayImage

if __name__ == '__main__':
    image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('img/girl.ppm')
    # 显示原图
    cv2.imshow('image', image)

    # opencv 里面的函数
    # src：输入的图像，可以是单通道或多通道，数据类型为uint8或float32。
    #
    # d：滤波过程中每个像素邻域的直径，一般为正数，表示像素邻域的宽度，如3、5、7等。
    #
    # sigmaColor：颜色空间滤波器的sigma值，可理解为图像中颜色差异范围的阈值，一般取非负数。
    #
    # sigmaSpace：坐标空间滤波器的sigma值，可理解为像素之间的距离阈值，一般取非负数。
    #
    # dst（可选）：输出的图像，大小与输入图像相同
    #
    # borderType（可选）：用于处理边界的像素填充方式。
    dst = cv2.bilateralFilter(img, 3, 0.5, 19)
    # 将灰度值归一化
    image = image / 255.0
    # 双边滤波
    bfltImage = bfltGray(image, 33, 33, 19, 0.5)


    # 显示双边滤波的结果
    cv2.imshow('bfltBlur', bfltImage)
    cv2.imshow('dst', dst)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


























