import cv2
import numpy as np
import math
import _双边滤波 as blf

'''
    首先，构建相似形权重模板，双边滤波根据原图对于每一个位置，通过该位置和其邻域的灰度值的差的指数来估计相似形，而
联合双边滤波是首先对原图进行高斯平滑，根据平滑的结果，用当前位置及其邻域的值来估计相似性权重模板
    接下来，空间距离权重模板和相似性权重模板点乘，然后归一化，作为最后的权重模板，最后将权重模板与原图（注意不是高斯平滑的结果）
在该位置的邻域对应位置积的和作为输出值。
    整个过程只有第二步计算相似性权重模板时和双边滤波不同，但是对图像平滑的效果，特别是对纹理图像来说，却有很大的不同

'''

def jointBLF(I, H, W, sigma_g, sigma_d, borderType=cv2.BORDER_DEFAULT):
    # 构建空间距离权重模板
    closenessWeight = blf.getClosenessWeight(sigma_g, H, W)
    # 对 I 进行高斯平滑
    Ig = cv2.GaussianBlur(I, (H, W), sigma_g)
    # 模板的中心点位置
    cH = int((H - 1) / 2)
    cW = int((W - 1) / 2)
    # 对原图和高斯平滑的结果扩充边界
    Ip = cv2.copyMakeBorder(I, cH, cH, cW, cW, borderType)
    Igp = cv2.copyMakeBorder(Ig, cH, cH, cW, cW, borderType)
    # 图像矩阵的行数和列数
    rows, cols = I.shape
    i, j = 0, 0
    # 联合双边滤波的结果
    jblf = np.zeros(I.shape, np.float64)
    for r in range(cH, cH + rows, 1):
        for c in range(cW, cW + cols, 1):
            # 获取当前位置的值
            pixel = Igp[r][c]
            # 当前位置的邻域
            rTop, rBottom = r - cH, r + cH
            cLeft, cRight = c - cW, c + cW
            # 从Igp中截取该邻域，用于构建该位置的相似性权重模板
            region = Igp[rTop: rBottom + 1, cLeft: cRight + 1]
            # 通过上述邻域，构建该位置的相似形权重模板
            similarityWeight = np.exp(-0.5 * np.power(region - pixel, 2.0) / math.pow(sigma_d, 2.0))
            # 相似形权重模板和空间距离权重模板相乘
            weight = closenessWeight * similarityWeight
            # 将权重进行归一化
            weight = weight / np.sum(weight)
            # 权重模板和邻域对应位置相乘并求和
            jblf[i][j] = np.sum(Ip[rTop: rBottom + 1, cLeft: cRight + 1] * weight)
            j += 1
        j = 0
        i += 1
    return jblf


if __name__ == '__main__':
    image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)
    # 将 8 位图转换为浮点型
    FI = image.astype(np.float64)
    # 联合双边滤波，返回值的数据类型为浮点型
    jblf = jointBLF(image, 7, 7, 7, 2)

    bilateral_filter = cv2.ximgproc.jointBilateralFilter(image, image, 7, 7, 2)

    # 转换为 8 位图
    jblf = np.round(jblf)
    jblf = jblf.astype(np.uint8)
    cv2.imshow('image', image)
    cv2.imshow('bilateral_filter', bilateral_filter)
    cv2.imshow('jblf', jblf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




























