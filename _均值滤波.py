import cv2
import numpy as np

def integral(image):
    rows, cols = image.shape
    # 行积分运算
    inteImageC = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            if c == 0:
                inteImageC[r][c] = image[r][c]
            else:
                inteImageC[r][c] = inteImageC[r][c - 1] + image[r][c]
    # 列积分运算
    inteImage = np.zeros(image.shape, np.float32)
    for c in range(cols):
        for r in range(rows):
            if r == 0:
                inteImage[r][c] = inteImageC[r][c]
            else:
                inteImage[r][c] = inteImage[r - 1][c] + inteImageC[r][c]
    # 上边和左边进行补零
    inteImage_0 = np.zeros((rows + 1, cols + 1), np.float32)
    inteImage_0[1 : rows + 1, 1 : cols + 1] = inteImage
    return inteImage_0


# 实现了图像的积分后，通过定义函数fastMeanBlur来实现均值平滑，其中 image 是输入矩阵，winSize是平滑窗口尺寸
# 宽、高均为奇数   borderType为边界扩充类型（理想的边界扩充类型是镜像扩充）
def fastMeanBlur(image, winSize, borderType = cv2.BORDER_DEFAULT):
    halfH = int((winSize[0] - 1) / 2)
    halfW = int((winSize[1] - 1) / 2)
    ratio = 1.0 / (winSize[0] * winSize[1])
    # 边界扩充
    paddImage = cv2.copyMakeBorder(image, halfH, halfH, halfW, halfW, borderType)
    # 图像积分
    paddIntegral = integral(paddImage)
    # 图像的高、宽
    rows, cols = image.shape
    # 均值滤波后的结果
    meanBlurImage = np.zeros(image.shape, np.float32)
    r, c = 0, 0
    for h in range(halfH, halfH + rows, 1):
        for w in range(halfW, halfW + cols, 1):
            meanBlurImage[r][c] = (paddIntegral[h + halfH + 1][w + halfW + 1] + paddIntegral[h - halfH][w - halfW] -
                                   paddIntegral[h + halfH + 1][w - halfW] - paddIntegral[h - halfH][w + halfW + 1]) * ratio
            c += 1
        r += 1
        c = 0
    return meanBlurImage

if __name__ == '__main__':
    image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)
    blurImage = fastMeanBlur(image, (9, 9), borderType=cv2.BORDER_DEFAULT)
    blurImage = np.round(blurImage)
    blurImage = blurImage.astype(np.uint8)

    dst = cv2.blur(image, (9, 9))
    cv2.imshow('blur', dst)

    cv2.imshow('blurImage', blurImage)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


























