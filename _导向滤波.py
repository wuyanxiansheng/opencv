import cv2
import numpy as np
import _均值滤波 as meanFilter

# 导向滤波
def guidedFilter(I, p, winSize, eps):
    # 输入图像的高、宽
    rows, cols = I.shape
    # I 的均值平滑
    mean_I = meanFilter.fastMeanBlur(I, winSize, cv2.BORDER_DEFAULT)
    # P 的均值平滑
    mean_P = meanFilter.fastMeanBlur(p, winSize, cv2.BORDER_DEFAULT)
    # I * p 的均值平滑
    Ip = I * p
    mean_Ip = meanFilter.fastMeanBlur(Ip, winSize, cv2.BORDER_DEFAULT)
    # 协方差
    cov_Ip = mean_Ip - mean_I * mean_P
    mean_II = meanFilter.fastMeanBlur(I * I, winSize, cv2.BORDER_DEFAULT)
    # 方差
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_P - a * mean_I
    # 对 a 和 b 进行均值平滑
    mean_a = meanFilter.fastMeanBlur(a, winSize, cv2.BORDER_DEFAULT)
    mean_b = meanFilter.fastMeanBlur(b, winSize, cv2.BORDER_DEFAULT)
    q = mean_a * I + mean_b
    return q

if __name__ == '__main__':
    image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('img/girl.ppm')
    # 将图像归一化
    image_0_1 = image / 255.0
    # 显示原图
    cv2.imshow('image', image)
    # 导向滤波
    result = guidedFilter(image_0_1, image_0_1, (17, 17), pow(0.2, 2.0))
    # src：输入图像，可以是灰度图像，也可以是多通道的彩色图像
    # guide：导向图像，大小和类型与 src 相同
    # dst：输出图像，大小和类型与 src 相同
    # d：滤波核的像素邻域直径
    # eps：规范化参数， eps 的平方类似于双边滤波中的 sigmaColor
    # dDepth：输出图片的数据深度

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    result[result > 255] = 255
    result = np.round(result * 255)
    result = result.astype(np.uint8)
    for i in range(85,89):
        for j in range(85, 89):
            print(result[i][j])

    print()
    guided_filter_img = cv2.ximgproc.guidedFilter(image, result, 7, 0.01)
    for i in range(85,89):
        for j in range(85, 89):
            print(guided_filter_img[i][j])


    cv2.imshow('guidedFilter', result)
    cv2.imshow('gray', gray)
    cv2.imshow('guidedFilterImg', guided_filter_img)
    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





















