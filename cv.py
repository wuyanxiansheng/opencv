import cv2
import numpy as np
from scipy import signal
import math

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


#多尺度Retinex
def singleScaleRetinexTemp(img, sigma):
    # 按照公式计算
    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(_temp == 0, 0.001, _temp)
    retinex = np.log10(img + 0.01) - np.log10(gaussian)

    return retinex

def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img * 1.0)
    for sigma in sigma_list:
        print("sigma:", sigma)
        retinex += singleScaleRetinexTemp(img, sigma)
    img_msr = retinex / len(sigma_list)

    img_msr[:, :] = (img_msr[:, :] - np.min(img_msr[:, :])) / \
                       (np.max(img_msr[:, :]) - np.min(img_msr[:, :])) * 255
    img_msr = np.uint8(np.minimum(np.maximum(img_msr, 0), 255))
    return img_msr


#最大值滤波
def maxBlur(image, kernel, limit):
    """
    Parameters
    ----------
    image : array, 输入矩阵或数组.
    kernel : tuple or list, optional
        分别为x、y方向上的最大值取值区间范围. The default is (3, 3).
    limit : tuple or list, optional
        指定进行最大值滤波的像素范围. The default is (0, 255).
    Returns
    -------
    image_c : array，处理后矩阵或数组。
    """
    image_c = image.copy()
    if len(image_c.shape) == 2:
        image_c = image_c[:, :, np.newaxis]
    h, w, c = image_c.shape
    image_c1 = image_c.copy()
    for i in range(h):
        for j in range(w):
            x1 = max(j-kernel[0]//2, 0)
            x2 = min(x1 + kernel[0], w)
            y1 = max(i-kernel[1]//2, 0)
            y2 = min(y1 + kernel[1], h)
            for k in range(c):
                if image_c[i, j, k] >= limit[0] and image_c[i, j, k] <= limit[1]:
                    sub_img = image_c1[y1:y2, x1:x2, k]
                    image_c[i, j, k] = np.max(sub_img)
    if len(image.shape) == 2:
        image_c = image_c.reshape(h, w)
    return image_c


def max_value_filter(V):
    # 将RGB图像转换为HSV图像

    #对v进行高斯滤波
    V_l = cv2.GaussianBlur(V, (0, 0), 3)
    V_r = multiScaleRetinex(V, sigma_list=[15,80,200])

    cv2.imshow('V_r',V_r)
    cv2.imshow('V_L', V_l)


    #根据l求出r
    #V_r_new = 255 * np.log10(np.true_divide(V, V_l + 1e-5))
    #V_r_new = cv2.normalize(V_r_new, 0, 255, cv2.NORM_MINMAX)
    #V_r_new = np.log10(V + 0.01) - np.log10(V_l + 0.01)
    #V_r_new = cv2.normalize(V_r_new, 0, 255, cv2.NORM_MINMAX)
    V_l_new = maxBlur(V_l, (5, 5), (0, 255))
    cv2.imshow('V_L_new', V_l_new)
    V_r = np.clip(V_r, 0, 255)
    V_r = np.array(V_r, dtype=np.uint8)
    #V_r_new = 100 * V_r_new + 120


    return V_r, V_l_new

if __name__ == '__main__':
    # 读取RGB图像
    img = cv2.imread('img/girl.ppm')
    # 设置最大值滤波的窗口大小
    #filter_size = 5
    # 调用最大值滤波函数
    img = replaceZeroes(img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 分割H、S、V三个分量
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]


    V_r , V_l_new = max_value_filter(v)
    V_new = V_r * V_l_new

    # V_new = np.log10(V_r) + np.log10(V_l_new)

    # V_new = np.uint8(np.minimum(np.maximum(V_new, 0), 255))    # V_new = np.array(V_new, dtype=np.uint8)

    ########################################

    hsv_msr = np.zeros(hsv_img.shape, dtype=np.uint8)
    hsv_msr[:, :, 0] = h  # H通道不变
    hsv_msr[:, :, 1] = s  # S通道不变
    hsv_msr[:, :, 2] = V_l_new
    new_msr = cv2.cvtColor(hsv_msr, cv2.COLOR_HSV2BGR)

    #########################################
    hsv_msr1 = np.zeros(hsv_img.shape, dtype=np.uint8)
    hsv_msr1[:, :, 0] = h  # H通道不变
    hsv_msr1[:, :, 1] = s  # S通道不变
    hsv_msr1[:, :, 2] = V_new
    new_msr1 = cv2.cvtColor(hsv_msr1, cv2.COLOR_HSV2BGR)


    #########################################
    V_max = maxBlur(v, (5, 5), (0, 255))
    hsv_max = np.zeros(hsv_img.shape, dtype=np.uint8)
    hsv_max[:, :, 0] = h  # H通道不变
    hsv_max[:, :, 1] = s  # S通道不变
    hsv_max[:, :, 2] = V_max
    new_max = cv2.cvtColor(hsv_max, cv2.COLOR_HSV2BGR)
    #cv2.imshow('new_max', new_max)



    # 显示原始图像和滤波后的图像
    cv2.imshow('Original Image', img)
    #cv2.imshow('V_r_new', V_r_new)
    #cv2.imshow('ssr', new_msr)
    cv2.imshow('maxfilter', new_msr1)

    cv2.imshow('v_new', V_l_new)
    cv2.imshow('v', v)
    #cv2.imwrite("D:/testpicture/result_image/maxblur15-0.ppm", new_msr1, [int(cv2.IMWRITE_PXM_BINARY), 0])
    #cv2.imwrite("D:/testpicture/result_image/maxblur15-1.ppm", new_msr1, [int(cv2.IMWRITE_PXM_BINARY), 1])
    #cv2.imshow('h', h)
    #cv2.imshow('s', s)
    cv2.waitKey(0)
    cv2.destroyAllWindows()