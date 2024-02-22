import cv2
import numpy as np
from scipy import signal


def prewitt(I, _boundary='symm'):
    # prewitt 是可分离卷积的，根据卷积运算的结合律，分两次小卷积核运算
    # 1. 垂直方向上的均值平滑
    ones_y = np.array([[1], [1], [1]], np.float32)
    i_conv_pre_x = signal.convolve2d(I, ones_y, mode='same', boundary=_boundary)
    # 2. 水平方向上的差分
    diff_x = np.array([[1, 0, -1]], np.float32)
    i_conv_pre_x = signal.convolve2d(i_conv_pre_x, diff_x, mode='same', boundary=_boundary)
    
    # prewitt_y 是可分离的，根据卷积运算的结合律，分两次小卷积核运算
    # 1. 水平方向上的均值平滑
    ones_x = np.array([[1, 1, 1]], np.float32)
    i_conv_pre_y = signal.convolve2d(I, ones_x, mode='same', boundary=_boundary)
    # 2. 垂直方向上的差分
    diff_y = np.array([[1], [0], [-1]], np.float32)
    i_conv_pre_y = signal.convolve2d(i_conv_pre_y, diff_y, mode='same', boundary=_boundary)
    return (i_conv_pre_x, i_conv_pre_y)

if __name__ == '__main__':
    image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)
    i_conv_pre_x, i_conv_pre_y = prewitt(image)
    # 取绝对值，分别得到水平方向和垂直方向上的边缘强度
    abs_i_conv_pre_x = np.abs(i_conv_pre_x)
    abs_i_conv_pre_y = np.abs(i_conv_pre_y)
    # 水平方向和垂直方向上的边缘强度的灰度级显示
    edge_x = abs_i_conv_pre_x.copy()
    edge_y = abs_i_conv_pre_y.copy()
    # 将大于255的值截断为 255
    edge_x[edge_x > 255] = 255
    edge_y[edge_y > 255] = 255
    # 数据类型转换
    edge_x = edge_x.astype(np.uint8)
    edge_y = edge_y.astype(np.uint8)
    cv2.imshow('edge_x', edge_x)
    cv2.imshow('edge_y', edge_y)
    # 利用 abs_i_conv_pre_x 和 abs_i_conv_pre_y
    # 求边缘强度，有多种方式，这里使用插值法
    edge = 0.5 * abs_i_conv_pre_x + 0.5 * abs_i_conv_pre_y
    edge[edge > 255] = 255
    edge = edge.astype(np.uint8)
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




























    