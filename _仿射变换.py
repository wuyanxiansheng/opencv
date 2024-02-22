import cv2
import numpy as np

mat = cv2.imread('img/girl.ppm')

'''
split = cv2.split(mat)

# cv2.imshow('B', split[0])
# cv2.imshow('G', split[1])
# cv2.imshow('R', split[2])


# for i in range(mat.shape[0]):
#     for j in range(mat.shape[1]):
#         print(mat[i][j][0])
#         print(mat[i][j][1])
#         print(mat[i][j][2])

B = mat[:, :, 0]
G = mat[:, :, 1]
R = mat[:, :, 2]


cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)

# cv2.imshow('img', mat)
cv2.waitKey(0)
'''
# 仿射变换：平移 缩小 放大 旋转
image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)

# 获取原图的高、宽
h, w = image.shape[:2]
# 仿射变换矩阵，缩小2倍
A1 = np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32)
d1 = cv2.warpAffine(image, A1, (w, h), borderValue=125)
cv2.imshow('d1', d1)

# 先缩小2倍，再平移
A2 = np.array([[0.5, 0, w / 4], [0, 0.5, h / 4]], np.float32)
d2 = cv2.warpAffine(image, A2, (w, h), borderValue=125)
cv2.imshow('d2', d2)

# 在 d2 的基础上，绕图像的中心点旋转
A3 = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 90, 1)
d3 = cv2.warpAffine(d2, A3, (w, h), borderValue=125)
cv2.imshow('d3', d3)

cv2.imshow('image', image)

# 旋转函数 rotate（OpenCV3.0 新特性） rotate(src, rotateCode)：
# ROTATE_90_CLOCKWISE：顺时针旋转90度
# ROTATE_180：顺时针旋转180度
# ROTATE_90_COUNTERCLOCKWISE：顺时针旋转270度

rImg = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('rImg', rImg)


cv2.waitKey(0)
cv2.destroyAllWindows()






