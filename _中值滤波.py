import cv2
import numpy as np

def medianBlur(image, winSize):
    rows, cols = image.shape
    WinH, WinW = winSize
    halfWinH = int((WinH - 1) / 2)
    halfWinW = int((WinW - 1) / 2)

    # 中值滤波后的输出图像
    medianBlurImage = np.zeros(image.shape, image.dtype)
    for r in range(rows):
        for c in range(cols):
            # 判断边界
            rTop = 0 if r - halfWinH < 0 else r - halfWinH
            rBottom = rows - 1 if r + halfWinH > rows - 1 else r + halfWinH
            cLeft = 0 if c - halfWinH < 0 else c - halfWinW
            cRight = cols - 1 if c + halfWinW > cols - 1 else c + halfWinW
            # 取邻域
            region = image[rTop: rBottom + 1, cLeft: cRight + 1]
            # 取中值
            medianBlurImage[r][c] = np.median(region)
    return medianBlurImage

if __name__ == '__main__':
    image = cv2.imread('img/girl.ppm', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('img/girl.ppm')
    # 显示原图
    cv2.imshow('image', image)
    # 中值滤波
    medianBlurImage = medianBlur(image, (3, 3))

    # opencv 实现的中值滤波函数
    blur = cv2.medianBlur(img, 3)
    cv2.imshow('blur', blur)

    # 显示中值滤波的结果
    cv2.imshow('medianBlurImage', medianBlurImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


























