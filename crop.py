import cv2
import numpy as np

def crop_from_non_black_pixel(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像的形状
    height, width = gray.shape

    # 初始化裁剪边界
    top, bottom, left, right = height, 0, width, 0

    # 逐个像素检测非黑色像素
    for y in range(height):
        for x in range(width):
            if gray[y, x] > 0:
                top = min(top, y)
                bottom = max(bottom, y)
                left = min(left, x)
                right = max(right, x)

    # 裁剪图像
    cropped_img = img[top:bottom+1, left:right+1]

    return cropped_img

# 输入图像路径
image_path = 'green_parts.jpg'

# 调用函数裁剪图像
cropped_image = crop_from_non_black_pixel(image_path)

# 显示原始图像和裁剪后的图像
cv2.imshow('Original Image', cv2.imread(image_path))
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('croped.jpg', cropped_image)
