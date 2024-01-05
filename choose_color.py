
# import cv2
# import numpy as np

# # 读取图像 a 和 b
# image_a = cv2.imread('./shape1.png')
# image_b = cv2.imread('./green_parts.jpg')

# # 将图像 a 转换为灰度
# gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)

# # 进行边缘检测并寻找轮廓
# edges_a = cv2.Canny(gray_a, 50, 150)
# contours, _ = cv2.findContours(edges_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 打印轮廓数量
# print("Number of contours:", len(contours))
# # 创建一个全黑的图像，与图像 b 具有相同的大小和通道数
# mask = np.zeros_like(image_b).astype('uint8')

# # 在掩码上绘制与图像 a 相同的轮廓
# # contours = np.clip(contours, 0, 255).astype(np.uint8)
# cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=-1)

# # cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# # 在图像 b 上应用掩码
# result_image = cv2.bitwise_and(image_b, mask)
# print(result_image.shape)

# # 显示结果
# cv2.imshow('Result', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # 读取图像和模板
# image = cv2.imread('green_parts.jpg')
# template = cv2.imread('want2.jpg')

# # 使用模板匹配
# result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# # 设置阈值，找到匹配位置
# threshold = 0.5
# loc = np.where(result >= threshold)

# # 在所有匹配位置绘制矩形框
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (255, 0, 0), 2)

# # 显示结果
# cv2.imshow('Matching Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ret,thresh = cv2.threshold(gray,50,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# print("Number of contours detected:", len(contours))

# for cnt in contours:
#    x1,y1 = cnt[0][0]
#    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#    if len(approx) == 4:
#       x, y, w, h = cv2.boundingRect(cnt)
#       ratio = float(w)/h
#       if ratio >= 0.9 and ratio <= 1.1:
#          image = cv2.drawContours(image, [cnt], -1, (0,255,255), 3)
#          cv2.putText(image, 'rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#       else:
#          cv2.putText(image, 'squere', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#          image= cv2.drawContours(image, [cnt], -1, (0,255,0), 3)

# cv2.imshow("形状", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np
import os
import shutil
from PIL import Image, ImageEnhance, ImageFilter

# 初始化全局变量
dragging = False
refPt = (0, 0)
tx, ty = 0, 0
scale = 1

def click_event(event, x, y, flags, param):
    global dragging, refPt, tx, ty, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        refPt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # 更新平移变量
        dx, dy = x - refPt[0], y - refPt[1]
        tx += dx
        ty += dy
        refPt = (x, y)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 调整坐标以获取正确的颜色值
        adjusted_x, adjusted_y = int((x - tx) / scale), int((y - ty) / scale)
        if 0 <= adjusted_x < image.shape[1] and 0 <= adjusted_y < image.shape[0]:
            colorsBGR = image[adjusted_y, adjusted_x]
            colorsHSV = cv2.cvtColor(np.uint8([[colorsBGR]]), cv2.COLOR_BGR2HSV)[0][0]
            print(f"HSV Values at ({adjusted_x}, {adjusted_y}): {colorsHSV} BGR values is {colorsBGR}")

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scale += 0.1
        else:
            scale = max(0.1, scale - 0.1)
        tx, ty = 0, 0  # 重置平移



def pickcolor():
    global image
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', click_event)

    while True:
        # 根据缩放和平移变量调整图片
        resized_image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(resized_image, M, (resized_image.shape[1], resized_image.shape[0]))

        cv2.imshow('Image', translated_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # cv2.destroyAllWindows()

def crop(path):


    # 加载图像
    image = cv2.imread(path)
    if image is None:
        print("图像未加载。请检查路径。", path)
        return
    # 将图像从BGR转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义绿色的HSV阈值范围
    # 注意：这些值可能需要根据您的具体图像进行调整
    green_lower = np.array([55, 255, 255])
    green_upper = np.array([65, 255, 255])

    # 创建一个只包含绿色的掩码
    mask = cv2.inRange(hsv, green_lower, green_upper)

    # 查找掩码中的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 假设最大的轮廓是绿色框
    # 注意：这个假设在有多个绿色区域的图像中可能不成立
    c = max(contours, key=cv2.contourArea)

    # 获取绿色框的边界
    x, y, w, h = cv2.boundingRect(c)

    # 裁剪图像
    cropped = image[y:y+h, x:x+w]

    # 保存或显示裁剪的图像
    out_path = os.path.join(os.path.dirname(path), 'crop')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_name = os.path.join(out_path, os.path.basename(path))
    divide(cropped, out_name)
    # cv2.imwrite(out_name, cropped)
    # cv2.imshow('Cropped Image', cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def divide(image, outpath):
    img_large = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    lw, lh = img_large.shape[0], img_large.shape[1]

    width = 4000
    height = 6000
    idx = 0
    for i in range(0, lw, width):
        for j in range(0, lh, height):
            ri, rj = min(i + width, lw), min(j+ height, lh)

            img_cropped = img_large[i:ri+1, j:rj+1]
            out_name = outpath[:-4] + "_" + str(idx) + ".png"
            cv2.imwrite(out_name, img_cropped)
            idx += 1
    # cv2.imshow("aaa", img_cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def preprocess(image_path):
    image = cv2.imread(image_path)
    print(image.shape)
    # unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    # print(image_path, unique_colors.shape)
    if image is None:
        print("Error: Unable to open image.")
        exit()


    # # 显示图像
    # cv2.imshow('Unique Colors', color_palette)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # pickcolor()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # select green parts
    # lower_color = np.array([60, 0, 255])
    # upper_color = np.array([60, 255, 255])
    # 定义的HSV范围 text
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([0, 0, 100])

    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    result = cv2.bitwise_and(image, image, mask=mask)

    out_path = "../images/text/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_name = out_path + os.path.basename(image_path)
    print(out_name)
    image = Image.fromarray(result)

    # 提高对比度，这里使用2作为增强因子，这个值可以根据需要调整
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(3)

    # 对提高了对比度的图片应用锐化滤镜
    image_sharpened = image_enhanced.filter(ImageFilter.SHARPEN)
    # 将图片转换为黑白，使用一个阈值将灰度图片二值化以提高文本的可读性
    # 默认阈值设置为128，这个值也可以根据实际图片的情况进行调整
    image_bw = image_sharpened.convert('L').point(lambda x: 0 if x < 20 else 255, '1')

    # 保存增强后的图片
    image_bw.save(out_name)
    # cv2.imwrite(out_name, result)

# def preprocess2():
# # 转换为灰度图
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # 应用阈值处理。如果灰度值大于某个值，设置为黑色，否则设置为白色
    # # 这里的阈值可以调整，以找到最佳的分割效果
    # _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)

    # # 保存或显示图像
    # cv2.imwrite('black_background_image.png', binary_image)


def main():
    test_data_path = "../images/need/"
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for filename in os.listdir(test_data_path):
        for ext in exts:
            if filename.endswith(ext):
                files.append(test_data_path + filename)
                break
    # print(files)
    target_color = [60, 255, 255]
    global image
    image = cv2.imread('D:/wzk/images/crop/2_5.png')


    # pickcolor()
    for image_path in files:
        """
            rename files
        """
        # last = image_path.split("/")[-1]
        # if last[0] == "视":
        #     print(image_path)
        #     new_name = last[8:].replace(")", "", 1)
        #     os.rename(image_path, test_data_path + new_name)


        """
            pick color to crop
        """
        # global image
        # image = cv2.imread(image_path)
        # if image is None:
        #     print("图像未加载。请检查路径。")
        # cv2.imshow('choosen Parts', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # pickcolor()
        # print("over")
        
        """
            crop image
        """
        # crop(image_path)


        """
            choose color mask
        """
        preprocess(image_path)
        # break

    # dir = "../images/preprocess_device/"
    # outdir = "../images/mydata/images/"
    # need = "../images/mydata/labels"
    # for doc in os.listdir(need):
    #     doc = doc[:-4] + ".png"
    #     for ext in exts:
    #         if doc.endswith(ext):
    #             outname = outdir + doc[:-4] + "_mask" + ".png"
    #             print(outname)
    #             shutil.copy(dir + doc, outname)

    # dir = "../images/mydata/labels/"
    # print(len(os.listdir(dir)))
    # for doc in os.listdir(dir):
    #     if doc.endswith('txt'):
    #         outname = dir + doc[:-4] + "_mask" + ".txt"
    #         print(outname)
    #         shutil.copy(dir + doc, outname)

        # break


    

if __name__ == "__main__":
    main()


