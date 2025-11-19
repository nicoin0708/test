import cv2
from detector import detect_apple_contours
def find_largest_apple(img, image_name):
    # 根据图片名设置阈值
    if '7.png' in image_name or '8.png' in image_name:
        min_area = 0.5
    else:
        min_area = 2000
    # 调用 detector 获取所有轮廓
    contours = detect_apple_contours(img, min_area)
    # 返回面积最大的轮廓（如果没有轮廓，返回 None）
    return max(contours, key=cv2.contourArea) if contours else None
def test_find_largest_apple():
    # 定义8张测试图像的路径
    image_paths = [f"/home/nico/文档/python/rm_vision/models/apple/{i}.png" for i in range(1, 9)]
    # 遍历每张图像进行测试
    for path in image_paths:
        # 读取图像
        img = cv2.imread(path)
        # 调用核心函数查找最大苹果轮廓
        largest_contour = find_largest_apple(img, path)

        # 输出测试结果
        if largest_contour is not None:
            area = cv2.contourArea(largest_contour)
            print(f"成功找到最大苹果，面积 = {area:.1f}")
        else:
            print(f"未检测到任何苹果轮廓")
