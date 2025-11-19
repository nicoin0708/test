import cv2
import numpy as np
import math

def darker(img):
    """降低亮度，辅助函数"""
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换为 HSV 颜色空间
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * 0.5  # 将 V 通道乘以 0.5
    darker_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # 转换回 BGR
    return darker_image

def process_img(img, val):
    """处理图像，返回二值图像、调整大小和亮度的图像"""    
    img_resized = cv2.resize(img, (640, 480))  # 调整图像大小
    img_dark = cv2.convertScaleAbs(img_resized, alpha=0.5)  # 调整亮度
    img_darker = darker(img_dark)  # 降低亮度
    img_gray = cv2.cvtColor(img_darker, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    _, img_binary = cv2.threshold(img_gray, val, 255, cv2.THRESH_BINARY)  # 二值化处理
    #cv2.imshow("Binary Image", img_binary)  # 显示二值图像
    return img_dark, img_binary 

def adjust(rect):
    c, (w, h), angle = rect
    if w > h:
        w, h = h, w
        angle = (angle + 90) % 360
        angle = angle - 360 if angle > 180 else angle - 180 if angle > 90 else angle
    return c, (w, h), angle

def find_light(resized_img, binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        area = cv2.contourArea(contour)
        if area < 5:
            continue
        rect = adjust(rect)
        if -35 < rect[2] < 35:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rects.append(rect)
            cv2.drawContours(resized_img, [box], 0, (0, 255, 0), 2)
    #cv2.imshow('Detected Rotated Rectangles', resized_img)
    return resized_img, rects

def is_close(rect1, rect2, light_angle_tol, line_angle_tol, height_tol, width_tol, cy_tol):
    (cx1, cy1), (w1, h1), angle1 = rect1
    (cx2, cy2), (w2, h2), angle2 = rect2
    distance = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    if distance > 20:
        angle_diff = min(abs(angle1 - angle2), 360 - abs(angle1 - angle2))
        if angle_diff <= light_angle_tol:
            if abs(h1 - h2) <= height_tol and abs(w1 - w2) <= width_tol:
                line_angle = math.degrees(math.atan2(cy2 - cy1, cx2 - cx1))
                if line_angle > 90:
                    line_angle -= 180
                elif line_angle < -90:
                    line_angle += 180
                if (abs(line_angle - angle1) <= line_angle_tol or abs(line_angle - angle2) <= line_angle_tol or abs(cy1 - cy2) < cy_tol):
                    return True
    return False

def is_armor(img, lights, light_angle_tol=5, line_angle_tol=7, height_tol=10, width_tol=10, cy_tol=5):
    lights_matched = []
    processed_indices = set()
    lights_count = len(lights)
    for i in range(lights_count):
        if i in processed_indices:
            continue
        light1 = lights[i]
        close_lights = [j for j in range(lights_count) if j != i and is_close(light1, lights[j], light_angle_tol, line_angle_tol, height_tol, width_tol, cy_tol)]
        if close_lights:
            group = [light1] + [lights[j] for j in close_lights]
            lights_matched.append(group)
            processed_indices.update([i] + close_lights)
    armors = []
    for light_matched in lights_matched:
        if light_matched:
            points = np.concatenate([cv2.boxPoints(light) for light in light_matched])
            armor_raw = cv2.minAreaRect(points)
            if 200 <= armor_raw[1][0] * armor_raw[1][1] <= 11000:
                armor_flit = adjust(armor_raw)
                if 1 <= armor_flit[1][1] / armor_flit[1][0] <= 3.5:
                    armors.append(adjust(armor_flit))
    armors_center = []
    for armor in armors:#画armor
        center, (width, height), angle = armor
        max_size = max(width, height)
        box = cv2.boxPoints(((center[0], center[1]), (max_size, max_size), angle)).astype(int)
        cv2.drawContours(img, [box], 0, (255, 0, 255), 2)
        cv2.circle(img, (int(center[0]), int(center[1])), 5, (255, 0, 255), -1)
        (center_x, center_y) = map(int, armor[0])
        cv2.putText(img, f"({center_x}, {center_y})", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 255), 1)  # 在图像上标记坐标
        armor_center = (center_x, center_y)
        armors_center.append(armor_center)
        #cv2.imshow("armor", img)
    return armors_center
def detector(img_raw):
    resized_img,binary_img = process_img(img_raw, 35)#返回binary
    # 找灯条
    img_light = resized_img.copy()
    img_light, rects = find_light(img_light, binary_img)
    # 找装甲板
    img_armor = resized_img.copy()
    armors_center = is_armor(img_armor, rects)
    cv2.imshow("ARMOR", img_armor)
    return armors_center
if __name__ == "__main__" :
    img_paths = [f"/home/nico/文档/python/rm_vision/models/armor/{i}.png" for i in range(1, 13)]
    for name in img_paths:
        img = cv2.imread(name)
        armors_center = detector(img)
        #cv2.imshow("armor", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
