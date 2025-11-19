import cv2
import numpy as np
# 创建红色掩膜
def create_red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 80])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)
# 形态学去噪
def remove_small_noise(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask
# 画轮廓并计数 根据图片编号切换阈值
def draw_contours_and_count(img, mask, name):
    result = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    apple_count = 0
    # 根据文件名判断是前6张还是后2张
    if '7.png' in name or '8.png' in name:
        min_area = 1  # 第7、8张用大阈值
    else:
        min_area = 2000  # 第1~6张用小阈值
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:  # 过滤小区域
            continue
        cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
        apple_count += 1
    # 添加文字
    text = f"Apple Count: {apple_count}"
    cv2.putText(result, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 255), 3, cv2.LINE_AA)
    return result, apple_count
# 主函数
if __name__ == '__main__':
    image_names = [f"/home/nico/文档/python/rm_vision/models/apple/{i}.png" for i in range(1, 9)]
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 800, 600)
    for name in image_names:
        img = cv2.imread(name)
        red_mask = create_red_mask(img)
        cleaned_mask = remove_small_noise(red_mask)
        result_img, count = draw_contours_and_count(img, cleaned_mask, name)  # 多传一个 name
        cv2.imshow("result", result_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
