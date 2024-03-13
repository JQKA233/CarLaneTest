import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import Counter


def denoise(img):
    # Method 1
    filtered_img = cv2.edgePreservingFilter(img, flags=1, sigma_s=200, sigma_r=0.4)

    # # Method 2
    # # 创建引导滤波器
    # radius = 8
    # eps = 0.01
    # guided_filter = cv2.ximgproc.createGuidedFilter(guide=img, radius=radius, eps=eps)
    #
    # # 应用引导滤波器
    # filtered_img = guided_filter.filter(img)

    # 使用中值模糊滤波器去除椒盐噪声
    img = cv2.medianBlur(filtered_img, 7)

    # # # 使用中值模糊滤波器去除椒盐噪声
    # img = cv2.medianBlur(img, 3)

    # 定义膨胀核（膨胀也有滤波的效果）
    # kernel = np.ones((3, 3), np.uint8)
    # 椭圆核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 膨胀操作
    img = cv2.dilate(img, kernel, iterations=1)

    # 显示图像
    # 转换回 RGB 图像
    plot = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用 matplotlib 显示图像
    plt.imshow(plot)
    plt.show()

    return img


def get_poly_dp(img_path: str, with_noise: bool = False):
    # 使用 collection 中的 Counter 记录结果
    result = Counter()
    # 读取图像
    img = cv2.imread(str(img_path))

    if with_noise:
        img = denoise(img)

    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 30, 90)

    # 定义结构元素
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


    # 闭运算
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 展示运算后的轮廓图
    plt.imshow(edges)
    plt.show()

    # 轮廓提取
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)

    # 多边形逼近
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.025 * cv2.arcLength(cnt, True), True)
        # 判断多边形边数，至少为 3 条边才能是多边形，并且多边形中每条边的长度都大于5
        if len(approx) >= 3 and all(cv2.norm(approx[i] - approx[(i+1) % len(approx)]) >= 5 for i in range(len(approx))):
            # 在图中绘制多边形边缘【绿色】
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
            # 输出多边形边数
            print(len(approx))
            # 在结果中记录
            result[str(len(approx))] += 1
            # 绘制多边形顶点【红色】 帮助 Debug
            for i in range(len(approx)):
                # 获取顶点坐标
                x, y = approx[i][0]
                # 在原图像中绘制圆点标记
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    # 显示图像
    # 转换回 RGB 图像
    plot = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用 matplotlib 显示图像
    plt.imshow(plot)
    plt.show()

    # 展示并返回结果
    print(result)
    return result


class TestClass:
    answer_sheet = [
        Counter({'6': 2, '7': 1, '8': 2, '10': 2}),
        Counter({'6': 0, '7': 3, '8': 2, '10': 2}),
        Counter({'6': 1, '7': 3, '8': 5, '10': 1}),
        Counter({'6': 1, '7': 2, '8': 2, '10': 2})
    ]

    def test_original_1(self):
        imgPath = Path.cwd() / 'data' / f'O1.jpg'
        assert get_poly_dp(imgPath) == self.answer_sheet[0]

    def test_original_2(self):
        imgPath = Path.cwd() / 'data' / f'O2.jpg'
        assert get_poly_dp(imgPath) == self.answer_sheet[1]

    def test_original_3(self):
        imgPath = Path.cwd() / 'data' / f'O3.jpg'
        assert get_poly_dp(imgPath) == self.answer_sheet[2]

    def test_original_4(self):
        imgPath = Path.cwd() / 'data' / f'O4.jpg'
        assert get_poly_dp(imgPath) == self.answer_sheet[3]


class TestClassWithNoise:
    answer_sheet_with_noise = [
        Counter({'6': 2, '7': 3, '8': 1, '10': 2}),
        Counter({'6': 2, '7': 2, '8': 4, '10': 4}),
        Counter({'6': 2, '7': 2, '8': 3, '10': 1}),
        Counter({'6': 2, '7': 3, '8': 1, '10': 1})
    ]

    def test_with_noise_1(self):
        imgPath = Path.cwd() / 'data' / f'1Q.jpg'
        assert get_poly_dp(imgPath, True) == self.answer_sheet_with_noise[0]


    def test_with_noise_2(self):
        imgPath = Path.cwd() / 'data' / f'2Q.jpg'
        assert get_poly_dp(imgPath, True) == self.answer_sheet_with_noise[1]

    def test_with_noise_3(self):
        imgPath = Path.cwd() / 'data' / f'3Q.jpg'
        assert get_poly_dp(imgPath, True) == self.answer_sheet_with_noise[2]

    def test_with_noise_4(self):
        imgPath = Path.cwd() / 'data' / f'4Q.jpg'
        assert get_poly_dp(imgPath, True) == self.answer_sheet_with_noise[3]


if __name__ == '__main__':
    # Original
    # imgPath = Path.cwd() / 'data' / 'O1.jpg'
    # imgPath = Path.cwd() / 'data' / 'O1.jpg'

    # With Noice
    imgPath = '/home/dar/Desktop/test/5Q.jpg'
    # imgPath = Path.cwd() / 'data' / '2Q.jpg'
    get_poly_dp(imgPath, True)

