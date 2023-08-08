# -*- coding: utf-8 -*-
# @Time : 2023/8/3 10:54
# @Author : zihua.zeng
# @File : test_edz.py


import os
import cv2
import random
import string
import math
import numpy as np
from glob import glob
from PIL import Image, ImageDraw, ImageFont


class WMGenerator:
    def __init__(self, image_path, logo_path, font_path):
        self.image_files = glob("%s/*.jpg" % image_path) + glob("%s/*.png" % image_path)
        self.logo_files = glob("%s/**/*.png" % logo_path) + glob("%s/**/*.jpg" % logo_path)
        self.font_files = glob("%s/*.ttf" % font_path)

    def __call__(self):
        image = cv2.imread(random.sample(self.image_files, 1)[0])
        logo = False if random.sample([True, False], 1)[0] else cv2.imread(random.sample(self.logo_files, 1)[0],
                                                                           cv2.IMREAD_UNCHANGED)
        if isinstance(logo, np.ndarray) and logo.shape[2] == 3:
            logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
        text = self.__random_text_wm()

        mask = None
        if not isinstance(logo, np.ndarray):
            mask = text
        else:
            scale = logo.shape[0] / text.shape[0]
            text = cv2.resize(text, (int(scale * text.shape[1]), logo.shape[0]))
            mask = np.hstack([logo, text])

        # 获取图像尺寸和中心坐标
        height, width = mask.shape[:2]
        center_x, center_y = width // 2, height // 2
        # 随机选择旋转角度
        rotation_angle = np.random.uniform(-30, 30)
        # 根据角度，计算宽高变化
        rotation_angle_radians = math.radians(rotation_angle)
        rotated_width = int(
            abs(width * math.cos(rotation_angle_radians)) + abs(height * math.sin(rotation_angle_radians)))
        rotated_height = int(
            abs(width * math.sin(rotation_angle_radians)) + abs(height * math.cos(rotation_angle_radians)))
        # 定义旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
        # 防止图片被裁剪
        rotation_matrix[0, 2] += (rotated_width - width) / 2
        rotation_matrix[1, 2] += (rotated_height - height) / 2
        # 进行图像旋转
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (rotated_width, rotated_height))

        rep = random.randint(1, 10)
        final_wm = np.hstack([rotated_mask] * rep)
        final_wm = np.vstack([final_wm] * rep)

        res_wm = cv2.resize(final_wm, (image.shape[1], image.shape[0]))
        alpha = res_wm[:, :, 3] / 255.
        alpha = alpha[:, :, np.newaxis]
        transparent_ratio = random.uniform(0, 1)
        watermarked_image = (1 - alpha) * image + (
                transparent_ratio * alpha * res_wm[:, :, :3] + (1 - transparent_ratio) * alpha * image)
        watermarked_image = watermarked_image.astype(np.uint8)
        return watermarked_image

    def __random_text_wm(self):
        # 设置参数
        text = self.__generate_random_string()  # 要显示的文本
        font_path = random.sample(self.font_files, 1)[0]  # 替换为您的字体文件路径
        font_size = 88
        text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)  # 文本颜色，(R, G, B, A)
        # 一个字符的宽为 52
        image_size = (len(text) * 52, 150)  # 图像尺寸 (宽度, 高度)
        # 创建一个RGBA模式的PIL图像，带有透明背景
        image = Image.new('RGBA', image_size, (255, 255, 255, 0))
        # 加载指定字体
        font = ImageFont.truetype(font_path, font_size)
        # 获取绘图对象
        draw = ImageDraw.Draw(image)
        # 在图像上绘制文本
        draw.text((0, 20), text, fill=text_color, font=font)
        # 将PIL图像转换为OpenCV图像
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
        return cv_image

    def __generate_random_string(self, min_l=5, max_l=15):
        # 生成随机长度，范围在5到15之间
        length = random.randint(min_l, max_l)
        # 生成所有可能的字母，暂时不包括符号  + string.punctuation 为符号
        all_chars = string.ascii_letters
        # 随机选择length个字符
        random_string = ''.join(random.choice(all_chars) for _ in range(length))
        return random_string


if __name__ == '__main__':
    font_path = "fonts_ttf"
    image_path = "/Users/zihua.zeng/Dataset/white_0720"
    logo_path = "/Users/zihua.zeng/Dataset/logo_watermark_ds/redirect_logo"
    out_path = "generated_wm"
    os.makedirs(out_path, exist_ok=True)
    generator = WMGenerator(image_path, logo_path, font_path)

    for _ in range(100):
        wm = generator()
        cv2.imwrite(os.path.join(out_path, "%d.jpg" % _), wm)
