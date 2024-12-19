import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from torchvision import transforms
from PIL import Image
import random
import svgwrite
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QFileDialog, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from transformers import pipeline


class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EZHoloPY 简单刮擦PY")
        self.setGeometry(100, 100, 1200, 600)

        self.img = None
        self.file_path = None
        self.pointMode = "contour"
        self.take_contour = True
        self.brightness = 200
        self.points = []
        self.depth_map_resized = None
        self.depth_map = None
        self.density = 10
        self.line_density = 10
        self.zero_depth = 128
        self.t1 = 80
        self.t2 = 180
        self.a = 0.16 # a factor for curvature
        self.model = None  # 用于存储加载的模型

        self.init_ui()

    def init_ui(self):
        # 随便吧，没怎么用过python和Qt写界面
        # 创建主布局
        main_layout = QVBoxLayout()

        # 创建图像显示区域的布局
        image_layout = QHBoxLayout()

        # 创建三个标签用于显示图像
        self.original_image_label = QLabel(self)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMaximumSize(512, 512)  # 设置最大尺寸
        image_layout.addWidget(self.original_image_label)
        

        self.depth_image_label = QLabel(self)
        self.depth_image_label.setAlignment(Qt.AlignCenter)
        self.depth_image_label.setMaximumSize(512, 512)  # 设置最大尺寸
        image_layout.addWidget(self.depth_image_label)

        self.final_image_label = QLabel(self)
        self.final_image_label.setAlignment(Qt.AlignCenter)
        self.final_image_label.setMaximumSize(512, 512)  # 设置最大尺寸
        image_layout.addWidget(self.final_image_label)

        main_layout.addLayout(image_layout)
        #main_layout.addLayout(result_layout)

        # 控制参数的滑块
        countour_control_layout = QHBoxLayout()

        # Density 滑块
        self.line_density_slider = QSlider(Qt.Horizontal)
        self.line_density_slider.setRange(3, 100)
        self.line_density_slider.setValue(self.line_density)
        self.line_density_slider.valueChanged.connect(self.update_line_density)
        self.line_density_label = QLabel("轮廓点密度0.1")
        countour_control_layout.addWidget(self.line_density_slider)
        countour_control_layout.addWidget(self.line_density_label)

        # t1 滑块
        self.t1_slider = QSlider(Qt.Horizontal)
        self.t1_slider.setRange(0, 255)
        self.t1_slider.setValue(self.t1)
        self.t1_slider.valueChanged.connect(self.update_t1)
        countour_control_layout.addWidget(self.t1_slider)
        countour_control_layout.addWidget(QLabel("t1"))

        # t2 滑块
        self.t2_slider = QSlider(Qt.Horizontal)
        self.t2_slider.setRange(0, 255)
        self.t2_slider.setValue(self.t2)
        self.t2_slider.valueChanged.connect(self.update_t2)
        countour_control_layout.addWidget(self.t2_slider)
        countour_control_layout.addWidget(QLabel("t2"))
        main_layout.addLayout(countour_control_layout)

        # 控制参数的滑块
        area_control_layout = QHBoxLayout()

        # Density 滑块
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(3, 100)
        self.density_slider.setValue(self.density)
        self.density_slider.valueChanged.connect(self.update_density)
        self.density_label = QLabel("密度0.1")
        area_control_layout.addWidget(self.density_slider)
        area_control_layout.addWidget(self.density_label)

        # brightness 滑块
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 255)
        self.brightness_slider.setValue(self.brightness)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        area_control_layout.addWidget(self.brightness_slider)
        self.brightness_label = QLabel("明暗阈值：200")
        area_control_layout.addWidget(self.brightness_label)

        main_layout.addLayout(area_control_layout)
        
        # zerodepth 滑块
        self.zerodepth_slider = QSlider(Qt.Horizontal)
        self.zerodepth_slider.setRange(0, 255)
        self.zerodepth_slider.setValue(self.brightness)
        self.zerodepth_slider.valueChanged.connect(self.update_zero_depth)
        area_control_layout.addWidget(self.zerodepth_slider)
        self.zerodepth_label = QLabel("零深度：128")
        area_control_layout.addWidget(self.zerodepth_label)

        # 添加按钮选择文件和开始处理
        button_layout = QHBoxLayout()

        # 打开文件按钮
        self.open_button = QPushButton("选择图片")
        self.open_button.clicked.connect(self.open_image)
        button_layout.addWidget(self.open_button)

        # 处理按钮
        self.process_button = QPushButton("开始处理")
        self.process_button.clicked.connect(self.process_image)
        button_layout.addWidget(self.process_button)

        # 加载模型按钮
        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.clicked.connect(self.toggle_load_model)
        button_layout.addWidget(self.load_model_button)

        # 取点模式切换按钮
        self.toggle_pointMode_button = QPushButton("当前模式：仅轮廓点")
        self.toggle_pointMode_button.clicked.connect(self.toggle_pointMode)
        button_layout.addWidget(self.toggle_pointMode_button)

        self.take_contour_checkbox = QCheckBox("考虑轮廓", self)
        self.take_contour_checkbox.setChecked(True)  # 默认选中

        self.output_message_label = QLabel("")
        button_layout.addWidget(self.output_message_label)

        # 连接复选框的状态变化信号
        self.take_contour_checkbox.stateChanged.connect(self.on_take_contour_checkbox_state_change)
        button_layout.addWidget(self.take_contour_checkbox)
        #self.take_contour_label = QLabel("复选框状态：选中", self)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def open_image(self):
        # 打开文件对话框选择图像
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.png *.bmp *.jpeg)")
        if file_path:
            self.img = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            self.file_path = file_path.split("/")[-1]
            print(self.file_path)
            self.display_image(self.img, "original")

    def update_density(self):
        self.density = self.density_slider.value()
        self.density_label.setText(f"密度{self.density/100:.5f}")

    def update_line_density(self):
        self.line_density = self.line_density_slider.value()
        self.line_density_label.setText(f"轮廓点密度{self.line_density/100:.5f}")

    def update_t1(self):
        self.t1 = self.t1_slider.value()

    def update_t2(self):
        self.t2 = self.t2_slider.value()

    def update_brightness(self):
        self.brightness = self.brightness_slider.value()
        self.brightness_label.setText(f"明暗阈值{self.brightness}")

    def update_zero_depth(self):
        self.zero_depth = self.zerodepth_slider.value()
        self.zerodepth_label.setText(f"零深度：{self.zero_depth}")

    def display_image(self, img, image_type):
        """显示图像：原图、深度图或结果图"""
        if image_type == "original":
            #img_bgr = cv2.cvtColor(img, cv2.COLOR_)
            height, width, channels = img.shape
            bytes_per_line = channels * width
            q_img = QPixmap.fromImage(QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888))
            self.original_image_label.setPixmap(q_img)
            self.original_image_label.setScaledContents(True)  # 允许图片根据标签大小缩放

        elif image_type == "depth":
            depth_map_resized = np.uint8(img / np.max(img) * 255)
            height, width = depth_map_resized.shape
            bytes_per_line = width
            q_img = QPixmap.fromImage(QImage(depth_map_resized.data, width, height, bytes_per_line, QImage.Format_Grayscale8))
            self.depth_image_label.setPixmap(q_img)
            self.depth_image_label.setScaledContents(True)  # 允许图片根据标签大小缩放

        elif image_type == "final":
            #for point in self.points:
            #    cv2.circle(img_bgr, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)
            height, width, channels = img.shape
            bytes_per_line = channels * width
            q_img = QPixmap.fromImage(QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888))
            self.final_image_label.setPixmap(q_img)
            self.final_image_label.setScaledContents(True)  # 允许图片根据标签大小缩放

    def toggle_load_model(self):
        if self.model is None:
            # 加载 Depth-Anything 预训练模型
            retry = 0
            while(retry<6):
                try:
                    self.model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
                    self.load_model_button.setText("深度估算模型已加载")
                    break
                except:
                    print(f"模型加载失败，正在重试({retry}/5)")
                    self.load_model_button.setText(f"模型加载失败，正在重试({retry})")
                finally:
                    retry += 1
        else:
            print("模型已经加载")

    def toggle_pointMode(self):
        if self.pointMode == "contour":
            self.pointMode = "brightness"
            self.toggle_pointMode_button.setText("当前模式：明度阈值")
            self.take_contour_checkbox.setCheckState(2)
        elif self.pointMode == "brightness":
            self.pointMode = "darkness"
            self.toggle_pointMode_button.setText("当前模式：暗度阈值")        
        else:
            self.pointMode = "contour"
            self.toggle_pointMode_button.setText("当前模式：仅轮廓点")

    def on_take_contour_checkbox_state_change(self, state):
        # 当复选框状态变化时，更新标签文本
        self.take_contour = state == 2

    def process_image(self):
        if self.img is None:
            print("请先选择一张图片！")
            return

        if self.model is None:
            print("请先加载模型！")
            return


        self.points = []
        # 获取轮廓/重要部分

        blurred_img = cv2.GaussianBlur(self.img, (5, 5), 0)
        edges = cv2.Canny(blurred_img, self.t1, self.t2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask_important = np.zeros_like(self.img, dtype=np.uint8)
        for contour in contours:
            cv2.drawContours(mask_important, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        mask_important_binary = np.all(mask_important == (255, 255, 255), axis=-1)  # 检查每个像素是否为轮廓内部

        # 提取轮廓点
        #print(self.pointMode == "contour")
        #print(self.take_contour)

        if (self.pointMode == "contour") or self.take_contour:
            print("take contour")
            for contour in contours:
                for i in range(0, len(contour), int(100/self.line_density)):  # 更简洁的方式处理密度
                    self.points.append(contour[i][0])

        line_points_count = len(self.points)

        if self.pointMode != "contour":
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            mask = gray_img > self.brightness if self.pointMode == "brightness" else gray_img < self.brightness

            # 根据明度生成点密度

            height, width = gray_img.shape
            size = width * height

            for y in range(0, height, int(100/self.density)):
                for x in range(0, width, int(100/self.density)):
                    #print(self.density)
                    #if (x * y) % int(self.density) == 0:
                    if mask[y, x]:  # 明度大于阈值的区域
                        important_factor = 1 if mask_important_binary[y,x] else 0.5
                        # 根据明度值调整点的密度
                        brightness = gray_img[y, x] 
                        probability = (brightness - self.brightness) / (255 - self.brightness) if self.pointMode == "brightness" else 1 - brightness / self.brightness
                        if random.random() < probability * important_factor:  # 按概率生成点
                            self.points.append((x, y))
        
        area_points_count = len(self.points) - line_points_count


        # 计算深度图
        input_img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        depth_result = self.model(input_img)["depth"]

        # 将深度图从PIL.Image.Image转换为NumPy数组
        self.depth_map = np.array(depth_result)
        depth_map_displayed = np.array(depth_result)
        
        cv2.drawContours(depth_map_displayed, contours, -1, (0, 0, 255), thickness=2)
        #self.depth_map_resized = depth_result
        # 调整深度图的大小以匹配原始图像
        #self.depth_map_resized = cv2.resize(depth_np, (self.img.shape[1], self.img.shape[0]))
        # 显示原图和深度图
        self.display_image(self.img, "original")
        self.display_image(depth_map_displayed, "depth")

        # 绘制最终图像
        final_img = self.img.copy()

        # 生成 SVG
        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        save_path = f"outputs/{self.file_path}.svg"
        dwg = svgwrite.Drawing(save_path, profile='tiny')
        for point in self.points:
            depth = self.depth_map[point[1], point[0]] - self.zero_depth
            #print(f"Depth: {depth}")
            curvature = depth * self.img.shape[0] / 1000

            offset = (1 + 3 * self.a) * curvature / 4
            x0 = point[0] - curvature
            x1 = point[0] + curvature
            y0 = point[1] - curvature  + offset
            h_x0 = point[0] - curvature * self.a
            h_x1 = point[0] + curvature * self.a
            h_y = point[1] - curvature * self.a  + offset


            #print(f"x0: {x0}, x1: {x1}, y: {y0}, h_x0: {h_x0}, h_x1: {h_x1}, h_y: {h_y}")
            path = dwg.path(d=f"M {x0},{y0} C {h_x0},{h_y} {h_x1},{h_y} {x1},{y0}", stroke="blue", fill="none", stroke_width=1)

            #cv2.putText(final_img, str(depth), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.01, (0,0,0), 1)
            
            num_points = 50  # 曲线上的点的数量
            for i in range(num_points + 1):
                t = i / num_points
                x = (1 - t)**3 * x0 + 3 * (1 - t)**2 * t * h_x0 + 3 * (1 - t) * t**2 * h_x1 + t**3 * x1
                y = (1 - t)**3 * y0 + 3 * (1 - t)**2 * t * h_y + 3 * (1 - t) * t**2 * h_y + t**3 * y0
                cv2.circle(final_img, (int(x), int(y)), 1, (255, 255, 255), -1)
                #print(int(x), int(y))
            cv2.circle(final_img, (int(x0), int(y0)), 2, (0, 255, 0), -1)  # 起点
            cv2.circle(final_img, (int(x1), int(y0)), 2, (0, 0, 255), -1)  # 终点
            
            #cv2.circle(final_img, (int(h_x0), int(h_y)), 1, (255, 0, 255), -1)  # 手柄1
            #cv2.circle(final_img, (int(h_x1), int(h_y)), 1, (255, 255, 0), -1)  # 手柄2

            cv2.circle(final_img, (point[0], point[1]), 3, (255, 0, 0), -1)  # 中心点
            dwg.add(path)
        self.display_image(final_img, "final")
        dwg.save()

        self.output_message_label.setText(f"SVG 文件已生成：{save_path}\n轮廓点量:{line_points_count}|区域点量:{area_points_count}")
        print(f"SVG 文件已生成：{save_path}")
        print(f"轮廓点量:{line_points_count}|区域点量:{area_points_count}")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
