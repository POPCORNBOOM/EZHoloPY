import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import svgwrite

input_file = '33.jpg'

# 第一步 获取轮廓
# 读取动漫图像
img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

# 高斯滤波去噪
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# Canny 边缘检测
#edges = cv2.Canny(blurred_img, 100, 200)

t1 = 100 # 低阈值, 低于这个值的像素点会被抛弃
t2 = 200 # 高阈值, 高于这个值的像素点会被保留
edges = cv2.Canny(blurred_img, t1, t2)

# 第二步 按照特定密度在轮廓上布置点
# 密度
density = 3

# 获取轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 为每个轮廓点布置点
points = []
for contour in contours:
    for i in range(len(contour)):
        if i % density == 0:
            points.append(contour[i][0])



# 第三步 密度填充暗面
import random

# 设置明度阈值
brightness_threshold = 50  # 明度阈值
mask = img < brightness_threshold  # 创建明度掩码

# 根据明度生成点密度

height, width = img.shape

for y in range(height):
    for x in range(width):
        if mask[y, x]:  # 明度大于阈值的区域
            # 根据明度值调整点的密度
            brightness = img[y, x]
            probability = 1 - brightness / 99.0  # 将明度归一化为 [0, 1]
            if random.random() < probability * 0.003:  # 按概率生成点
                points.append((x, y))

# 绘制结果
# 将点转换为 NumPy 数组
points = np.array(points)

plt.imshow(edges, cmap='gray')
plt.scatter(points[:, 0], points[:, 1], s=1, c='r')
plt.title('Points on Contours')
plt.show()


# 第三步 取得点的深度值

# 加载 MiDaS 预训练模型
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
model.to("cpu")
model.eval()

# 图像预处理
img = Image.open(input_file)
input_size = img.size  # 保存原始图像的尺寸
transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor(), transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])])
input_tensor = transform(img).unsqueeze(0)

# 进行深度估计
with torch.no_grad():
    depth_map = model(input_tensor)

# 后处理并恢复深度图的尺寸
depth_map = depth_map.squeeze().cpu().numpy()

# 使用 OpenCV 或者其他方法将深度图恢复为与原图相同的尺寸
depth_map_resized = cv2.resize(depth_map, input_size, interpolation=cv2.INTER_CUBIC)

# 显示深度图
depth_map_resized = np.uint8(depth_map_resized / np.max(depth_map_resized) * 255)  # 归一化并转为图像
depth_map_img = Image.fromarray(depth_map_resized)
depth_map_img.show()


# 第四步 为每个点获取深度值，并以此点为顶点，深度值为弯曲程度绘制双曲线
# 获取每个点的深度值

# 设置 SVG 文件
dwg = svgwrite.Drawing(input_file+".svg", profile='tiny')

depths = []
for point in points:
    depths.append(depth_map_resized[point[1], point[0]])

# 将 img 转换为 NumPy 数组，以便使用 OpenCV 绘制
output_img = np.array(img)  # 转换为 NumPy 数组

# 如果原图是灰度图，转换为 BGR 图像
if len(output_img.shape) == 2:  # 灰度图像（只有一个通道）
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)

# 绘制双曲线并简化成带手柄的三次贝塞尔曲线
for i, point in enumerate(points):
    depth = depths[i]-128

    # 曲率与深度值的关系：离128越远，曲率越小
    curvature = depth * output_img.shape[0] / 1000

    x0=point[0]-curvature
    x1=point[0]+curvature
    y=point[1]-curvature

    h_x0 = point[0] - curvature*0.16
    h_x1 = point[0] + curvature*0.16
    h_y = point[1] - curvature*0.16


    # 在 OpenCV 图像上绘制端点和手柄
    cv2.circle(output_img, (int(x0), int(y)), 2, (0, 255, 0), -1)  # 起点
    cv2.circle(output_img, (int(x1), int(y)), 2, (0, 0, 255), -1)  # 终点
    
    cv2.circle(output_img, (int(h_x0), int(h_y)), 1, (255, 0, 255), -1)  # 手柄1
    cv2.circle(output_img, (int(h_x1), int(h_y)), 1, (255, 255, 0), -1)  # 手柄2

    cv2.circle(output_img, (point[0], point[1]), 3, (255, 0, 0), -1)  # 中心点

    # 绘制手柄的线
    #cv2.line(output_img, (x0, y0), (point[0], point[1]), (255, 255, 0), 1)
    #cv2.line(output_img, (x1, y1), (point[0], point[1]), (255, 255, 0), 1)

    # 在 SVG 中绘制三次贝塞尔曲线
    path = dwg.path(d=f"M {x0},{y} C {h_x0},{h_y} {h_x1},{h_y} {x1},{y}", 
                    stroke="blue", fill="none", stroke_width=1)
    dwg.add(path)


# 显示带有双曲线的结果
plt.imshow(output_img)
plt.title('Hyperbolas on Contours')
plt.show()

# 保存生成的 SVG 文件
dwg.save()

print("SVG 文件已生成：", input_file+".svg")

