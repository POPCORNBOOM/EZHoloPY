import numpy as np
import struct
import math
import matplotlib.pyplot as plt

SAVE_SEPARATE = 0
CANVAS_SIZE=3
DENSTIY=3
# 读取二进制STL文件中的三角面片数据
def read_stl(file_path):
    vertices = []
    with open(file_path, 'rb') as file:  # 以二进制模式打开文件
        file.read(80)  # 跳过文件头（80字节）
        num_triangles = struct.unpack('I', file.read(4))[0]  # 读取三角形数目
        for _ in range(num_triangles):
            # 读取每个三角形的法向量（3个浮点数）和3个顶点（每个顶点3个浮点数）
            normal = struct.unpack('3f', file.read(12))
            v1 = struct.unpack('3f', file.read(12))
            v2 = struct.unpack('3f', file.read(12))
            v3 = struct.unpack('3f', file.read(12))
            vertices.append((np.array(v1), np.array(v2), np.array(v3)))
            file.read(2)  # 跳过每个三角形的附加字节
    return vertices

# 根据给定的切割层数，获取每一层的Z轴范围并筛选三角形
def slice_stl(vertices, num_layers, min_z, z_step):
    layers = {i: [] for i in range(num_layers)}
    for v1, v2, v3 in vertices:
        z_vals = [v1[2], v2[2], v3[2]]
        min_z_tri = min(z_vals)
        max_z_tri = max(z_vals)
        for layer in range(num_layers):
            layer_z = min_z + layer * z_step
            if min_z_tri <= layer_z < max_z_tri:
                layers[layer].append((v1, v2, v3))
    return layers

# 计算两点之间的欧几里得距离
def point_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# 合并首尾相接的线段，避免重复点
def merge_contours(layer_contours, tolerance=1e-3):
    merged_contours = []  # 存储合并后的轮廓
    while layer_contours:
        # 从剩余的线段中取出一条作为初始轮廓
        current_contour = list(layer_contours.pop(0))
        
        i = 0
        while i < len(layer_contours):
            contour2 = layer_contours[i]
            # 检查当前轮廓的末尾与另一线段的起点是否接近
            if point_distance(current_contour[-1], contour2[0]) < tolerance:
                # 将整个 contour2 加到 current_contour 的末尾，忽略 contour2 的第一个点，因为已经是接点
                current_contour.extend(contour2[1:])
                layer_contours.pop(i)  # 移除该线段
                i = 0  # 重置检查
            # 检查当前轮廓的末尾与另一线段的终点是否接近
            elif point_distance(current_contour[-1], contour2[-1]) < tolerance:
                # 将整个 contour2 加到 current_contour 的末尾，倒序添加（从倒数第二个点开始）
                current_contour.extend(contour2[-2::-1])
                layer_contours.pop(i)  # 移除该线段
                i = 0  # 重置检查
            # 检查当前轮廓的开头与另一线段的终点是否接近
            elif point_distance(current_contour[0], contour2[-1]) < tolerance:
                # 将 contour2 的所有点（顺序）添加到 current_contour 的开头
                current_contour.insert(0, contour2[0])
                current_contour[1:0] = contour2[1:]  # 插入剩余的部分
                layer_contours.pop(i)  # 移除该线段
                i = 0  # 重置检查
            # 检查当前轮廓的开头与另一线段的起点是否接近
            elif point_distance(current_contour[0], contour2[0]) < tolerance:
                # 将 contour2 的所有点（倒序）添加到 current_contour 的开头
                current_contour.insert(0, contour2[-1])
                current_contour[1:0] = contour2[-2::-1]  # 插入倒序的部分
                layer_contours.pop(i)  # 移除该线段
                i = 0  # 重置检查
            else:
                i += 1  # 检查下一个线段

        # 如果轮廓的首尾点接近，可以认为它是闭合的，去掉重复点
        if point_distance(current_contour[0], current_contour[-1]) < tolerance:
            current_contour.pop()  # 去掉最后一个重复的点

        merged_contours.append(current_contour)  # 当前轮廓已整理完毕，存储
    
    return merged_contours
# 计算每一层的外轮廓（每一层的交线）
def compute_outer_contours(layers, min_z, z_step):
    contours = {}
    for layer, triangles in layers.items():
        layer_z = min_z + 1e-3 + layer * z_step
        layer_contours = []
        for v1, v2, v3 in triangles:
            edges = [(v1, v2), (v2, v3), (v3, v1)]
            intersections = []
            for edge in edges:
                p1, p2 = edge
                if (min(p1[2], p2[2]) <= layer_z <= max(p1[2], p2[2])):
                    t = (layer_z - p1[2]) / (p2[2] - p1[2])
                    intersection = (p1[0] + t * (p2[0] - p1[0]), 
                                    p1[1] + t * (p2[1] - p1[1]))
                    intersections.append(intersection)
            if len(intersections) == 2:
                #layer_contours.append(intersections)
                it = insert_points_on_segments(intersections, DENSTIY)
                layer_contours.append(it)
        
        # 合并轮廓线段
        merged_contours = merge_contours(layer_contours)
        contours[layer] = merged_contours
    return contours

# 在线段中按照指定的密度插入点
def insert_points_on_segments(segment, density):
    """
    在给定的线段上按指定的密度插入点
    :param segment: 线段，两端点 (p1, p2)
    :param density: 点的密度，表示每单位长度插入的点数
    :return: 返回插入点后的点列表
    """
    p1, p2 = segment
    distance = point_distance(p1, p2)
    
    # 计算线段上应该插入的点数
    num_points = int(distance * density)
    
    # 如果线段长度较小，插入一个点（端点）
    if num_points == 0:
        return [p1, p2]
    
    # 计算每个点的插值
    points = [p1]  # 包括起始点
    for i in range(1, num_points):
        t = i / num_points  # 计算比例
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        points.append((x, y))  # 插入中间点
    
    points.append(p2)  # 包括终点
    return points

# 打印每一层的轮廓
def print_contours(contours):
    for layer, layer_contours in contours.items():
        print(f"Layer {layer} contours:")
        for contour in layer_contours:
            print(f"Contour: {contour}")
        print()

def plot_contours(contours, scatch, baseR=10, stepR=1):
    for layer, layer_contours in contours.items():
        temp_contour = []  # 用于存储只有一个点的轮廓
        color = (random.random(), random.random(), random.random())
        plt.close()  # 关闭之前的图形
        plt.figure()  # 创建一个新的图形
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for contour in layer_contours:
            if len(contour) == 1:
                # 如果轮廓只有一个点，添加到临时变量
                temp_contour.append(contour[0])
            else:
                if(scatch==0 or scatch==1):
                    # 如果轮廓有多个点，正常绘制
                    x_coords = [point[0] for point in contour]
                    y_coords = [point[1] for point in contour]
                    plt.plot(x_coords, y_coords, label=f"Contour {layer}",color=color)
                    plt.plot(x_coords, y_coords,marker='.', label=f"Contour {layer}",color=color)
                    # 绘制首尾相接的线段
                    #plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], color=color)
                if(scatch==1 or scatch==2):
                    #遍历contour中的每个点
                    for point in contour:
                        #绘制以point为中心baseR+stepR*layer为半径的圆
                        circle = plt.Circle(point, baseR + stepR * layer, color=color, fill=False, lw=0.5)
                        plt.gca().add_artist(circle)
                        # 更新坐标轴范围
                        min_x = min(min_x, point[0] - baseR - stepR * layer)
                        max_x = max(max_x, point[0] + baseR + stepR * layer)
                        min_y = min(min_y, point[1] - baseR - stepR * layer)
                        max_y = max(max_y, point[1] + baseR + stepR * layer)

        # 绘制临时存储的只有一个点的轮廓
        if temp_contour:
            if(scatch==0 or scatch==1):
                x_coords = [point[0] for point in temp_contour]
                y_coords = [point[1] for point in temp_contour]
                plt.scatter(x_coords, y_coords, color=color, label="Single-point Contour")
                plt.scatter(x_coords, y_coords, color=color,marker='.', label="Single-point Contour")
            if(scatch==1 or scatch==2):
                for point in temp_contour:
                    #绘制以point为中心baseR+stepR*layer为半径的圆
                    circle = plt.Circle(point, baseR + stepR * layer, color=color, fill=False, lw=0.5)
                    plt.gca().add_artist(circle)
                    # 更新坐标轴范围
                    min_x = min(min_x, point[0] - baseR - stepR * layer)
                    max_x = max(max_x, point[0] + baseR + stepR * layer)
                    min_y = min(min_y, point[1] - baseR - stepR * layer)
                    max_y = max(max_y, point[1] + baseR + stepR * layer)
        if(scatch==1 or scatch==2):
            # 更新坐标轴范围
            plt.xlim(-CANVAS_SIZE, CANVAS_SIZE)
            plt.ylim(-CANVAS_SIZE, CANVAS_SIZE)
        plt.title(f"Layer {layer}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().set_aspect('equal', adjustable='box')  # 保持纵横比
        #plt.show()
        plt.savefig(f"Layer {layer}.png",dpi=300)
import random


'''
scatch:是否绘制scatch图 0:仅原图 1:原图+scatch图 2:仅scatch图
'''
def plot_all_layers_in_one(contours, scatch, baseR=10, stepR=1):
    plt.close()  # 关闭之前的图形
    plt.figure()  # 创建一个新的图形
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for layer, layer_contours in contours.items():
        # 随机生成颜色
        color = (random.random(), random.random(), random.random())
        temp_contour = []  # 用于存储只有一个点的轮廓
        for contour in layer_contours:
            if len(contour) == 1:
                # 如果轮廓只有一个点，加入到临时变量
                temp_contour.append(contour[0])
            else:
                if(scatch==0 or scatch==1):
                    # 如果轮廓有多个点，正常绘制
                    x_coords = [point[0] for point in contour]
                    y_coords = [point[1] for point in contour]
                    plt.plot(x_coords, y_coords, color=color)
                    plt.plot(x_coords, y_coords,marker='.', color=color)
                    # 绘制首尾相接的线段
                    #plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], color=color)

                if(scatch==1 or scatch==2):
                    #遍历contour中的每个点
                    for point in contour:
                        #绘制以point为中心baseR+stepR*layer为半径的圆
                        circle = plt.Circle(point, baseR + stepR * layer, color=color, fill=False, lw=0.5)
                        plt.gca().add_artist(circle)
                        # 更新坐标轴范围
                        min_x = min(min_x, point[0] - baseR - stepR * layer)
                        max_x = max(max_x, point[0] + baseR + stepR * layer)
                        min_y = min(min_y, point[1] - baseR - stepR * layer)
                        max_y = max(max_y, point[1] + baseR + stepR * layer)

        if temp_contour:
            if(scatch==0 or scatch==1):
                # 如果轮廓有多个点，正常绘制
                x_coords = [point[0] for point in temp_contour]
                y_coords = [point[1] for point in temp_contour]
                plt.plot(x_coords, y_coords, color=color)
                plt.plot(x_coords, y_coords, marker='.',color=color)
                # 绘制首尾相接的线段
                plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], color=color)

            if(scatch==1 or scatch==2):
                # 遍历contour中的每个点
                for point in temp_contour:
                    # 绘制以point为中心baseR+stepR*layer为半径的圆
                    circle = plt.Circle(point, baseR + stepR * layer, color=color, fill=False, lw=0.5)
                    plt.gca().add_artist(circle)
                    # 更新坐标轴范围
                    min_x = min(min_x, point[0] - baseR - stepR * layer)
                    max_x = max(max_x, point[0] + baseR + stepR * layer)
                    min_y = min(min_y, point[1] - baseR - stepR * layer)
                    max_y = max(max_y, point[1] + baseR + stepR * layer)
    if(scatch==1 or scatch==2):
        # 更新坐标轴范围
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
    plt.title('All Layers Contours')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')  # 保持纵横比
    #plt.show()       
    plt.savefig(f"all_layers.png",dpi=400)

# 导出轮廓到DXF文件
def export_contours_to_dxf(contours, file_path):
    with open(file_path, 'w') as file:
        file.write("0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nSECTION\n2\nTABLES\n0\nENDSEC\n0\nSECTION\n2\nBLOCKS\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n")
        for layer, layer_contours in contours.items():
            for contour in layer_contours:
                file.write("0\nPOLYLINE\n8\nLayer\n")
                for point in contour:
                    file.write(f"0\nVERTEX\n8\nLayer\n10\n{point[0]}\n20\n{point[1]}\n")
                file.write("0\nSEQEND\n")
        file.write("0\nENDSEC\n0\nEOF\n")


import argparse
import sys
import os

def load_file(file_path):
    """加载文件并检查其存在性"""
    while not os.path.exists(file_path):
        file_path = input(f"文件 '{file_path}' 不存在，请重新输入路径 (输入 'exit' 退出): ")
        if file_path.lower() == 'exit':
            print("程序已退出。")
            sys.exit()  # 退出程序
        if file_path.find(".") == -1:
            file_path += ".stl"
    return file_path

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="STL 文件切割程序")
    parser.add_argument("file", type=str, help="STL 文件路径")
    parser.add_argument("num_layers", type=int, help="切割的层数")
    parser.add_argument("-r", "--radius", type=float, default=2.5, help="底层半径")
    parser.add_argument("-s", "--step-radius", type=float, default=0.2, help="递增半径")
    parser.add_argument("-d", "--density", type=int, default=3, help="插入点的密度")
    parser.add_argument("-cs", "--canvas-size", type=int, default=3, help="画布大小")
    parser.add_argument("-m", "--mode", type=int, choices=[0, 1, 2], default=0, help="绘图模式 (0: 仅原图, 1: 原图+scatch图, 2: 仅scatch图)")
    parser.add_argument("-o", "--overlap", type=int, choices=[0, 1], default=0, help="是否保存分层图 (0: 保存, 1: 不保存)")
    parser.add_argument("-e", "--export", type=int, choices=[0, 1], default=0, help="是否导出DXF文件 (0: 不导出, 1: 导出)")
    return parser.parse_args()

def main(file_path, num_layers, save_separate=False, base_radius=2.5, step_radius=0.2, scatch=0, export_dxf=False):
    """主程序逻辑"""
    vertices = read_stl(file_path)
    min_z = min([min(v[2][2], v[1][2], v[0][2]) for v in vertices])
    max_z = max([max(v[2][2], v[1][2], v[0][2]) for v in vertices])
    z_step = (max_z - min_z) / num_layers
    layers = slice_stl(vertices, num_layers, min_z, z_step)
    contours = compute_outer_contours(layers, min_z, z_step)
    
    if save_separate:
        plot_contours(contours, scatch, base_radius, step_radius)
    else:
        plot_all_layers_in_one(contours, scatch, base_radius, step_radius)

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    DENSTIY = args.density
    CANVAS_SIZE = args.canvas_size
    # 加载文件并确保文件路径有效
    file_path = load_file(args.file)
    #file_path = load_file('cube.stl')

    # 调用主程序
    main(file_path, args.num_layers, save_separate=(args.overlap == 0), base_radius=args.radius, step_radius=args.step_radius, scatch=args.mode, export_dxf=(args.export == 1))
    #main(file_path, 10, save_separate=(1), base_radius=10, step_radius=-0.4)

