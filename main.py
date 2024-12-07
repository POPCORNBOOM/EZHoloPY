import numpy as np
import struct
import math
import matplotlib.pyplot as plt

SAVE_SEPARATE = 0
CANVAS_SIZE=3
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
def merge_contours(layer_contours, tolerance=1e-6):
    merged_contours = []  # 存储合并后的轮廓
    while layer_contours:
        # 从剩余的线段中取出一条作为初始轮廓
        current_contour = list(layer_contours.pop(0))
        
        i = 0
        while i < len(layer_contours):
            segment = layer_contours[i]
            # 检查当前轮廓的末尾与另一线段的起点是否接近
            if point_distance(current_contour[-1], segment[0]) < tolerance:
                current_contour.append(segment[1])  # 加入该线段的终点
                layer_contours.pop(i)  # 移除该线段
                i = 0  # 重置检查
            # 检查当前轮廓的末尾与另一线段的终点是否接近
            elif point_distance(current_contour[-1], segment[1]) < tolerance:
                current_contour.append(segment[0])  # 加入该线段的起点
                layer_contours.pop(i)  # 移除该线段
                i = 0  # 重置检查
            # 检查当前轮廓的开头与另一线段的终点是否接近
            elif point_distance(current_contour[0], segment[1]) < tolerance:
                current_contour.insert(0, segment[0])  # 加入该线段的起点
                layer_contours.pop(i)  # 移除该线段
                i = 0  # 重置检查
            # 检查当前轮廓的开头与另一线段的起点是否接近
            elif point_distance(current_contour[0], segment[0]) < tolerance:
                current_contour.insert(0, segment[1])  # 加入该线段的终点
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
                layer_contours.append(intersections)
        # 合并轮廓线段
        merged_contours = merge_contours(layer_contours)
        contours[layer] = merged_contours
    return contours

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
                    plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], color=color)
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
                    plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], color=color)

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


# 主程序
def main(file_path, num_layers,scratch = 0, R=2.5, stepR=0.2):
    vertices = read_stl(file_path)
    min_z = min([min(v[2][2], v[1][2], v[0][2]) for v in vertices])
    max_z = max([max(v[2][2], v[1][2], v[0][2]) for v in vertices])
    z_step = (max_z - min_z) / num_layers
    layers = slice_stl(vertices, num_layers, min_z, z_step)
    contours = compute_outer_contours(layers, min_z, z_step)
    #print_contours(contours)
    #plot_contours(contours)
    if(SAVE_SEPARATE):  
        plot_contours(contours,scratch,R,stepR)
    else:  
        plot_all_layers_in_one(contours,scratch,R,stepR)

import os
# 示例：使用 'example.stl' 文件，切割成10层
if __name__ == "__main__":
    #main("FINALY.STL", 40)
    args = input("[STL文件名] [切割层数] ([底层半径] [递增半径]) ([重叠:1|0])\n处理.stl>").split(" ")
    if(len(args)==5 or len(args)==3):
        SAVE_SEPARATE = args.pop(-1) == "0"
        print(SAVE_SEPARATE)
    if(len(args)==2):
        file = args[0]
        if(file.find(".")==-1):
            file += ".stl"     
        while not os.path.exists(file):
            file = input("load file:")
            if(file.find(".")==-1):
                file += ".stl"     
        slice = int(args[1])
        main(file, slice)
    elif(len(args)==4):
        file = args[0]
        if(file.find(".")==-1):
            file += ".stl"     
        while not os.path.exists(file):
            file = input("load file:")
            if(file.find(".")==-1):
                file += ".stl"     
        slice = int(args[1])
        scratch = 1
        baseR = float(args[2])
        stepR = float(args[3])
        main(file,slice,scratch,baseR,stepR)

