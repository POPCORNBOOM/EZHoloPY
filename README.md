# EZHoloPY | 简单刮擦

## Setup
1. 安装依赖
dependencies:
  - python=3.9
  - pip
  - pyqt=5
  - opencv
  - matplotlib
  - numpy
  - pillow
  - torchvision
  - transformers
  - svgwrite
  - pip:
      - torch

2. 运行EZHoloWaifu.py

## 食用方法
1. 加载深度估算模型
2. 导入图片
3. 点击开始处理
4. 享用你的刮擦全息.svg
5. 效果不满意？你可以试着调整轮廓获取阈值t1、t2（↓更多细节，↑更少细节），轮廓点密度（↑将有更多刮擦痕迹用于描述轮廓）
6. 进阶：将模式调整值明度阈值/暗度阈值，EZHoloPY将在亮度大于`亮度阈值`的区域按照亮度分布点（这将产生更多关键点/更多刮擦痕迹）
