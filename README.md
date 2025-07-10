# 介绍
### 2025年全国大学生嵌入式芯片与系统设计竞赛应用赛道参赛作品。队伍编号16168
### 本队选择瑞芯微赛道AI人工智能视觉应用方向的赛题。
## 项目文件夹介绍
### Item_camera
🚀基于 Qt5 框架开发的智能垃圾分类系统控制界面，实现从图像采集、模型推理到舵机控制的全流程可视化操作。系统支持双模态 PWM 输出控制，可直接部署于 ELF2 开发板。
### 改进yolov8模型代码
🔍 YOLOv8-SimSPPF-MPDIoU<br>
🚀本项目基于 Ultralytics YOLOv8 架构，提出改进模型 YOLOv8-SimSPPF-MPDIoU，面向智能垃圾分类等高效部署场景，具备更强的特征表达能力与定位精度。<br>
✅ 主要改进<br>
SimSPPF：引入轻量化的 SimSPPF 模块，替代原始 SPPF，降低计算量同时保留多尺度感受野信息。<br>
MPDIoU Loss：采用自定义 MPDIoU（Multi-Point Dynamic IoU）边界框回归损失，增强检测框对复杂目标轮廓的适应性，提升定位精度。<br>
![改进的yolov8架构图](https://github.com/wu20249/garbage-sorting_001/代码/图片/改进的yolov8架构图.png)
