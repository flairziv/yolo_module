# YOLOv5 模块工具

本项目是一个基于 YOLOv5 的模块工具，专为目标检测任务设计，使用 C++ 语言实现。该工具支持模型加载、图片推理、结果输出等功能，便于集成到各类视觉应用中。

## 项目简介

YOLOv5（You Only Look Once v5）是一种高效的目标检测算法，能够在保证准确率的同时实现实时检测。本仓库实现了 YOLOv5 的推理模块，适用于 C++ 项目。

## 主要功能

- 加载 YOLOv5 预训练模型
- 支持图片和视频的目标检测
- 输出检测结果（类别、置信度、位置等）
- 易于集成至其他 C++ 项目

## 环境依赖

- C++17 及以上
- OpenCV >= 4.0
- CUDA（可选，用于 GPU 加速）
- CMake >= 3.10

## 快速开始

1. **克隆项目**

   ```bash
   git clone https://github.com/flairziv/yolo_module.git
   cd yolo_module
   ```

2. **编译项目**

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. **运行示例**

   ```bash
   ./yolov5_demo path/to/image.jpg
   ```

   - `yolov5_demo` 为演示程序，具体名称以实际源码为准
   - `path/to/image.jpg` 需要替换为你自己的图片路径

## 模型准备

请从 [YOLOv5 官方仓库](https://github.com/ultralytics/yolov5) 下载预训练模型（如 `.pt` 文件），并按需转换为 ONNX 或其他 C++ 支持的格式。

## 使用方法

以代码片段为例：

```cpp
#include "yolov5_module.h"

YOLOv5Detector detector("yolov5.onnx");
auto results = detector.detect("test.jpg");

for (const auto& obj : results) {
    std::cout << "类别: " << obj.class_name
              << " 置信度: " << obj.confidence
              << " 位置: " << obj.bbox << std::endl;
}
```

## 项目结构

```
yolo_module/
├── include/         # 头文件
├── src/             # 源码
├── models/          # 存放模型文件
├── demo/            # 示例代码
└── README.md
```

## 参考资料

- [YOLOv5 官方仓库](https://github.com/ultralytics/yolov5)
- [OpenCV 官方文档](https://opencv.org/)
- [CMake 官方文档](https://cmake.org/)

## 许可证

本项目基于 MIT 许可证开源，欢迎 Fork 和贡献代码！

---

如有问题或建议，请提交 Issue 或 PR。
