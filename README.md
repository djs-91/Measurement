---

# RealSense 激光点云处理与距离测量系统

本项目使用 Intel RealSense 深度相机（如 L515）进行实时图像与深度流处理，结合绿色激光线识别、3D 点云生成、聚类分析和距离测量，适用于高精度物体识别与工业检测场景。

## ✨ 功能特色

- 实时采集 RGB、深度图像与 IMU 数据（加速度计、陀螺仪）
- 自动检测绿色激光线并生成分割提示点
- 基于 SAM2.1模型进行图像分割
- 点云生成与多种视角渲染
- DBSCAN 聚类分析，智能分离多条激光线
- 精确计算多条激光线之间的距离
- 支持 CloudCompare 和 Open3D 可视化工具
- 多线程与多进程加速图像和点云处理

## 🧱 项目结构

- `Measurement.py`：主程序，集成所有核心功能模块

- `configs/`：Hydra 配置文件目录

- `checkpoints/`：SAM 模型权重文件目录 

  ```
    cd checkpoints && \
    ./download_ckpts.sh && \
    cd ..
  ```

- `point_clouds/`：导出的点云数据保存目录

## 📦 依赖安装

推荐 Python 3.10+，建议使用虚拟环境：

```bash
pip install -r requirements.txt
```

主要依赖：

- pyrealsense2
- numpy, pandas, matplotlib, opencv-python
- torch
- open3d
- cloudComPy
- hydra-core
- scikit-learn
- Pillow, joblib

> ⚠️ 注意：
>
> - `cloudComPy` 无法通过 PyPI 安装，请从 [CloudCompare GitHub](https://github.com/CloudCompare/CloudCompare) 编译。
> - `torch` 应根据 CUDA 版本选择正确的 wheel 包。
> - pyrealsense2 编译说明见后文。

------

## 🧠 深度优化说明

### ✅ 编译 pyrealsense2（建议版本 ≤ 2.54，L515 兼容）

关键 CMake 选项：

```bash
cmake .. \
 -DFORCE_RSUSB_BACKEND=ON \
 -DBUILD_PYTHON_BINDINGS=ON \
 -DPYTHON_EXECUTABLE=$(which python3) \
 -DBUILD_WITH_CUDA=ON \
 -DENABLE_MEMS_MODULES=ON
```

> - 确保安装 `pybind11`
> - 编译失败如出现 libcurl/pybind 问题需手动下载对应源码
> - L515 不支持 2.55+ 版本，请使用 2.54.2 编译

### ✅ OpenCV + CUDA 编译建议（Jetson）

参考构建命令（OpenCV 4.11，Python 3.10，Jetson 平台）：

```bash
cmake \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=/usr/local \
 -DWITH_CUDA=1 \
 -DCUDA_ARCH_BIN=8.7 \
 -DOPENCV_ENABLE_NONFREE=1 \
 -DBUILD_opencv_python3=1 \
 -DPYTHON3_EXECUTABLE=... \
 -DPYTHON3_INCLUDE_DIR=... \
 -DPYTHON3_LIBRARY=... \
 ..
```

构建完成后，复制 `cv2.so` 到你的 Conda 环境 `site-packages/` 中。

验证：

```bash
python -c "import cv2; print(cv2.__version__)"
```

### ✅ Jetson 电流限制调整（防止 Over-current 报错）

Jetson 在高负载下可能触发系统电流限制：

```bash
sudo chmod 777 /sys/.../curr4_crit
echo 5000 > /sys/.../curr4_crit
```

原始电流值为 3240，可改为 5000 提升稳定性。详情参见 Jetson 官方文档。

## 🚀 使用方法

1. 确保 RealSense 相机连接正确（支持 L515）
2. 运行主程序：

```bash
python Measuremen.py
```

3. 使用快捷键进行交互，例如：

- `e` 导出点云为 `.ply` 文件
- `m` 切换“激光区域”与“全场景”点云显示模式
- `p` 开启/关闭点云预览

## 🖼️ 示例效果

- 激光线自动检测与分割
- 点云渲染图（Open3D）
- 距离测量结果与端点可视化

（待更新）

## 📁 导出文件

点云数据将自动导出至 `point_clouds/` 文件夹，格式为 `.ply`，可在 CloudCompare 或 MeshLab 中打开。

## ⚙️ 配置说明

配置使用 Hydra 管理，默认从 `configs/sam2.1_hiera_l.yaml` 加载模型参数，可自定义修改。

## 📌 注意事项

- 推荐在支持 GPU 的环境中运行以提高性能
- 如果通过 SSH 连接运行程序，请确保开启 `X11` 转发
- 不建议在无 GUI 环境中运行

## 📄 License

本项目仅用于科研与教学用途，禁止商业使用。

---

## 🛠️ 开发者说明

项目结构较复杂，以下是一些关键模块说明：

| 模块/类               | 功能简介 |
|----------------------|----------|
| `ClusterTracker`     | 跟踪聚类数量与参数，自动调整 DBSCAN 聚类精度 |
| `IMUChangeDetector`  | 通过 IMU 检测装置是否移动，避免误测量 |
| `create_point_cloud()` | 将 RGB + 深度图像转换为点云数据，支持掩码过滤 |
| `detect_green_laser_for_sam()` | 基于 HSV 自动提取绿色激光线区域，并生成 SAM 提示点 |
| `calculate_laser_lines_distance_exclude_bends()` | 精确计算两条激光线之间的距离，并排除弯曲区域干扰 |

程序中采用多线程 + 多进程加速，确保在高分辨率、复杂点云下仍具备实时性能。

## ✅ TODO 列表

- [ ] 支持保存/加载测量历史记录
- [ ] 增加 Web 可视化界面（如 Flask + Three.js）
- [ ] 使用 ONNX 或 TensorRT 加速 SAM 模型推理
- [ ] 增加更友好的 GUI 操作界面
- [ ] 兼容更多 RealSense 型号（如 D435）

欢迎社区贡献代码！

## 🙏 致谢

- [Intel RealSense SDK](https://www.intelrealsense.com/sdk-2/)
- [Open3D](http://www.open3d.org/)
- [Segment Anything (SAM2)](https://github.com/facebookresearch/segment-anything)
- [CloudCompare](https://www.cloudcompare.org/)
- Jetson 社区文档
- 所有开源工具的开发者们 ❤️

---


