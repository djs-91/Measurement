import atexit
import pyrealsense2 as rs
import numpy as np
import pandas as pd
import cv2
import torch
import os
import tempfile
import time
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2_camera_predictor
import gc
import open3d as o3d
import threading
import subprocess

import sys
import os.path

# 添加CPU多核处理支持
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from threading import Lock, RLock
from collections import deque
import warnings
from matplotlib.colors import ListedColormap

# 抑制各种警告
warnings.filterwarnings(
    "ignore", message="The value of the smallest subnormal")  # NumPy警告
warnings.filterwarnings(
    "ignore", message=".*cannot import name '_C' from 'sam2'.*")  # SAM2警告
# 添加Qt和系统级警告过滤
warnings.filterwarnings("ignore", message="QSocketNotifier*")  # Qt警告
warnings.filterwarnings("ignore", message=".*xkbcommon*")  # xkbcommon警告
warnings.filterwarnings("ignore", message="Language not found*")  # CloudComPy警告
# warnings.filterwarnings("ignore", message=".*failed to add default include path.*")  # 路径错误警告

# 设置CPU优化参数
NUM_CORES = multiprocessing.cpu_count()
print(f"检测到{NUM_CORES}个CPU核心")

# 设置环境变量以优化并行处理
os.environ["OMP_NUM_THREADS"] = str(max(1, NUM_CORES // 2))  # OpenMP线程
os.environ["MKL_NUM_THREADS"] = str(max(1, NUM_CORES // 2))  # Intel MKL线程
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_CORES)  # Numexpr线程
os.environ["OPENBLAS_NUM_THREADS"] = str(max(1, NUM_CORES // 2))  # OpenBLAS线程

# 创建全局线程池和进程池
process_pool = None
process_pool_initialized = False
thread_pool = ThreadPoolExecutor(max_workers=NUM_CORES)

# 创建一个进程锁，用于同步对共享资源的访问
process_lock = RLock()

# 跟踪聚类历史的类


class ClusterTracker:
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.cluster_counts = deque(maxlen=max_history)
        self.cluster_sizes = deque(maxlen=max_history)
        self.dbscan_eps_history = deque(maxlen=max_history)
        self.min_samples_history = deque(maxlen=max_history)
        self.lock = RLock()

    def add_clustering_result(self, num_clusters, cluster_sizes, eps, min_samples):
        with self.lock:
            self.cluster_counts.append(num_clusters)
            self.cluster_sizes.append(cluster_sizes)
            self.dbscan_eps_history.append(eps)
            self.min_samples_history.append(min_samples)

    def get_stable_eps_min_samples(self, default_eps=10.0, default_min_samples=5):
        with self.lock:
            if len(self.dbscan_eps_history) < 3:
                return default_eps, default_min_samples

            # 使用最近3次聚类的平均值
            avg_eps = sum(list(self.dbscan_eps_history)[-3:]) / 3
            avg_min_samples = sum(list(self.min_samples_history)[-3:]) / 3

            # 四舍五入min_samples到整数
            avg_min_samples = max(3, round(avg_min_samples))

            return avg_eps, avg_min_samples

    def get_average_cluster_count(self):
        with self.lock:
            if not self.cluster_counts:
                return 0
            return sum(self.cluster_counts) / len(self.cluster_counts)

    def get_median_cluster_sizes(self):
        with self.lock:
            if not self.cluster_sizes:
                return []

            # 展平所有历史中的簇大小
            all_sizes = []
            for sizes in self.cluster_sizes:
                all_sizes.extend(sizes)

            # 如果没有簇，返回空列表
            if not all_sizes:
                return []

            # 计算中位数
            all_sizes.sort()
            mid = len(all_sizes) // 2

            if len(all_sizes) % 2 == 0:
                return (all_sizes[mid - 1] + all_sizes[mid]) / 2
            else:
                return all_sizes[mid]


# 全局聚类跟踪器
cluster_tracker = ClusterTracker(max_history=10)

# 进程池初始化函数


def initialize_process_pool_if_needed():
    """
    按需初始化进程池，避免不必要的资源占用

    返回:
        bool: 是否成功初始化
    """
    global process_pool, process_pool_initialized

    if process_pool_initialized:
        return True

    try:
        # 留一个核心给主进程和UI
        process_pool = multiprocessing.Pool(processes=max(1, NUM_CORES - 1))
        process_pool_initialized = True
        print(f"成功初始化多进程池，使用 {max(1, NUM_CORES-1)} 个进程")
        return True
    except Exception as e:
        print(f"初始化进程池失败: {e}")
        process_pool_initialized = False
        process_pool = None
        return False

# 用于处理DBSCAN分块数据的全局函数


def dbscan_process_chunk(chunk_data):
    """
    处理DBSCAN聚类的数据块

    参数:
        chunk_data: 包含(点数据, min_samples, eps)的元组

    返回:
        labels: 聚类标签
    """
    from sklearn.cluster import DBSCAN

    points, min_samples, eps = chunk_data

    if len(points) < min_samples:
        # 点数不足，全部标记为噪声
        return np.full(len(points), -1)

    # 对当前块执行DBSCAN
    return DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_


# 导入依赖库
import cloudComPy as cc
from sklearn.cluster import DBSCAN

# 设置库可用标志
HAS_CLOUDCOMPY = True
HAS_SKLEARN = True

# 设置环境变量以支持SSH X11转发中的Open3D
if "SSH_CONNECTION" in os.environ:
    # SSH远程连接时使用X11转发地址
    os.environ["DISPLAY"] = "192.168.137.1:0.0"

# 调用设置函数
os.environ["OPEN3D_CPU_RENDERING"] = "true"  # 强制使用CPU渲染，避免GPU依赖
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"  # 设置允许间接渲染
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"  # 设置OpenGL版本

# 检查X11转发是否可用


def check_x11_forwarding():
    try:
        # 检查DISPLAY环境变量是否设置
        if "DISPLAY" not in os.environ:
            print("警告: DISPLAY环境变量未设置，X11转发可能不可用")
            print("建议: 使用 'ssh -X' 或 'ssh -Y' 连接服务器")
            return False

        # 尝试使用xdpyinfo检查X11连接
        result = subprocess.run(
            ["xdpyinfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            print("警告: xdpyinfo 命令失败，X11转发可能不可用")
            print("错误信息:", result.stderr.decode())
            return False

        print("X11转发正常工作")
        return True
    except Exception as e:
        print(f"检查X11转发时出错: {e}")
        return False

# 在ClusterTracker类定义之后，其他函数之前
# IMU变化检测类


class IMUChangeDetector:
    def __init__(self, threshold_roll=2.0, threshold_pitch=2.0, threshold_accel=0.5, history_size=5):
        """
        初始化IMU变化检测器

        参数:
            threshold_roll: Roll角度变化阈值(度)
            threshold_pitch: Pitch角度变化阈值(度)
            threshold_accel: 加速度变化阈值(m/s²)
            history_size: 历史数据保存数量
        """
        self.threshold_roll_rad = np.radians(threshold_roll)  # 转换为弧度
        self.threshold_pitch_rad = np.radians(threshold_pitch)  # 转换为弧度
        self.threshold_accel = threshold_accel
        self.history_size = history_size

        # 历史数据
        self.roll_history = deque(maxlen=history_size)
        self.pitch_history = deque(maxlen=history_size)
        self.accel_history = deque(maxlen=history_size)

        # 上次检测到变化的时间
        self.last_change_time = 0
        self.last_print_time = 0

        # 变化标志
        self.has_significant_change = False

        # 锁，用于线程安全
        self.lock = RLock()

    def update(self, roll, pitch, accel_data):
        """
        更新IMU数据并检测变化

        参数:
            roll: 当前Roll角度(弧度)
            pitch: 当前Pitch角度(弧度)
            accel_data: 当前加速度数据[x,y,z]

        返回:
            是否检测到明显变化
        """
        with self.lock:
            current_time = time.time()

            # 添加新数据
            self.roll_history.append(roll)
            self.pitch_history.append(pitch)
            self.accel_history.append(accel_data)

            # 如果历史数据不足，不进行检测
            if len(self.roll_history) < 2:
                return False

            # 计算角度变化
            roll_change = abs(self.roll_history[-1] - self.roll_history[-2])
            pitch_change = abs(self.pitch_history[-1] - self.pitch_history[-2])

            # 计算加速度变化
            if len(self.accel_history) >= 2:
                accel_change = np.linalg.norm(
                    np.array(self.accel_history[-1]) -
                    np.array(self.accel_history[-2])
                )
            else:
                accel_change = 0

            # 判断是否有显著变化
            significant_change = (
                roll_change > self.threshold_roll_rad
                or pitch_change > self.threshold_pitch_rad
                or accel_change > self.threshold_accel
            )

            # 如果有显著变化或距离上次打印时间超过10秒，允许打印
            if significant_change:
                self.last_change_time = current_time
                self.has_significant_change = True
            elif current_time - self.last_change_time > 10.0:  # 如果10秒没有变化，重置状态
                self.has_significant_change = False

            # 控制打印频率
            allow_print = (
                significant_change or  # 有明显变化时允许打印
                (current_time - self.last_print_time > 10.0)  # 或距离上次打印超过10秒
            )

            if allow_print:
                self.last_print_time = current_time

            return allow_print


# 在全局空间创建单一检测器实例
imu_detector = IMUChangeDetector(
    threshold_roll=1.0, threshold_pitch=1.0, threshold_accel=0.3)

WINDOW_RGB = "RGB Image"
WINDOW_DEPTH = "Depth Image"
WINDOW_POINTCLOUD = "Point Cloud View"

# ====================
# 初始化模块
# ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(script_dir, "configs")
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
initialize_config_dir(config_dir=os.path.abspath(
    config_dir), version_base="1.3")

# 测量结果显示变量
last_measurement_result = None  # 存储最后一次测量结果
last_measurement_std = None  # 存储最后一次测量标准差
last_measurement_count = None  # 存储最后一次有效测量次数
last_measurement_time = 0  # 存储最后一次测量时间
MEASUREMENT_DISPLAY_DURATION = 30  # 测量结果显示持续时间(秒)

# 创建基于 RAM 的临时目录
temp_frame_dir = Path("/dev/shm/temp_frames")
try:
    temp_frame_dir.mkdir(mode=0o777, exist_ok=True)
    for f in temp_frame_dir.glob("*.jpg"):
        f.unlink()
except Exception as e:
    print(f"创建目录失败: {e}")

# 张量计算时的精度设置
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 加载SAM2相机预测器
model_cfg = "sam2.1_hiera_l.yaml"
sam2_checkpoint = os.path.join(script_dir, "checkpoints/sam2.1_hiera_large.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 点云显示线程停止标志
pointcloud_thread_running = False
# 点云数据共享变量
pointcloud_data = {
    "points": None,
    "colors": None,
    "mask": None,
    "updated": False,
    "render_image": None,  # 存储渲染的点云图像
    "full_points": None,  # 存储全场景点云坐标
    "full_colors": None,  # 存储全场景点云颜色
}

# 点云渲染选项
render_options = {
    "background_color": [0.05, 0.05, 0.05],  # 深灰色背景
    "point_size": 6.0,  # 增大点大小以便更好地观察
    "show_coordinate_frame": True,  # 显示坐标轴
    "zoom": 0.30,  # zoom值控制点云大小
    "front": [0, 0, -1],  # 默认前视图
    "lookat": [0, 0, 0],  # 视角中心
    "up": [0, -1, 0],  # 上方向
}

# 添加导出模式变量
export_laser_only = True  # 默认只导出激光区域

# 从深度图和彩色图生成点云


def create_point_cloud(
    depth_image, color_image, depth_scale, depth_intrinsics, mask=None, laser_only=True
):
    """
    从深度图和彩色图生成点云

    参数:
        depth_image: 深度图像数据
        color_image: 彩色图像数据
        depth_scale: 深度尺度因子
        depth_intrinsics: 深度相机内参
        mask: 可选的掩码，标记感兴趣区域
        laser_only: 是否只生成激光区域的点云

    返回:
        points: 点云坐标数组
        colors: 点云颜色数组
    """
    # u7f13u5b58u70b9u4e91u751fu6210u7ed3u679cuff0cu907fu514du91cdu590du8ba1u7b97
    if not hasattr(create_point_cloud, "last_params"):
        create_point_cloud.last_params = None
        create_point_cloud.last_result = None

    # u8ba1u7b97u5f53u524du53c2u6570u7684u54c8u5e0cu503cu4f5cu4e3au552fu4e00u6807u8bc6
    mask_hash = hash(mask.tobytes()) if mask is not None else None
    current_params = {
        "depth_shape": depth_image.shape,
        # u7b80u5355u7684u68c0u6d4bu6df1u5ea6u56feu662fu5426u53d8u5316
        "depth_sum": np.sum(depth_image),
        "mask_hash": mask_hash,
        "laser_only": laser_only
    }

    # u5982u679cu4e0eu4e0au6b21u8c03u7528u53c2u6570u76f8u540cuff0cu76f4u63a5u8fd4u56deu7f13u5b58u7ed3u679c
    if (create_point_cloud.last_params == current_params and
            create_point_cloud.last_result is not None):
        return create_point_cloud.last_result

    # u66f4u65b0u5f53u524du53c2u6570
    create_point_cloud.last_params = current_params

    height, width = depth_image.shape

    # u521du59cbu5316u70b9u4e91u6570u7ec4
    points = []
    colors = []

    # u5982u679cu9700u8981u53eau751fu6210u6fc0u5149u533au57dfu7684u70b9u4e91uff0cu4e14u6ca1u6709u6389u7801uff0cu76f4u63a5u8fd4u56deu7a7au7ed3u679c
    if laser_only and mask is None:
        # u8fd4u56deu7a7au6570u7ec4uff0cu4f46u4f7fu7528shape=(0,3)u786eu4fddu7ef4u5ea6u6b63u786e
        empty_result = np.zeros((0, 3), dtype=np.float32), np.zeros(
            (0, 3), dtype=np.float32), np.array([], dtype=bool)
        create_point_cloud.last_result = empty_result
        return empty_result

    # u521bu5efau6389u7801u7684u526fu672cu4ee5u907fu514du4feeu6539u539fu59cbu6570u636e
    if mask is not None:
        mask_copy = mask.copy().astype(np.uint8)
    else:
        # u5982u679cu4e0du9700u8981u6389u7801uff0cu5219u521bu5efau51681u6389u7801
        mask_copy = np.ones(
            (height, width), dtype=np.uint8) if not laser_only else None

    # u5982u679cu53eau5904u7406u6fc0u5149u533au57dfu4f46u6ca1u6709u6389u7801uff0cu5219u8fd4u56deu7a7au70b9u4e91
    if laser_only and mask_copy is None:
        # u8fd4u56deu7a7au6570u7ec4uff0cu4f46u4f7fu7528shape=(0,3)u786eu4fddu7ef4u5ea6u6b63u786e
        empty_result = np.zeros((0, 3), dtype=np.float32), np.zeros(
            (0, 3), dtype=np.float32), np.array([], dtype=bool)
        create_point_cloud.last_result = empty_result
        return empty_result

    # u5e76u884cu5904u7406u7684u5185u90e8u51fdu6570
    def process_chunk(v_range):
        chunk_points = []
        chunk_colors = []
        chunk_mask = []
        chunk_is_laser = []  # u65b0u589eu6807u8bb0u662fu5426u4e3au6fc0u5149u70b9

        # u964du91c7u6837u7387 - u6bcfu9694u51e0u4e2au50cfu7d20u53d6u4e00u4e2au70b9
        # u6062u590du539fu59cbu5206u8fa8u7387uff0cu7effu8272u6fc0u5149u533au57dfu4f7fu75281
        step = 2 if not laser_only else 1

        for v in v_range[::step]:  # u6bcfu9694stepu884cu91c7u6837u4e00u884c
            for u in range(0, width, step):  # u6bcfu9694stepu5217u91c7u6837u4e00u5217
                # u83b7u53d6u6df1u5ea6u503c
                depth_value = depth_image[v, u]

                # u8df3u8fc7u65e0u6548u6df1u5ea6
                if depth_value == 0 or depth_value > 10000:
                    continue

                # u5e94u7528u6df1u5ea6u5c3au5ea6u56e0u5b50uff0cu8f6cu6362u4e3au7c73
                depth_in_meters = depth_value * depth_scale

                # u68c0u67e5u50cfu7d20u662fu5426u5728u6389u7801u4e2duff0cu5982u679cu53eau5904u7406u6fc0u5149u533au57dfuff0cu5219u8df3u8fc7u6389u7801u5916u7684u50cfu7d20
                in_mask = mask_copy[v,
                                    u] > 0 if mask_copy is not None else True
                if laser_only and not in_mask:
                    continue

                # u5c06u50cfu7d20u5750u6807u548cu6df1u5ea6u503cu8f6cu6362u4e3a3Du70b9u4e91u5750u6807(u7c73)
                point = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics, [u, v], depth_in_meters
                )

                # u8f6cu6362u4e3au6bebu7c73u5355u4f4d
                point_in_mm = np.array(point) * 1000.0
                chunk_points.append(point_in_mm)

                # u83b7u53d6u5bf9u5e94u7684u989cu8272 - u786eu4fddBGRu8f6cRGB
                bgr_color = color_image[v, u].astype(np.float32)
                rgb_color = (
                    np.array([bgr_color[2], bgr_color[1], bgr_color[0]]) / 255.0
                )  # BGRu8f6cRGBu5e76u6807u51c6u5316u52300-1

                # u68c0u6d4bu662fu5426u4e3au7effu8272u6fc0u5149u70b9 - u7b80u5316u5224u65adu6761u4ef6u4ee5u63d0u9ad8u6027u80fd
                is_laser = (
                    # u7effu8272u901au9053u8981u8db3u591fu4eae
                    rgb_color[1] > 0.4
                    # u7effu8272u901au9053u660eu663eu9ad8u4e8eu5176u4ed6u901au9053u603bu548c
                    and rgb_color[1] > (rgb_color[0] + rgb_color[2]) * 0.6
                )

                chunk_colors.append(rgb_color)
                chunk_mask.append(in_mask)
                chunk_is_laser.append(is_laser)

        return chunk_points, chunk_colors, chunk_mask, chunk_is_laser

    # u5212u5206u5782u76f4u65b9u5411u4e3aNUM_CORESu4efduff0cu8fdbu884cu5e76u884cu5904u7406
    v_chunks = np.array_split(range(height), min(NUM_CORES, height))

    # u4f7fu7528u591au7ebfu7a0bu5904u7406u5404u4e2au5757
    results = []
    for chunk in v_chunks:
        results.append(thread_pool.submit(process_chunk, chunk))

    # u6536u96c6u7ed3u679c
    all_points = []
    all_colors = []
    all_mask = []
    all_is_laser = []

    for future in results:
        chunk_result = future.result()
        all_points.extend(chunk_result[0])
        all_colors.extend(chunk_result[1])
        all_mask.extend(chunk_result[2])
        all_is_laser.extend(chunk_result[3])

    # u8f6cu6362u4e3aNumPyu6570u7ec4
    points_np = np.array(all_points)
    colors_np = np.array(all_colors)
    mask_np = np.array(all_mask, dtype=bool)
    laser_np = np.array(all_is_laser, dtype=bool)

    # u7f13u5b58u7ed3u679c
    result = (points_np, colors_np, laser_np)
    create_point_cloud.last_result = result

    # u8fd4u56deu70b9u4e91u6570u636euff0cu9644u52a0u6fc0u5149u70b9u6807u8bb0u4fe1u606f
    return result

# 新函数：自动检测绿色激光并生成提示点


def detect_green_laser_for_sam(image):
    """
    自动检测图像中的绿色激光线，并生成适合SAM模型的提示点

    参数:
        image: 输入的彩色图像

    返回:
        points: SAM提示点坐标列表
        labels: SAM提示点标签列表(1=正点，0=负点)
        mask: 激光区域掩码
        success: 是否成功检测到激光
    """
    # 增强检测精度 - 使用更宽松的HSV范围
    h_min, h_max = 40, 80  # 绿色范围
    s_min, s_max = 100, 255  # 饱和度范围
    v_min, v_max = 130, 255  # 亮度范围

    # 使用线程池处理HSV转换和掩码生成
    def create_hsv_mask():
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 创建HSV掩码
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        return cv2.inRange(hsv, lower_bound, upper_bound)

    # 使用线程池处理形态学操作
    def process_morphology(mask):
        kernel = np.ones((5, 5), np.uint8)  # 更大的核
        mask = cv2.dilate(mask, kernel, iterations=3)  # 增加迭代次数，确保覆盖全部激光
        mask = cv2.erode(mask, kernel, iterations=1)  # 轻微收缩，去除噪点
        return mask

    # 并行执行HSV转换和掩码生成
    mask_future = thread_pool.submit(create_hsv_mask)
    mask = mask_future.result()

    # 并行处理形态学操作
    morphology_future = thread_pool.submit(process_morphology, mask)
    mask = morphology_future.result()

    # 查找轮廓
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有检测到轮廓，返回空结果
    if not contours:
        return [], [], None, False

    # 按面积排序轮廓，取最大的几个
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 限制最多处理的轮廓数量(通常激光线应该是最大的几个轮廓)
    max_contours = min(3, len(contours))
    filtered_contours = []

    # 添加长宽比检查，确保只选择细长形状的轮廓
    min_aspect_ratio = 3.0  # 最小长宽比阈值，可以根据实际情况调整

    # 并行处理轮廓筛选
    def filter_contour(contour):
        # 计算轮廓的最小外接矩形
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]

        # 确保不除以零
        if width == 0 or height == 0:
            return None

        # 计算长宽比，取宽高比中的较大值作为长宽比
        aspect_ratio = max(width / height, height / width)

        # 只选择满足长宽比要求的轮廓（细长形状）
        if aspect_ratio >= min_aspect_ratio:
            return contour
        return None

    # 并行筛选轮廓
    contour_futures = []
    for contour in contours[:max_contours]:
        contour_futures.append(thread_pool.submit(filter_contour, contour))

    # 收集结果
    for future in contour_futures:
        result = future.result()
        if result is not None:
            filtered_contours.append(result)

    # 如果没有满足条件的轮廓，返回空结果
    if not filtered_contours:
        return [], [], None, False

    contours = filtered_contours

    # 为每个轮廓生成提示点
    points = []
    labels = []

    # 创建扩展掩码 - 比原始激光区域更大
    safe_mask = np.zeros_like(mask)
    for contour in contours:
        # 计算轮廓的质心，添加为正点
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        points.append([center_x, center_y])
        labels.append(1)  # 正点

        # 将轮廓区域填充到安全掩码
        cv2.drawContours(safe_mask, [contour], -1, 255, -1)

    # 2. 扩展安全掩码，确保负点远离激光
    safe_mask = cv2.dilate(safe_mask, np.ones(
        (15, 15), np.uint8), iterations=5)

    # 3. 在图像四周固定位置添加负点，避免随机性
    h, w = image.shape[:2]
    # 固定位置的候选点 - 在图像边缘区域
    candidate_points = [
        [50, 50],
        [w - 50, 50],
        [50, h - 50],
        [w - 50, h - 50],  # 四角
        [w // 2, 50],
        [w // 2, h - 50],
        [50, h // 2],
        [w - 50, h // 2],  # 中点
    ]

    # 添加一些固定位置的负点，确保它们在安全掩码外
    for pt in candidate_points:
        x, y = pt
        if 0 <= x < w and 0 <= y < h and safe_mask[y, x] == 0:
            points.append([x, y])
            labels.append(0)  # 负点
            # 一旦找到足够多的负点就停止
            if labels.count(0) >= 4:
                break

    # 如果固定位置没找到足够的负点，则寻找其他区域
    if labels.count(0) < 2:
        # 寻找安全掩码外的大块区域
        inv_mask = cv2.bitwise_not(safe_mask)
        bg_contours, _ = cv2.findContours(
            inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bg_contours = sorted(bg_contours, key=cv2.contourArea, reverse=True)

        for bg_contour in bg_contours[:2]:  # 取最大的两个背景区域
            M = cv2.moments(bg_contour)
            if M["m00"] > 0:
                bg_x = int(M["m10"] / M["m00"])
                bg_y = int(M["m01"] / M["m00"])
                points.append([bg_x, bg_y])
                labels.append(0)  # 负点

    return points, labels, mask, len(points) > 0 and 1 in labels and 0 in labels


# 渲染3D点云为图像，添加参数用于区分显示模式
def render_point_cloud_to_image(pcd, width=640, height=480, title_prefix="Laser"):
    """
    将点云渲染为图像

    参数:
        pcd: Open3D点云对象
        width, height: 输出图像尺寸
        title_prefix: 标题前缀，用于区分模式

    返回:
        渲染的图像数组
    """
    try:
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)

        # 添加点云
        vis.add_geometry(pcd)

        # 配置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.array(render_options["background_color"])
        opt.point_size = render_options["point_size"]
        opt.light_on = True  # 确保光照开启
        opt.show_coordinate_frame = True  # 显示坐标轴

        # 重置视图以自动包含整个点云
        vis.reset_view_point(True)

        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_front(render_options["front"])
        ctr.set_lookat(render_options["lookat"])
        ctr.set_up(render_options["up"])
        ctr.set_zoom(render_options["zoom"])

        # 自动缩放点云使其充满视野
        vis.get_view_control().change_field_of_view(step=-90)

        # 降低渲染质量以提高性能
        opt.point_size = 1.5  # 比默认点尺寸更小
        # 移除不兼容的quality_level设置，此属性在新版Open3D中不存在

        # 更新场景
        vis.poll_events()
        vis.update_renderer()

        # 捕获图像
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # 转换为OpenCV格式
        image_np = np.asarray(image) * 255
        image_np = image_np.astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 添加点云信息
        num_points = len(pcd.points)
        cv2.putText(
            image_np,
            f"{title_prefix} Point Cloud: {num_points} points",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # 添加视图控制提示
        cv2.putText(
            image_np,
            "View: [1]=Front [2]=Side [3]=Top [+/-]=Zoom",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        # 添加导出提示
        cv2.putText(
            image_np,
            "Press 'e' to export as PLY file",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return image_np
    except Exception as e:
        print(f"点云渲染失败: {e}")
        import traceback

        traceback.print_exc()
        return None

# 点云可视化线程函数 - 性能优化版本


def pointcloud_visualization_thread():
    global pointcloud_thread_running, pointcloud_data, export_laser_only

    # 降低渲染频率，减少CPU使用
    update_interval = 0.3  # 从0.1增加到0.3秒
    last_update_time = 0

    # 添加缓存机制，避免不必要的重复渲染
    previous_points_hash = None
    cached_image = None
    points_update_count = 0

    try:
        while pointcloud_thread_running:
            current_time = time.time()

            # 如果有新数据并且达到渲染间隔，更新渲染
            if (
                pointcloud_data["updated"]
                and (current_time - last_update_time) > update_interval
            ):
                # 根据当前模式选择要显示的点云数据
                if export_laser_only:
                    points = pointcloud_data["points"]
                    colors = pointcloud_data["colors"]
                    title_prefix = "Laser"
                else:
                    points = pointcloud_data["full_points"]
                    colors = pointcloud_data["full_colors"]
                    title_prefix = "Full Scene"

                # 检查是否有足够的点云数据
                if points is not None and len(points) > 0:
                    try:
                        # 计算当前点云数据的简单哈希，以检测数据变化
                        # 仅使用前100个点计算哈希，这已足够检测大多数变化
                        points_sample = points[:min(100, len(points))]
                        current_hash = hash(points_sample.tobytes()[:1000])

                        # 只有当点云数据变化，或者每间隔4次更新才进行完整渲染
                        # 这减少了生成大量相同的渲染图像
                        if previous_points_hash != current_hash or points_update_count % 4 == 0:
                            # 创建点云对象
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(points)
                            pcd.colors = o3d.utility.Vector3dVector(colors)

                            # 对点云进行简化的统计离群值滤波 - 减少计算量
                            if len(points) > 2000:  # 只当点过多时去除离群点
                                cl, ind = pcd.remove_statistical_outlier(
                                    nb_neighbors=10,  # 使用更小的领域
                                    std_ratio=2.0
                                )
                                pcd = pcd.select_by_index(ind)

                            # 渲染为图像
                            render_img = render_point_cloud_to_image(
                                pcd, 640, 480, title_prefix
                            )
                            if render_img is not None:
                                cached_image = render_img.copy()  # 缓存渲染结果
                                pointcloud_data["render_image"] = render_img

                                previous_points_hash = current_hash  # 更新哈希值
                            else:
                                # 使用缓存的图像，但更新点计数
                                if cached_image is not None:
                                    num_points = len(points)
                                    img_copy = cached_image.copy()
                                    # 更新点计数文字
                                    # 黑色矩形遮掩原文字
                                    cv2.rectangle(
                                        img_copy, (5, 5), (300, 40), (0, 0, 0), -1)
                                    cv2.putText(
                                        img_copy,
                                        f"{title_prefix} Point Cloud: {num_points} points",
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (255, 255, 255),
                                        2,
                                    )
                                    pointcloud_data["render_image"] = img_copy

                        points_update_count += 1  # 增加更新计数器
                    except Exception as e:
                        print(f"点云渲染错误: {e}")
                else:
                    # 处理点云数据不足的情况，生成提示信息图像
                    info_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    # 设置背景颜色为深灰色
                    info_image[:, :] = [30, 30, 30]  # 深灰色背景

                    # 添加说明信息
                    if export_laser_only:
                        message1 = "无法检测到足够的激光线点云"
                        message2 = "请确保激光线正确照射在目标表面"
                        message3 = "或按'M'键切换到全场景模式"
                    else:
                        message1 = "正在生成全场景点云..."
                        message2 = "如果长时间未显示，请检查深度相机设置"
                        message3 = "或按'P'键关闭点云显示"

                    # 在图像上添加文本
                    cv2.putText(
                        info_image,
                        message1,
                        (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        info_image,
                        message2,
                        (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 200, 200),
                        2,
                    )
                    cv2.putText(
                        info_image,
                        message3,
                        (120, 280),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    # 更新渲染图像
                    pointcloud_data["render_image"] = info_image
                    cached_image = None  # 清除缓存图像

                    pointcloud_data["updated"] = False
                    last_update_time = current_time

                # 添加时间戳用于超时检测
                pointcloud_data["last_update_time"] = current_time  # 用于超时检测

            # 增加休眠时间，减少CPU使用
            time.sleep(0.15)  # 从0.05增加到0.15

    except Exception as e:
        print(f"点云可视化线程异常: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("点云可视化线程已结束")


pipeline = rs.pipeline()
config = rs.config()

try:
    # 添加视频流配置
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)  # 颜色流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度流
    # 添加加速度计和陀螺仪流 - 降低采样率以避免配置错误
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)  # 加速度计
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 100)  # 陀螺仪

    profile = pipeline.start(config)

    # 获取设备并设置 L515 专用参数
    rs_device = profile.get_device()
    depth_sensor = rs_device.first_depth_sensor()

    # 设置预设模式（L515 专用参数）
    preset = rs.l500_visual_preset.short_range
    depth_sensor.set_option(rs.option.visual_preset, int(preset))
    print("成功初始化 L515 配置")

    # 获取深度尺度因子
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度尺度因子: {depth_scale}")

    # 创建并配置深度滤波器链
    print("配置深度滤波器链...")

    # 1. 抽取滤波器 - 降低噪声但保持细节
    decimation = rs.decimation_filter()
    decimation.set_option(
        rs.option.filter_magnitude, 1
    )  # 恢复为1，保持原始分辨率

    # 2. 阈值滤波器 - 去除不可靠的深度值
    threshold_filter = rs.threshold_filter()
    threshold_filter.set_option(rs.option.min_distance, 0.15)  # 最小深度值(米)
    threshold_filter.set_option(rs.option.max_distance, 6.0)  # 最大深度值(米)

    # 3. 视差变换 - 在视差域进行处理以提高精度
    disparity = rs.disparity_transform(True)  # 转换为视差
    disparity_to_depth = rs.disparity_transform(False)  # 视差转回深度

    # 4. 空间滤波器 - 平滑深度图像（调整参数平衡质量和速度）
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)  # 滤波强度 (1-5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)  # 平滑度 (0-1)，增加平滑度
    spatial.set_option(rs.option.filter_smooth_delta, 20)  # 深度差异阈值
    spatial.set_option(rs.option.holes_fill, 1)  # 孔洞填充等级 (0-5)，降低更精确

    # 5. 时间滤波器 - 减少时间维度上的深度噪声
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)  # 时间平滑度 (0-1)
    temporal.set_option(rs.option.filter_smooth_delta, 20)  # 深度波动阈值
    temporal.set_option(rs.option.holes_fill, 3)  # 使用3个历史帧填充孔洞，增强稳定性

    # 6. 孔洞填充滤波器 - 填充缺失的深度值
    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 2)  # 设置填充模式为2（中等填充强度）

    print("深度滤波器链配置完成 - 高质量滤波模式")

except Exception as e:
    print(f"设备初始化失败: {e}")
    exit()

align_to = rs.stream.color
align = rs.align(align_to)

# 检查X11转发
have_x11 = check_x11_forwarding()

# 检查是否在SSH连接中但没有X11转发
if "SSH_CONNECTION" in os.environ and not have_x11:
    print("\n错误: 检测到SSH连接但X11转发不可用")
    print("确保本地X服务器正在运行并正确配置")
    print("程序无法在没有X11转发的SSH会话中运行图形界面，即将退出...\n")
    sys.exit(1)

# 定义相关显示函数


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


# 掩码颜色
mask_colors = [
    np.array([0, 128, 255], dtype=np.uint8),
    np.array([255, 128, 0], dtype=np.uint8),
]


# 修改掩码添加函数，确保正确处理返回值
def add_mask(image, mask, color_id):
    # 确保掩码是二维的
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)  # 移除多余的维度
    elif len(mask.shape) == 1:
        mask = mask.reshape(image.shape[0], image.shape[1])

    # 单通道
    int_mask = mask.astype(np.uint8)

    # 创建掩码图像
    mask_color = mask_colors[color_id % len(mask_colors)]
    mask_img = np.full_like(image, mask_color)

    # 使用掩码矩阵来控制叠加
    mask_img[int_mask == 0] = 0

    # 使用 cv2.addWeighted 叠加原始图像和掩码图像
    res = cv2.addWeighted(image, 0.6, mask_img, 0.4, 0, dtype=cv2.CV_8U)

    # 将mask中为黑色部分保留原图
    res[int_mask == 0] = image[int_mask == 0]
    return res, int_mask  # 同时返回结果图像和掩码

# 在图像上绘制提示点


def draw_points(image, points, labels):
    result = image.copy()
    for i, (point, label) in enumerate(zip(points, labels)):
        x, y = point
        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # 绿色=正点，红色=负点
        cv2.circle(result, (x, y), 5, color, -1)  # 实心圆
        cv2.circle(result, (x, y), 8, color, 2)  # 空心圆边框
    return result

# 检测激光线


def detect_laser_lines(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10
    )

    return lines

# 计算激光线端点


def get_laser_endpoints(lines):
    if lines is None:
        return None

    # 按x坐标排序
    sorted_lines = sorted(lines, key=lambda x: x[0][0])

    # 获取两个激光线的端点
    endpoints = []
    for line in sorted_lines[:2]:  # 只取前两条线
        x1, y1, x2, y2 = line[0]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))

    # 按x坐标排序端点
    endpoints.sort(key=lambda x: x[0])

    return endpoints

# 计算中间两个端点的距离


def calculate_distance(endpoints):
    if len(endpoints) < 4:
        return None

    # 获取中间两个端点
    middle_points = endpoints[1:3]

    # 计算距离
    distance = abs(middle_points[1][0] - middle_points[0][0])

    return distance, middle_points

# 从掩码中提取激光线端点，区分不同板子上的激光线


def get_endpoints_from_mask(mask, orientation="auto"):
    """
    从掩码中提取激光线端点

    参数:
        mask: 分割掩码
        orientation: 方向 - "auto"(自动检测), "horizontal"(水平), "vertical"(垂直)

    返回:
        两个区域的端点: ((point1, point2), (point3, point4))
    """
    # 确保掩码是二维的
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)

    # 将掩码转换为二值图像
    binary_mask = (mask > 0).astype(np.uint8) * 255

    # 找到连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask)

    if num_labels <= 1:  # 如果没有连通区域（除了背景）
        return None

    # 过滤掉太小的区域，只保留大于最小面积的区域
    min_area = 100  # 增大最小面积阈值，过滤掉可能的噪点
    valid_regions = []

    for i in range(1, num_labels):  # 跳过背景(0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            # 存储区域信息：(标签, 质心x, 质心y, 宽度, 高度)
            region_info = (
                i,
                centroids[i][0],
                centroids[i][1],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
            valid_regions.append(region_info)

    if len(valid_regions) < 2:  # 需要至少两个有效区域
        return None

    # 自动检测方向
    if orientation == "auto":
        # 计算所有区域的平均宽高比
        avg_aspect_ratio = sum(
            [w / h if h > 0 else float("inf")
             for _, _, _, w, h in valid_regions]
        ) / len(valid_regions)

        # 计算质心间的x和y方向距离
        if len(valid_regions) >= 2:
            x_distances = []
            y_distances = []

            for i in range(len(valid_regions)):
                for j in range(i + 1, len(valid_regions)):
                    x_dist = abs(valid_regions[i][1] - valid_regions[j][1])
                    y_dist = abs(valid_regions[i][2] - valid_regions[j][2])
                    x_distances.append(x_dist)
                    y_distances.append(y_dist)

            avg_x_dist = sum(x_distances) / \
            len(x_distances) if x_distances else 0
            avg_y_dist = sum(y_distances) / \
            len(y_distances) if y_distances else 0

            # 根据质心距离和平均宽高比决定方向
            if avg_x_dist > avg_y_dist or avg_aspect_ratio > 1.5:
                orientation = "horizontal"  # 水平方向
            else:
                orientation = "vertical"  # 垂直方向
        else:
            # 默认水平
            orientation = "horizontal"

    endpoints_by_region = []

    # 根据方向排序区域
    if orientation == "horizontal":
        # 按质心的x坐标排序，找到左侧和右侧的区域
        valid_regions.sort(key=lambda x: x[1])  # 按x坐标排序
    else:  # vertical
        # 按质心的y坐标排序，找到上方和下方的区域
        valid_regions.sort(key=lambda x: x[2])  # 按y坐标排序

    # 获取前两个区域的端点
    for region_idx, _, _, _, _ in valid_regions[:2]:
        # 创建该连通区域的掩码
        component_mask = (labels == region_idx).astype(np.uint8) * 255

        # 找到轮廓
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if orientation == "horizontal":
                # 找到最左和最右的点
                leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
                rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
                endpoints_by_region.append((leftmost, rightmost))
            else:  # vertical
                # 找到最上和最下的点
                topmost = tuple(contour[contour[:, :, 1].argmin()][0])
                bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
                endpoints_by_region.append((topmost, bottommost))

    # 如果找到的区域少于2个，返回None
    if len(endpoints_by_region) < 2:
        return None

    # 返回两个区域的端点和识别的方向
    return endpoints_by_region[0], endpoints_by_region[1], orientation

# 计算两个板子之间的距离


def calculate_mask_distance(mask):
    """
    计算两个板子之间的距离（自动识别方向）

    参数:
        mask: 分割掩码

    返回:
        (distance, middle_points, all_points, orientation): 距离、中间点、所有端点和方向
    """
    # 获取两个区域的端点和方向
    result = get_endpoints_from_mask(mask)

    if result is None:
        return None, None, None, None

    region1_endpoints, region2_endpoints, orientation = result

    if orientation == "horizontal":
        # 水平方向 - 左右板子
        left1, right1 = region1_endpoints  # 第一块板子的左右端点
        left2, right2 = region2_endpoints  # 第二块板子的左右端点

        # 只计算x坐标差值的绝对值 - 右板左端点和左板右端点之间的距离
        distance = abs(left2[0] - right1[0])

        # 用于显示的中间点
        middle_points = [right1, left2]
    else:
        # 垂直方向 - 上下板子
        top1, bottom1 = region1_endpoints  # 上方板子的上下端点
        top2, bottom2 = region2_endpoints  # 下方板子的上下端点

        # 计算y坐标差值的绝对值 - 下板顶端和上板底端之间的距离
        distance = abs(top2[1] - bottom1[1])

        # 用于显示的中间点
        middle_points = [bottom1, top2]

    # 所有端点
    all_points = [
        region1_endpoints[0],
        region1_endpoints[1],
        region2_endpoints[0],
        region2_endpoints[1],
    ]

    return distance, middle_points, all_points, orientation


export_dir = os.path.join(script_dir, "point_clouds")

try:
    os.makedirs(export_dir, mode=0o777, exist_ok=True)
    print(f"点云将导出到: {export_dir}")
except Exception as e:
    print(f"使用点云目录失败: {e}")

# 导出点云数据为PLY文件


def export_point_cloud(points, colors, filename):
    """
    导出点云数据为PLY文件

    参数:
        points: 点云坐标数组
        colors: 点云颜色数组
        filename: 输出文件名（绝对路径）

    返回:
        是否导出成功
    """
    if points is None or len(points) == 0:
        print("没有点云数据可导出")
        return False

    try:
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 对点云进行统计离群值滤波
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        # 应用坐标变换，使点云在CloudCompare中呈现正确的视角
        # 创建180度绕X轴旋转的变换矩阵（将Z轴朝上变为Z轴朝下）
        rotation = np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # Y轴反向  # Z轴反向

        # 将变换应用到点云
        points_np = np.asarray(pcd.points)
        transformed_points = np.dot(points_np, rotation.T)
        pcd.points = o3d.utility.Vector3dVector(transformed_points)

        # 确保目录存在（虽然我们已经创建了目录，但为了健壮性）
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # 保存为PLY格式
        success = o3d.io.write_point_cloud(filename, pcd)

        if success:
            print(f"点云成功导出到: {filename}")
            return True
        else:
            print(f"点云导出失败: 无法写入文件")
            return False
    except Exception as e:
        print(f"导出点云时出错: {e}")
        import traceback

        traceback.print_exc()
        return False


def calculate_laser_lines_distance_exclude_bends(ply_file_path=None, input_points=None, input_colors=None):
    """
    计算两条激光线之间的距离，同时排除垂直弯曲部分
    改进版：支持直接处理点云数据，仅保留"最近点对"、"投影距离"和"全局邻域"三个方法

    参数:
        ply_file_path: PLY文件路径，当为None时使用input_points和input_colors
        input_points: 直接输入的点云数据，避免文件IO
        input_colors: 直接输入的点云颜色数据

    返回:
        intersections: 交点坐标列表
        distances: 距离列表[最终距离, 最近点对距离, 投影距离, 全局邻域距离]
    """
    if ply_file_path:
        print(f"从文件计算激光线距离: {ply_file_path}")
    elif input_points is not None:
        print(f"直接处理点云数据计算激光线距离，点数: {len(input_points)}")
    else:
        print("错误: 未提供有效的点云数据")
        return None, None

    # 维护一个历史结果队列，用于平均和异常检测
    if not hasattr(calculate_laser_lines_distance_exclude_bends, "history"):
        calculate_laser_lines_distance_exclude_bends.history = []
    # 历史结果的最大长度 - 减少为3，提高动态响应性
    max_history = 3

    try:
        # 尝试导入必要的库
        import open3d as o3d
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA

        # 定义辅助函数:计算两点间的三维空间距离(考虑X、Y和Z轴)
        def calculate_horizontal_distance(p1, p2):
            # 完整3D欧氏距离计算
            return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

        # 获取点云数据 - 支持文件和直接输入
        if ply_file_path:
            # 从文件加载点云
            pcd = o3d.io.read_point_cloud(ply_file_path)
            if not pcd or len(pcd.points) == 0:
                print("点云加载失败或为空")
                return None, None
            # 获取点云数据
            points = np.asarray(pcd.points)
        else:
            # 直接使用输入的点云数据
            points = input_points
            # 为输入的点云数据创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if input_colors is not None:
                # 确保颜色值在[0,1]范围内
                normalized_colors = input_colors.astype(np.float64) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(normalized_colors)

        if len(points) < 10:
            print(f"点云中的点数量不足: {len(points)}")
            return None, None

        # 点云预处理：使用多级滤波，移除离群点
        # 1. 体素下采样，减少点云密度提高处理速度
        pcd_down = pcd.voxel_down_sample(voxel_size=2.0)  # 使用2mm体素大小提高速度

        # 2. 统计滤波，移除离群点 - 使用更宽松的参数加快速度
        pcd_filtered, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=15, std_ratio=2.5)

        # 获取滤波后的点云
        laser_points = np.asarray(pcd_filtered.points)

        # 确保有足够的点进行处理
        if len(laser_points) < 15:
            print(f"滤波后点云点数不足: {len(laser_points)}")
            return None, None

        # 自适应DBSCAN参数 - 根据点云密度调整
        point_cloud_density = len(
            laser_points) / (np.max(laser_points, axis=0) - np.min(laser_points, axis=0)).prod()

        # 使用固定的eps值，避免复杂的自适应计算，提高速度和稳定性
        fixed_eps = 20.0  # 参考之前的有效代码使用更大的eps值

        # 最小样本数也使用固定值
        fixed_min_samples = 5  # 使用5个点作为最小聚类要求

        print(
            f"使用自适应聚类参数: eps={fixed_eps:.2f}, min_samples={fixed_min_samples}")

        # DBSCAN聚类
        clustering = DBSCAN(
            eps=fixed_eps, min_samples=fixed_min_samples).fit(laser_points)
        labels = clustering.labels_

        # 获取有效的簇（排除噪声点，噪声点标签为-1）
        unique_labels = np.unique(labels)
        valid_clusters = [label for label in unique_labels if label != -1]

        # 计算每个簇的点数和基本属性
        cluster_sizes = []
        for label in valid_clusters:
            cluster_points = laser_points[labels == label]
            point_count = len(cluster_points)
            # 计算簇的体积（通过边界框）以评估其重要性
            if point_count > 0:
                bbox_volume = np.prod(
                    np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0))
                cluster_sizes.append((label, point_count, bbox_volume))

        print(f"找到 {len(valid_clusters)} 个簇")

        # 根据簇处理情况分情况处理
        if len(valid_clusters) == 0:
            print("未找到有效簇")
            return None, None
        elif len(valid_clusters) == 1:
            print("只找到一个簇，尝试细分该簇")
            # 选择该簇的点，尝试再次聚类，使用更小的eps值
            single_cluster_points = laser_points[labels == valid_clusters[0]]

            if len(single_cluster_points) > fixed_min_samples * 3:  # 确保有足够的点来细分
                # 为细分聚类设置更小的eps参数
                sub_eps = fixed_eps / 3.0  # 更激进的细分
                sub_min_samples = 3  # 更小的最小样本数以便形成簇

                # 尝试更精细的DBSCAN聚类
                sub_clustering = DBSCAN(eps=sub_eps, min_samples=sub_min_samples).fit(
                    single_cluster_points)
                sub_labels = sub_clustering.labels_
                sub_valid_clusters = [
                    label for label in np.unique(sub_labels) if label != -1]

                # 如果细分后得到多个簇
                if len(sub_valid_clusters) >= 2:
                    print(f"成功将单个簇细分为 {len(sub_valid_clusters)} 个子簇")

                    # 为原始点云重新分配标签
                    new_labels = np.full_like(labels, -1)
                    mask = labels == valid_clusters[0]
                    for i, sl in enumerate(sub_labels):
                        if sl != -1:
                            # 找到对应的原始点索引
                            orig_idx = np.where(mask)[0][i]
                            new_labels[orig_idx] = sl

                    labels = new_labels
                    valid_clusters = sub_valid_clusters

                     # 对有效簇按大小排序
                    cluster_sizes = []
                    for label in valid_clusters:
                        cluster_points = laser_points[labels == label]
                        point_count = len(cluster_points)
                        if point_count > 0:
                            bbox_volume = np.prod(
                                np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0))
                            cluster_sizes.append(
                                (label, point_count, bbox_volume))
                else:
                    print("无法细分单个簇，无法测量距离")
                    return None, None
            else:
                print("单个簇的点数太少，无法细分")
                return None, None

        # 如果有超过2个簇，只选择最主要的两个
        if len(cluster_sizes) > 2:
            print(f"找到{len(cluster_sizes)}个簇，将使用最主要的两个簇")

            # 按点数排序，同时考虑簇的空间分布（体积）
            # 这里使用加权方法：0.7*点数比例 + 0.3*体积比例
            max_points = max(
                [size for _, size, _ in cluster_sizes]) if cluster_sizes else 1
            max_volume = max([vol for _, _, vol in cluster_sizes]
                             ) if cluster_sizes else 1

            # 计算每个簇的重要性得分
            importance_scores = []
            for label, size, volume in cluster_sizes:
                # 归一化后加权计算总分
                normalized_size = size / max_points
                normalized_volume = volume / max_volume
                score = 0.7 * normalized_size + 0.3 * normalized_volume
                importance_scores.append((label, score))

            # 根据重要性得分排序
            importance_scores.sort(key=lambda x: x[1], reverse=True)

            # 选择得分最高的两个簇
            chosen_clusters = [importance_scores[0]
                               [0], importance_scores[1][0]]
            print(f"选择了两个最主要的簇: {chosen_clusters}")

            # 过滤掉所有其他簇（将它们的标签设为-1）
            for i in range(len(labels)):
                if labels[i] != -1 and labels[i] not in chosen_clusters:
                    labels[i] = -1

            # 更新有效簇列表
            valid_clusters = chosen_clusters
        elif len(cluster_sizes) == 2:
            # 直接使用两个簇
            chosen_clusters = [cluster_sizes[0][0], cluster_sizes[1][0]]
        else:
            print("没有足够的有效簇来计算距离")
            return None, None

        # 提取两个主要簇的点
        line1_points = laser_points[labels == chosen_clusters[0]]
        line2_points = laser_points[labels == chosen_clusters[1]]

        # 验证两条线是足够分开的 - 计算质心距离
        centroid1 = np.mean(line1_points, axis=0)
        centroid2 = np.mean(line2_points, axis=0)
        centroid_distance = calculate_horizontal_distance(centroid1, centroid2)

        # 如果质心距离太小，可能是错误地将同一条线分成了两部分
        if centroid_distance < 5.0:  # 5mm阈值
            print(
                f"警告: 两条激光线的质心距离过小 ({centroid_distance:.2f}mm)，可能是同一条线被错误分成两部分")
            return None, None

        # 计算空间方向 - 使用PCA分析
        # 处理激光线1
        pca1 = PCA(n_components=1)
        pca1.fit(line1_points)
        main_dir1 = pca1.components_[0]

        # 沿主方向投影点
        proj1 = np.dot(
            line1_points - np.mean(line1_points, axis=0), main_dir1)

        # 确保main_dir的方向一致性
        if proj1.max() - proj1.min() < 0:
            main_dir1 = -main_dir1
            proj1 = -proj1

        # 排序投影
        sorted_indices1 = np.argsort(proj1)
        sorted_points1 = line1_points[sorted_indices1]
        sorted_proj1 = proj1[sorted_indices1]

        # 处理激光线2
        pca2 = PCA(n_components=1)
        pca2.fit(line2_points)
        main_dir2 = pca2.components_[0]

        # 沿主方向投影点
        proj2 = np.dot(
            line2_points - np.mean(line2_points, axis=0), main_dir2)

        # 确保main_dir的方向一致性
        if proj2.max() - proj2.min() < 0:
            main_dir2 = -main_dir2
            proj2 = -proj2

        # 排序投影
        sorted_indices2 = np.argsort(proj2)
        sorted_points2 = line2_points[sorted_indices2]
        sorted_proj2 = proj2[sorted_indices2]

        # 增强的端点检测算法 - 更鲁棒，能处理噪声
        def find_valid_endpoints(sorted_points, sorted_proj, main_dir, line_num):
            if len(sorted_points) < 5:
                # 点数太少，直接返回首尾点
                return sorted_points[0], sorted_points[-1]

            # 计算有效端点的进阶方法

            # 1. 计算投影的差分 - 用于识别弯曲点
            proj_diffs = np.diff(sorted_proj)

            # 2. 计算更稳健的阈值 - 使用四分位距
            q25 = np.percentile(proj_diffs, 25)
            q75 = np.percentile(proj_diffs, 75)
            iqr = q75 - q25  # 四分位距

            # 根据簇的大小采用不同的检测策略
            if len(sorted_points) < 50:  # 较小的簇
                # 对于小簇，使用更保守的阈值，主要关注端点本身的质量
                threshold = q75 + 1.2 * iqr

                # 使用平滑的窗口过滤极端点以降低噪声影响
                window_size = max(3, int(len(sorted_points) * 0.05))
                start_idx = window_size  # 跳过最前端的一些点，减少噪声影响
                end_idx = len(sorted_points) - window_size - 1  # 同理跳过末端的一些点

                # 如果簇太小，使用首尾点
                if end_idx <= start_idx:
                    return sorted_points[0], sorted_points[-1]

                # 计算平滑后的起点和终点
                start_point = np.mean(sorted_points[:window_size], axis=0)
                end_point = np.mean(sorted_points[-window_size:], axis=0)

            else:  # 较大的簇
                # 对于大簇，可以使用更复杂的分析
                threshold = q75 + 2.0 * iqr

                # 适应性端点检测 - 根据簇的形状确定合适的检测区域
                # 前30%和后30%是检测端点的合适区域
                first_segment_end = int(len(sorted_points) * 0.3)
                last_segment_start = int(len(sorted_points) * 0.7)

                # 在首尾部分寻找可能的弯曲点
                front_diffs = proj_diffs[:first_segment_end]
                rear_diffs = proj_diffs[last_segment_start:]

                # 找出可能的弯曲点下标
                front_threshold_idxs = np.where(front_diffs > threshold)[0]
                rear_threshold_idxs = np.where(rear_diffs > threshold)[0]

                # 如果找到了弯曲点，使用它们作为端点的候选
                # 在前段区域，我们选择最远的弯曲点
                if len(front_threshold_idxs) > 0:
                    # 调整为实际索引（考虑到diff操作减少了一个元素）
                    start_idx = front_threshold_idxs[-1] + 1
                else:
                    # 如果没找到，使用平滑的起点
                    start_idx = int(len(sorted_points) * 0.05)  # 前5%

                # 在后段区域，我们选择最早的弯曲点
                if len(rear_threshold_idxs) > 0:
                    # 这里需要调整后段索引
                    end_idx = last_segment_start + rear_threshold_idxs[0]
                else:
                    # 如果没找到，使用平滑的终点
                    end_idx = int(len(sorted_points) * 0.95)  # 后5%

                # 获取实际的端点
                start_point = sorted_points[start_idx]
                end_point = sorted_points[end_idx]

            # 验证端点之间的距离
            endpoint_distance = np.linalg.norm(end_point - start_point)
            min_distance_threshold = 5.0

            # 对较小的簇使用更小的阈值
            if len(sorted_points) < 50:
                min_distance_threshold = 3.0

            # 如果端点距离太小，可能有问题，回退到简单的首尾点
            if endpoint_distance < min_distance_threshold:
                print(f"警告: 线段{line_num}检测到的端点距离过小，使用首尾点")
                return sorted_points[0], sorted_points[-1]

            return start_point, end_point

        # 获取两条线的有效端点
        start_point1, end_point1 = find_valid_endpoints(
            sorted_points1, sorted_proj1, main_dir1, 1)
        start_point2, end_point2 = find_valid_endpoints(
            sorted_points2, sorted_proj2, main_dir2, 2)

        # 查找两个点云簇之间的最近点对
        def find_closest_points_between_clusters(cluster1_points, cluster2_points, max_pairs=5):
            """
            查找两个点云簇之间最近的点对

            参数:
                cluster1_points: 第一个点云簇的点
                cluster2_points: 第二个点云簇的点
                max_pairs: 返回的最近点对数量

            返回:
                closest_pairs: 最近点对列表[(p1, p2, distance), ...]
                min_distance: 最小距离
            """
            # 为了提高效率，如果点云太大，进行下采样
            if len(cluster1_points) > 100:
                sample_ratio1 = min(1.0, 100 / len(cluster1_points))
                indices1 = np.random.choice(len(cluster1_points),
                                            size=int(
                                                len(cluster1_points) * sample_ratio1),
                                            replace=False)
                sampled_points1 = cluster1_points[indices1]
            else:
                sampled_points1 = cluster1_points

            if len(cluster2_points) > 100:
                sample_ratio2 = min(1.0, 100 / len(cluster2_points))
                indices2 = np.random.choice(len(cluster2_points),
                                            size=int(
                                                len(cluster2_points) * sample_ratio2),
                                            replace=False)
                sampled_points2 = cluster2_points[indices2]
            else:
                sampled_points2 = cluster2_points

            # 计算所有点对之间的距离
            all_pairs = []
            for i, p1 in enumerate(sampled_points1):
                for j, p2 in enumerate(sampled_points2):
                    dist = calculate_horizontal_distance(p1, p2)
                    all_pairs.append((p1, p2, dist))

            # 按距离排序
            all_pairs.sort(key=lambda x: x[2])

            # 返回前max_pairs个最近点对
            closest_pairs = all_pairs[:max_pairs]
            min_distance = closest_pairs[0][2] if closest_pairs else float(
                'inf')

            return closest_pairs, min_distance

        # 找到两个点云簇之间的最近点对
        closest_point_pairs, nearest_distance = find_closest_points_between_clusters(
            line1_points, line2_points)

        # 使用最近点对作为主要的距离测量
        print(f"最近点对距离 = {nearest_distance:.2f}mm")

        # 为了保持代码兼容性，保留closest_endpoints变量但不再计算端点距离
        closest_endpoints = None

        # 邻域点法 - 计算近邻点间的距离
        def compute_neighborhood_distance(points1, points2, k=10):
            """计算两组点之间k个最近点对的平均距离"""
            all_distances = []
            for p1 in points1:
                for p2 in points2:
                    all_distances.append(calculate_horizontal_distance(p1, p2))

            all_distances.sort()
            # 确保不会越界
            k = min(k, len(all_distances))
            if k == 0:
                return float('inf')

            return np.mean(all_distances[:k])

        # 计算两种邻域距离
        # 1. 全局邻域 - 使用两条线的所有点
        global_neighborhood_distance = compute_neighborhood_distance(
            line1_points, line2_points, k=15)

        # 2. 局部邻域 - 只使用端点附近的点
        def get_points_near_endpoint(all_points, endpoint, radius=10.0):
            """获取端点附近的点"""
            distances = np.array(
                [calculate_horizontal_distance(p, endpoint) for p in all_points])
            nearby_indices = np.where(distances < radius)[0]

            # 确保至少有几个点
            if len(nearby_indices) < 5:
                nearby_indices = np.argsort(
                    distances)[:min(5, len(all_points))]

            return all_points[nearby_indices]

        # 不再计算局部邻域距离
        local_neighborhood_distance = float('inf')

        # 投影距离 - 计算两线段之间的投影距离
        # 计算两线段的主要方向的夹角
        angle_between_lines = np.arccos(
            np.clip(np.dot(main_dir1, main_dir2), -1.0, 1.0)) * 180 / np.pi
        print(f"两线段主方向夹角 = {angle_between_lines:.2f}度")

        # 基于夹角选择合适的投影方法
        if angle_between_lines < 20 or angle_between_lines > 160:
            # 线段近似平行，使用简单的最短点对距离
            min_point_distance = float('inf')
            for p1 in line1_points:
                for p2 in line2_points:
                    dist = calculate_horizontal_distance(p1, p2)
                    if dist < min_point_distance:
                        min_point_distance = dist
            projection_distance = min_point_distance
        else:
            # 线段有明显角度，计算两线段之间的连接方向
            # 使用质心之间的连接向量
            centroid1 = np.mean(line1_points, axis=0)
            centroid2 = np.mean(line2_points, axis=0)

            connecting_vector = centroid2 - centroid1
            connecting_length = np.linalg.norm(connecting_vector)

            # 归一化连接向量
            if connecting_length > 0:
                connecting_vector = connecting_vector / connecting_length

            # 计算两线段的平均距离（考虑投影）
            total_distances = []

            # 从第一条线到第二条线的距离
            for p1 in line1_points:
                min_dist = float('inf')
                for p2 in line2_points:
                    dist = calculate_horizontal_distance(p1, p2)
                    min_dist = min(min_dist, dist)
                total_distances.append(min_dist)

            # 从第二条线到第一条线的距离
            for p2 in line2_points:
                min_dist = float('inf')
                for p1 in line1_points:
                    dist = calculate_horizontal_distance(p2, p1)
                    min_dist = min(min_dist, dist)
                total_distances.append(min_dist)

            # 使用较小的距离作为投影距离
            if total_distances:
                # 使用一种更稳健的平均方法 - 排序后取前30%的平均值
                total_distances.sort()
                k = max(1, int(len(total_distances) * 0.3))
                projection_distance = np.mean(total_distances[:k])
            else:
                projection_distance = float('inf')

        print(f"投影距离 = {projection_distance:.2f}mm")

        # 融合距离计算 - 只保留三种方法
        # 检查各种距离是否有效
        valid_nearest = nearest_distance < float('inf')
        valid_projection = projection_distance < float('inf')
        valid_global_nbh = global_neighborhood_distance < float('inf')

        # 计算融合距离 - 主要基于三种方法
        if valid_nearest:
            if valid_projection and valid_global_nbh:
                # 三种方法都可用时的融合策略
                area_distance = (0.60 * nearest_distance +
                                 0.25 * projection_distance +
                                 0.15 * global_neighborhood_distance)

                print(
                    f"加权融合距离 = {area_distance:.2f}mm [最近点对:{nearest_distance:.2f} + 投影:{projection_distance:.2f} + 全局邻域:{global_neighborhood_distance:.2f}]")

            elif valid_projection:
                # 最近点对和投影可用
                area_distance = 0.7 * nearest_distance + 0.3 * projection_distance
                print(
                    f"加权融合距离 = {area_distance:.2f}mm [最近点对:{nearest_distance:.2f} + 投影:{projection_distance:.2f}]")

            elif valid_global_nbh:
                # 最近点对和全局邻域可用
                area_distance = 0.7 * nearest_distance + 0.3 * global_neighborhood_distance
                print(
                    f"加权融合距离 = {area_distance:.2f}mm [最近点对:{nearest_distance:.2f} + 全局邻域:{global_neighborhood_distance:.2f}]")

            else:
                # 只有最近点对可用
                area_distance = nearest_distance
                print(f"使用最近点对距离 = {area_distance:.2f}mm")

        elif valid_projection and valid_global_nbh:
            # 投影和全局邻域可用
            area_distance = 0.6 * projection_distance + 0.4 * global_neighborhood_distance
            print(
                f"加权融合距离 = {area_distance:.2f}mm [投影:{projection_distance:.2f} + 全局邻域:{global_neighborhood_distance:.2f}]")

        elif valid_projection:
            # 只有投影可用
            area_distance = projection_distance
            print(f"使用投影距离 = {area_distance:.2f}mm")

        elif valid_global_nbh:
            # 只有全局邻域可用
            area_distance = global_neighborhood_distance
            print(f"使用全局邻域距离 = {area_distance:.2f}mm")

        else:
            # 没有有效距离
            print("无法计算有效距离")
            return None, None

        # 历史结果处理和异常检测
            calculate_laser_lines_distance_exclude_bends.history.append(
                area_distance)
            if len(calculate_laser_lines_distance_exclude_bends.history) > max_history:
                calculate_laser_lines_distance_exclude_bends.history = calculate_laser_lines_distance_exclude_bends.history[
                    -max_history:]

        # 如果有足够的历史数据，进行稳定性检测
            if len(calculate_laser_lines_distance_exclude_bends.history) > 2:
                current_result = area_distance
            previous_results = calculate_laser_lines_distance_exclude_bends.history[:-1]

            # 计算历史结果的统计指标
            history_mean = np.mean(previous_results)
            history_std = np.std(previous_results)

            # 计算当前结果与历史平均的偏差
            deviation = abs(current_result - history_mean)

            # 如果偏差超过3个标准差，可能是异常值
            if history_std > 0 and deviation > 3 * history_std:
                print(
                    f"检测到可能的异常值: {current_result:.2f}mm (历史平均: {history_mean:.2f}mm, 偏差: {deviation:.2f}mm)")

                # 使用加权平均替代极端值，但保持变化趋势
                if current_result > history_mean:
                    # 上升趋势，但限制幅度
                    area_distance = history_mean + \
                        min(deviation, 1.5 * history_std)
                else:
                    # 下降趋势，但限制幅度
                    area_distance = history_mean - \
                        min(deviation, 1.5 * history_std)

                print(f"调整后的距离: {area_distance:.2f}mm")

        # 提取最近点对的端点用于可视化
        nearest_points = []
        if closest_point_pairs and len(closest_point_pairs) > 0:
            # 取最近的几个点对
            for i, (p1, p2, _) in enumerate(closest_point_pairs[:2]):
                nearest_points.append(p1)
                nearest_points.append(p2)

        # 不再使用传统端点，只使用最近点对
        all_endpoints = nearest_points

        # 构建并返回结果
        # distances数组仅包含：[最终距离, 最近点对距离, 投影距离, 全局邻域距离]
        distances = [
            area_distance,
            nearest_distance,
            projection_distance,
            global_neighborhood_distance
        ]

        return all_endpoints, distances

    except ImportError as e:
        print(f"导入必要的库失败: {e}")
        return None, None
    except Exception as e:
        print(f"计算激光线距离时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None

# 添加多次测量函数，位于calculate_laser_lines_distance_exclude_bends函数之后


def calculate_multiple_measures(ply_file_path=None, num_measures=5, real_time=True, time_between=0.5, points=None, colors=None, depth_scale=None, depth_intrin=None):
    """
    执行多次测量并进行统计分析，提高测量准确性和可靠性
    支持并行处理以提高效率

    参数:
        ply_file_path: PLY文件路径，可选
        num_measures: 测量次数，默认值从10改为5，减少总时间
        real_time: 是否实时测量，设为True时会忽略points和colors参数，强制从相机获取新帧
        time_between: 两次测量之间的时间间隔(秒)，仅在非并行模式有效
        points: 点云数据，仅在real_time=False时使用
        colors: 点云颜色数据，仅在real_time=False时使用
        depth_scale: 深度比例，实时模式使用
        depth_intrin: 深度摄像头内参，实时模式使用

    返回:
        final_distance: 最终测量结果(mm)
        std_dev: 标准差，表示测量稳定性
        valid_measures: 有效测量值列表
    """
    # 导入必要的库，确保在所有代码路径中都可访问
    import numpy as np
    import time
    import os
    import tempfile
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    import warnings
    import sys
    from io import StringIO
    import pyrealsense2 as rs
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 存储所有有效的测量结果
    valid_measures = []
    temp_files = []

    # 确保有访问相机的pipeline和align对象
    global pipeline, align, current_mask, depth_filter_completed, depth_filtered_mask, process_depth_frame_parallel, export_point_cloud

    # 保存原始标准输出，确保最终正确恢复
    orig_stdout = sys.stdout

    # 定义处理单个点云数据的函数，用于多线程处理
    def process_point_cloud_data(index, current_points, current_colors):
        # 初始化临时文件路径和测量结果
        temp_ply = None
        measurement_result = None

        # 保存原始标准输出，确保后续恢复
        thread_orig_stdout = sys.stdout

        try:
            if current_points is None or len(current_points) < 10:
                print(
                    f"警告: 第{index+1}次测量点云数据不足({'无数据' if current_points is None else len(current_points)}点)，跳过")
                return index, None, None

            # 使用线程特定的StringIO对象捕获输出
            thread_output = StringIO()

            # 执行测量并捕获输出
            endpoints = None
            distances = None
            try:
                # 重定向标准输出到StringIO
                sys.stdout = thread_output
                
                # 直接使用点云数据计算距离，避免创建临时文件
                endpoints, distances = calculate_laser_lines_distance_exclude_bends(
                    ply_file_path=None,
                    input_points=current_points,
                    input_colors=current_colors
                )
                
                # 恢复标准输出
                sys.stdout = thread_orig_stdout
                
                # 获取捕获的输出
                output_text = thread_output.getvalue()
                
                # 只提取关键信息显示到控制台
                important_lines = []
                for line in output_text.split('\n'):
                    # 保留包含关键词的行
                    if any(keyword in line for keyword in ['距离', 'mm', '方法', '警告', '错误']):
                        important_lines.append(line)
                
                # 显示重要信息，但控制输出量
                if index <= 1 or len(important_lines) <= 3:  # 第一次测量或信息很少时显示全部
                    for line in important_lines:
                        print(f"[测量{index+1}] {line}")
                else:
                    # 后续测量只显示关键结果行
                    for line in important_lines[-3:]:  # 只显示最后3行重要信息
                        if '距离' in line:
                            print(f"[测量{index+1}] {line}")
            except Exception as e:
                # 确保恢复标准输出
                sys.stdout = thread_orig_stdout
                print(f"计算点云距离时出错: {e}")

            # 处理测量结果
            if distances and len(distances) > 0 and distances[0] is not None:
                measurement_result = distances[0]
                # 显示所有测量的详细结果
                if len(distances) >= 3:  # 如果有完整的三种方法结果
                    print(f"测量 {index+1}/{num_measures}: 距离 = {measurement_result:.2f} mm [最近点对:{distances[1]:.2f} + 投影:{distances[2]:.2f} + 全局邻域:{distances[3]:.2f}]")
                else:
                    print(f"测量 {index+1}/{num_measures}: 距离 = {measurement_result:.2f} mm")
                
                # 最后一次测量显示完成信息
                if index == num_measures - 1:
                    print(f"已完成 {num_measures} 次测量")

        except Exception as e:
            print(f"第 {index+1} 次测量出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 确保标准输出在任何情况下都被正确恢复
            sys.stdout = thread_orig_stdout

        return index, measurement_result, temp_ply

    # 原来的单次测量函数，仅用于非实时模式
    def perform_single_measurement(index, use_pcd=None, use_colors=None):
        # 初始化单次测量的临时文件路径
        temp_ply = None
        measurement_result = None

        try:
            if not real_time:
                # 非实时模式
                if ply_file_path:
                    # 直接使用PLY文件进行测量
                    intersections, distances = calculate_laser_lines_distance_exclude_bends(
                        ply_file_path)

                    if distances and len(distances) > 0 and distances[0] is not None:
                        measurement_result = distances[0]
                        print(
                            f"第 {index+1}/{num_measures} 次测量: 距离 = {measurement_result:.2f} mm")

                elif use_pcd is not None and use_colors is not None:
                    # 直接使用点云数据计算，不创建临时文件
                    endpoints, distances = calculate_laser_lines_distance_exclude_bends(
                        ply_file_path=None,
                        input_points=use_pcd,
                        input_colors=use_colors
                    )

                if distances and len(distances) > 0 and distances[0] is not None:
                    measurement_result = distances[0]
                    print(
                        f"第 {index+1}/{num_measures} 次测量: 距离 = {measurement_result:.2f} mm")

        except Exception as e:
            print(f"第 {index+1} 次测量出错: {e}")
            import traceback
            traceback.print_exc()

        return index, measurement_result, temp_ply

    try:
        # 临时禁用matplotlib中文字体警告
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=".*CJK.*")
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=".*Glyph.*missing from font.*")

        if 'pipeline' not in globals() or 'align' not in globals() and real_time:
            print("错误: 无法访问相机数据流，请确保已初始化pipeline")
            return None, None, []

        start_time = time.time()
        print(
            f"开始执行{num_measures}次{'实时' if real_time else '非实时'}测量{'(并行处理)' if num_measures > 1 else ''}")

        if not real_time:
            # 如果使用的是传入的点云数据，创建副本
            original_points = points.copy() if points is not None else None
            original_colors = colors.copy() if colors is not None else None

            # 使用线程池并行执行多次测量
            with ThreadPoolExecutor(max_workers=min(num_measures, os.cpu_count() or 4)) as executor:
                # 提交所有测量任务
                futures = []
                for i in range(num_measures):
                    futures.append(executor.submit(
                        perform_single_measurement, i,
                        use_pcd=original_points,
                        use_colors=original_colors
                    ))

                # 收集结果并显示进度
                completed_count = 0
                print(f"\r测量进度: {completed_count}/{num_measures} 完成", end="")

                for future in as_completed(futures):
                    index, result, temp_file = future.result()
                    if temp_file:
                        temp_files.append(temp_file)
                    if result is not None:
                        valid_measures.append(result)

                    # 更新并显示进度
                    completed_count += 1
                    elapsed_time = time.time() - start_time
                    remaining_time = (elapsed_time / completed_count) * (
                        num_measures - completed_count) if completed_count > 0 else 0
                    print(
                        f"\r测量进度: {completed_count}/{num_measures} 完成 | 用时: {elapsed_time:.1f}秒 | 预计剩余: {remaining_time:.1f}秒", end="")
        else:
            # 实时模式 - 先串行获取所有帧，然后并行处理
            all_points_data = []
            all_colors_data = []

            print("正在获取实时相机帧...")
            sys.stdout.flush()

            # 减少内存帧存储时间，使用更紧凑的获取帧策略
            frame_collection_timeout = 5  # 帧收集超时时间减少到5秒，加快整体速度
            start_frame_collection = time.time()

            # 串行获取所有帧 (现在默认只需3帧)
            for i in range(num_measures):
                try:
                    # 设置获取帧的超时检测
                    if time.time() - start_frame_collection > frame_collection_timeout:
                        print(f"\n警告: 帧获取超时，已收集{i}帧，将使用这些帧继续处理")
                        break

                    print(f"\r正在获取第 {i+1}/{num_measures} 帧...", end="")
                    sys.stdout.flush()  # 确保立即显示进度

                    # 获取新的相机帧
                    frames = pipeline.wait_for_frames(timeout_ms=500)  # 增加超时设置
                    if not frames:
                        print(f"\n警告: 获取第{i+1}帧超时，跳过")
                        continue

                    aligned_frames = align.process(frames)

                    # 提取深度和颜色帧
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not depth_frame or not color_frame:
                        print(f"\n警告: 第{i+1}次测量无法获取有效帧，跳过")
                        all_points_data.append(None)
                        all_colors_data.append(None)
                        continue

                    # 处理深度帧(应用滤波器)
                    filtered_depth = process_depth_frame_parallel(depth_frame)

                    # 获取图像数据
                    depth_image = np.asanyarray(filtered_depth.get_data())
                    rgb_image = np.asanyarray(color_frame.get_data())

                    # 获取深度相机内参
                    current_depth_intrin = filtered_depth.profile.as_video_stream_profile().intrinsics
                    current_depth_scale = depth_scale if depth_scale is not None else depth_frame.get_distance_scale()

                    # 生成点云数据
                    current_points = None
                    current_colors = None

                    if current_mask is not None:
                        # 生成激光区域点云数据
                        try:
                            current_points, current_colors, _ = create_point_cloud(
                                depth_image,
                                rgb_image,
                                current_depth_scale,
                                current_depth_intrin,
                                mask=current_mask,
                                laser_only=True
                            )
                        except Exception as e:
                            print(f"\n警告: 生成点云数据时出错: {e}")
                            all_points_data.append(None)
                            all_colors_data.append(None)
                            continue

                        # 应用深度过滤
                        if depth_filter_completed and depth_filtered_mask is not None:
                            filtered_point_indices = []

                            # 性能优化：减少不必要的投影计算，使用更高效的过滤
                            # 1. 使用矢量化操作代替循环
                            point_meters = current_points / 1000.0

                            # 2. 批量处理点云而不是逐点处理 - 进一步减少处理点数
                            points_to_process = min(
                                len(current_points), 3000)  # 降低处理点数，进一步提高速度
                            step = max(1, len(current_points) //
                                       points_to_process)  # 采样步长

                            for p_idx in range(0, len(current_points), step):
                                point_m = point_meters[p_idx]
                                try:
                                    pixel = rs.rs2_project_point_to_pixel(
                                        current_depth_intrin, point_m
                                    )
                                    x, y = int(pixel[0]), int(pixel[1])

                                    if (0 <= x < depth_filtered_mask.shape[1] and
                                        0 <= y < depth_filtered_mask.shape[0] and
                                            depth_filtered_mask[y, x] > 0):
                                        filtered_point_indices.append(p_idx)
                                except Exception:
                                    continue

                            if filtered_point_indices:
                                current_points = current_points[filtered_point_indices]
                                current_colors = current_colors[filtered_point_indices]
                                if i == 0:  # 只显示第一次的信息
                                    print(
                                        f"\n使用 {len(current_points)} 个深度过滤区域内的点")
                                    sys.stdout.flush()  # 确保立即显示
                    else:
                        print(f"\n警告: 第{i+1}次测量没有有效掩码，无法生成点云")
                        all_points_data.append(None)
                        all_colors_data.append(None)
                        continue

                    if current_points is None or len(current_points) < 10:
                        print(
                            f"\n警告: 第{i+1}次测量点云数据不足({'无数据' if current_points is None else len(current_points)}点)，跳过")
                        all_points_data.append(None)
                        all_colors_data.append(None)
                    else:
                        # 保存有效点云数据
                        all_points_data.append(current_points.copy())
                        all_colors_data.append(current_colors.copy())

                    # 进一步减少延迟时间以加快总体速度
                    time.sleep(0.02)  # 从0.05进一步减少到0.02

                except Exception as e:
                    print(f"\n获取第 {i+1} 帧出错: {e}")
                    all_points_data.append(None)
                    all_colors_data.append(None)

            print("\n所有相机帧获取完成，开始并行处理...")
            sys.stdout.flush()  # 确保终端立即显示

            # 使用线程池并行处理所有有效帧
            with ThreadPoolExecutor(max_workers=min(num_measures, os.cpu_count() or 4)) as executor:
                # 提交所有处理任务
                futures = []
                # 使用实际获取到的帧数量，而不是预期的num_measures
                actual_frames = len(all_points_data)
                for i in range(actual_frames):
                    if i < len(all_points_data) and i < len(all_colors_data) and all_points_data[i] is not None and all_colors_data[i] is not None:
                        futures.append(executor.submit(
                            process_point_cloud_data, i,
                            all_points_data[i],
                            all_colors_data[i]
                        ))

                # 收集结果并显示进度
                completed_count = 0
                valid_futures = len(futures)
                if valid_futures > 0:
                    print(
                        f"\r处理进度: {completed_count}/{valid_futures} 完成", end="")
                    sys.stdout.flush()  # 确保进度立即显示

                    for future in as_completed(futures):
                        index, result, temp_file = future.result()
                        if temp_file:
                            temp_files.append(temp_file)
                        if result is not None:
                            valid_measures.append(result)

                        # 更新并显示进度
                        completed_count += 1
                        elapsed_time = time.time() - start_time
                        remaining_time = (elapsed_time / completed_count) * (
                            valid_futures - completed_count) if completed_count > 0 else 0
                        print(
                            f"\r处理进度: {completed_count}/{valid_futures} 完成 | 用时: {elapsed_time:.1f}秒 | 预计剩余: {remaining_time:.1f}秒", end="")
                        sys.stdout.flush()  # 确保进度立即显示
                else:
                    print("没有有效的点云数据可以处理")

        print("\n")  # 完成所有测量后换行
        sys.stdout.flush()  # 确保输出已显示

        # 显示总耗时
        total_time = time.time() - start_time
        print(
            f"全部测量完成，总耗时: {total_time:.2f}秒，平均每次测量: {total_time/num_measures:.2f}秒")
        sys.stdout.flush()  # 确保输出已显示

        # 如果没有有效测量，返回None
        if not valid_measures:
            print("没有获得有效的测量结果")
            return None, None, []

        # 检查是否有足够的有效测量结果
        if len(valid_measures) < 2:
            print(f"警告: 有效测量结果数量不足 ({len(valid_measures)} < 2)")
            return np.mean(valid_measures) if valid_measures else None, 0.0, valid_measures

        # 统计处理
        mean_value = np.mean(valid_measures)
        std_dev = np.std(valid_measures)

        print(f"\n====== 测量统计分析 ======")
        print(f"全部测量结果: {[f'{x:.2f}' for x in valid_measures]}")
        print(f"有效测量次数: {len(valid_measures)}/{num_measures}")
        print(f"平均值: {mean_value:.2f}mm, 标准差: {std_dev:.2f}mm")
        print(
            f"最小值: {min(valid_measures):.2f}mm, 最大值: {max(valid_measures):.2f}mm")
        print(f"测量精度: {(std_dev/mean_value*100):.2f}% (标准差/平均值)")
        sys.stdout.flush()  # 确保分析结果立即显示

        # 如果有足够的样本，去除异常值
        if len(valid_measures) >= 5:
            sorted_distances = sorted(valid_measures)
            filtered_distances = sorted_distances[1:-1]  # 去除最大最小值

            filtered_mean = np.mean(filtered_distances)
            filtered_std = np.std(filtered_distances)

            print(f"\n去除最值后结果: {[f'{x:.2f}' for x in filtered_distances]}")
            print(f"处理后平均值: {filtered_mean:.2f}mm, 标准差: {filtered_std:.2f}mm")
            print(f"优化后精度: {(filtered_std/filtered_mean*100):.2f}%")
            print(f"====== 测量完成 ======\n")
            sys.stdout.flush()  # 确保输出已显示

            return filtered_mean, filtered_std, valid_measures
        else:
            print(
                f"\n完成{len(valid_measures)}次有效测量，最终距离：{mean_value:.2f}mm，标准差：{std_dev:.2f}mm")
            print(f"====== 测量完成 ======\n")
            sys.stdout.flush()  # 确保输出已显示
            return mean_value, std_dev, valid_measures

    except Exception as e:
        print(f"多次测量过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, valid_measures
    finally:
        # 确保在任何情况下都恢复原始标准输出
        sys.stdout = orig_stdout

        # 由于我们现在直接处理点云数据，几乎不创建临时文件
        # 但仍然保留临时文件清理代码，以防万一
        if temp_files:
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                        # 强制释放文件系统资源
                        if 'gc' in sys.modules:
                            import gc
                            gc.collect()  # 强制垃圾回收
                except Exception:
                    pass

        # 重置相机状态，确保资源被正确释放
        if real_time:
            try:
                # 清空相机缓冲区
                if 'pipeline' in globals() and pipeline:
                    for _ in range(5):  # 清空几帧缓冲
                        try:
                            frames = pipeline.wait_for_frames(timeout_ms=100)
                        except Exception:
                            break

                print("相机资源已重置")
            except Exception as e:
                print(f"重置相机状态时出错: {e}")


def parallel_dbscan(points, eps=20.0, min_samples=5):
    """
    并行化的DBSCAN聚类，使用多进程优化性能

    参数:
        points: 需要聚类的点云数据
        eps: DBSCAN的邻域半径参数
        min_samples: DBSCAN的最小样本数量参数

    返回:
        labels: 聚类标签数组
    """
    # 如果点云数据过多，进行随机下采样
    max_points_for_clustering = 10000  # 聚类的最大点数限制
    if len(points) > max_points_for_clustering:
        indices = np.random.choice(
            len(points), max_points_for_clustering, replace=False)
        points = points[indices]
        print(f"点云数据过多，已下采样到{max_points_for_clustering}个点以加速聚类")

    if len(points) < min_samples * 2:
        # 点数太少，不值得并行处理
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_

    # 减少并行计算的分块数，从而减少进程间通信开销
    used_cores = min(4, NUM_CORES)  # 限制最大使用核心数为4或系统核心数（取较小值）

    # 至少需要used_cores个点才能分割
    if len(points) < used_cores * min_samples:
        # 点数不够多，使用普通DBSCAN
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_

    # 确保进程池已初始化
    if not initialize_process_pool_if_needed() or process_pool is None:
        print("未能初始化进程池，使用单线程DBSCAN")
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_

    # 将点分割为used_cores个子集
    chunk_size = len(points) // used_cores
    chunks = []
    for i in range(used_cores):
        if i == used_cores - 1:
            # 最后一个块包含所有剩余点
            chunk = points[i * chunk_size:]
        else:
            chunk = points[i * chunk_size: (i + 1) * chunk_size]
        # 添加必要的额外参数
        chunks.append((chunk, min_samples, eps))

    # 使用进程池并行执行
    try:
        # 使用超时参数以防止进程卡住
        chunk_labels = process_pool.map_async(
            dbscan_process_chunk, chunks).get(timeout=10)
    except Exception as e:
        print(f"并行DBSCAN失败: {e}，回退到单线程处理")
        # 回退到单线程处理
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_

    # 合并结果，需要调整标签值以避免冲突
    merged_labels = np.zeros(len(points), dtype=int)
    offset = 0
    max_label = -1  # 开始于-1（噪声点）

    for i, labels in enumerate(chunk_labels):
        # 找出当前块中最大的非噪声标签
        if len(labels) > 0 and np.max(labels) > -1:
            curr_max = np.max(labels)

            # 调整非噪声点的标签
            adjusted_labels = labels.copy()
            mask = adjusted_labels > -1
            adjusted_labels[mask] = adjusted_labels[mask] + max_label + 1

            # 更新最大标签
            max_label = max_label + curr_max + 1
        else:
            adjusted_labels = labels

        # 复制到对应位置
        chunk_end = offset + len(labels)
        if i == used_cores - 1:
            # 最后一个块可能大小不同
            merged_labels[offset:] = adjusted_labels
        else:
            merged_labels[offset:chunk_end] = adjusted_labels
        offset = chunk_end

    return merged_labels


# ====================
# 主处理循环
# ====================
try:
    # 初始化进程池，避免在import时创建
    process_pool = multiprocessing.Pool(processes=max(1, NUM_CORES - 1))

    predictor = build_sam2_camera_predictor(
        model_cfg, sam2_checkpoint, device=device)
    frame_count = 1
    prev_time = time.time()
    fps_list = []  # 用于计算平均FPS

    # 是否执行分割处理的标志
    do_segmentation = False  # 初始状态为False
    waiting_for_clicks = False  # 是否等待用户点击
    running = True  # 添加回running变量
    tracking = False  # 是否正在跟踪

    # 分割结果存储
    current_mask = None  # 当前帧的分割掩码
    obj_id = 1  # 对象ID计数器
    is_first_frame = True  # 标记是否是第一帧

    # 存储端点和距离计算结果
    distance_info = None  # 存储距离信息
    all_endpoints = None  # 存储所有端点
    middle_endpoints = None  # 存储中间端点
    calculated_distance = None  # 计算得到的距离

    # 用户点击的提示点
    prompt_points = []  # 点的坐标列表
    prompt_labels = []  # 点的标签列表（1表示正点，0表示负点）

    # 添加处理状态和超时控制
    segmentation_in_progress = False
    segmentation_start_time = 0
    segmentation_timeout = 10  # 分割处理超时时间（秒）

    # 点云显示控制
    show_pointcloud = False
    pointcloud_window_created = False  # 跟踪窗口是否已创建
    use_open3d_visualization = have_x11  # 根据X11可用性决定是否使用Open3D

    # 启动点云可视化线程
    if use_open3d_visualization:
        print("将使用Open3D进行3D点云可视化")
        pointcloud_thread_running = True
        pointcloud_thread = threading.Thread(
            target=pointcloud_visualization_thread)
        pointcloud_thread.daemon = True
        pointcloud_thread.start()
    else:
        print("无法使用Open3D进行可视化，点云功能将不可用")

    # 创建主要窗口
    cv2.namedWindow(WINDOW_RGB)
    cv2.namedWindow(WINDOW_DEPTH)

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        global prompt_points, prompt_labels, waiting_for_clicks
        if not waiting_for_clicks:
            return

        if event == cv2.EVENT_LBUTTONDOWN:  # 左键添加正点
            prompt_points.append([x, y])
            prompt_labels.append(1)  # 正点标签为1
            print(f"添加正点 ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键添加负点
            prompt_points.append([x, y])
            prompt_labels.append(0)  # 负点标签为0
            print(f"添加负点 ({x}, {y})")

    # 添加鼠标监听
    cv2.setMouseCallback(WINDOW_RGB, mouse_callback)

    # 预分配内存
    display_image = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_colormap_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # 在主循环前设置默认显示点云
    show_pointcloud = False
    pointcloud_window_created = False

    # 在初始化部分添加方向检测变量
    detected_orientation = "horizontal"  # 默认水平方向
    last_orientation = None  # 跟踪上一次的方向，用于检测变化

    # 添加深度过滤状态跟踪变量
    depth_filter_completed = False  # 标记深度过滤是否完成
    depth_filtered_mask = None  # 存储深度过滤后的掩码
    depth_filtered_image = None  # 存储深度过滤后的图像
    display_distance_info = False  # 控制是否显示距离信息

    # 用于自动检测的变量
    auto_detect_mode = False  # 默认关闭自动检测
    auto_detect_interval = 30  # 自动检测的帧间隔
    auto_detect_counter = 0  # 自动检测计数器
    last_auto_detect_time = 0  # 上次自动检测的时间
    last_detect_status = None  # 上次检测状态
    last_status_print_time = 0  # 上次状态打印时间

    # 添加变量用于跟踪上一次的点数
    last_point_count = 0

    # 初始化IMU变量
    accel_sample = None
    gyro_sample = None

    # 添加全局变量用于存储Roll, Pitch和重力方向
    global_roll = None
    global_pitch = None
    global_gravity_dir = None

    # 添加全局变量用于存储深度过滤后保留的区域ID
    # global_selected_region_ids = None

    # 提供同步锁以避免多线程问题
    last_point_count_lock = threading.Lock()

    # 添加聚类历史追踪类
    class ClusterTracker:
        def __init__(self):
            self.clusters_history = []
            self.last_change_time = 0

    # 创建全局追踪器实例
    cluster_tracker = ClusterTracker()

    # 多进程初始化标志
    process_pool_initialized = False

    # 确保进程池已初始化
    def initialize_process_pool_if_needed():
        global process_pool, process_pool_initialized
        if not process_pool_initialized:
            try:
                process_pool = multiprocessing.Pool(
                    processes=max(1, NUM_CORES - 1))
                process_pool_initialized = True
                print(f"进程池初始化完成，共{max(1, NUM_CORES-1)}个工作进程")
            except Exception as e:
                print(f"初始化进程池失败: {e}")
                process_pool = None
        return process_pool_initialized

    # 创建用于并行处理深度滤波的多阶段管道
    def process_depth_frame_parallel(depth_frame):
        # 帧计数器，用于跳过部分帧的完整处理
        if not hasattr(process_depth_frame_parallel, "frame_count"):
            process_depth_frame_parallel.frame_count = 0
        process_depth_frame_parallel.frame_count += 1

        # 缓存结果，避免重复计算
        if not hasattr(process_depth_frame_parallel, "last_result"):
            process_depth_frame_parallel.last_result = None
            process_depth_frame_parallel.last_frame_number = -1

        # 检查是否可以重用上一帧的结果
        current_frame_number = depth_frame.get_frame_number()
        if (process_depth_frame_parallel.last_result is not None and
                current_frame_number == process_depth_frame_parallel.last_frame_number):
            return process_depth_frame_parallel.last_result

        # 使用线程池并行应用滤波器
        def apply_filter_chain(depth):
            # 帧计数器，用于自适应滤波
            if not hasattr(apply_filter_chain, "frame_count"):
                apply_filter_chain.frame_count = 0
            apply_filter_chain.frame_count += 1

            # 基本处理 - 对所有帧都执行
            # 1. 首先应用抽取滤波 - 降低噪声
            depth = decimation.process(depth)
            # 2. 应用阈值滤波 - 过滤掉不可靠的深度值
            depth = threshold_filter.process(depth)

            # 完整处理 - 仅对关键帧执行（每5帧或测量帧）
            # 判断是否为需要完整处理的帧
            is_key_frame = (apply_filter_chain.frame_count % 5 == 0)

            if is_key_frame:
                # 3. 应用视差变换 - 在视差域中处理可提高精度
                depth = disparity.process(depth)  # 转换为视差
                # 4. 应用空间滤波 - 平滑深度图像
                depth = spatial.process(depth)
                # 5. 应用时间滤波 - 减少时间噪声
                depth = temporal.process(depth)
                # 6. 从视差转回深度
                depth = disparity_to_depth.process(depth)
                # 7. 最后填充空洞
                depth = hole_filling.process(depth)

            return depth

        # 使用线程池处理滤波链
        filtered_depth = thread_pool.submit(
            apply_filter_chain, depth_frame).result()

        # 缓存结果
        process_depth_frame_parallel.last_result = filtered_depth
        process_depth_frame_parallel.last_frame_number = current_frame_number

        return filtered_depth

    # 新增函数：从IMU获取相机姿态
    def get_camera_orientation_from_imu(accel_sample, gravity_filter=0.95):
        """
        使用加速度计数据计算相机姿态

        参数:
            accel_sample: 加速度计样本
            gravity_filter: 低通滤波系数，用于稳定重力方向

        返回:
            (roll, pitch)角度，单位为弧度
        """
        # 静态全局变量用于滤波
        if not hasattr(get_camera_orientation_from_imu, "filtered_gravity"):
            get_camera_orientation_from_imu.filtered_gravity = np.array([
                                                                        0.0, 0.0, 9.8])

        # 获取加速度数据 (假设静止状态下主要是重力)
        accel_data = np.array([accel_sample.x, accel_sample.y, accel_sample.z])

        # 应用低通滤波，平滑加速度数据
        get_camera_orientation_from_imu.filtered_gravity = (
            gravity_filter * get_camera_orientation_from_imu.filtered_gravity
            + (1 - gravity_filter) * accel_data
        )

        gravity = get_camera_orientation_from_imu.filtered_gravity
        gravity_norm = np.linalg.norm(gravity)

        # 确保重力不为零
        if gravity_norm < 0.1:
            return 0.0, 0.0

        # 标准化重力向量
        gravity = gravity / gravity_norm

        # 计算roll和pitch (假设z轴向下为正方向)
        # roll: 绕x轴旋转
        # pitch: 绕y轴旋转
        roll = np.arctan2(gravity[1], gravity[2])
        pitch = np.arctan2(-gravity[0],
                           np.sqrt(gravity[1] ** 2 + gravity[2] ** 2))

        return roll, pitch

    def transform_point_cloud_to_laser_coordinate_with_imu(
        points, colors, detected_orientation, middle_endpoints=None, accel_sample=None
    ):
        """
        将点云从相机坐标系转换到世界坐标系，使用IMU数据确定重力方向
        坐标系原点保持在相机位置，但XYZ轴垂直于现实世界

        参数:
            points: 点云坐标数组
            colors: 点云颜色数组
            detected_orientation: 检测到的激光线方向 ('horizontal' 或 'vertical')，用于调试信息
            middle_endpoints: 激光线的关键点 (不再用于设置原点)
            accel_sample: 加速度计样本，用于获取重力方向

        返回:
            transformed_points: 转换后的点云坐标
            colors: 原始颜色数据(不变)
            transformation_matrix: 坐标变换矩阵
        """
        if points is None or len(points) < 10:
            return points, colors, None

        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 相机默认的前方向和上方向（如果没有IMU数据）
        default_front = np.array([0, 0, 1])  # 相机Z轴方向，指向前方
        default_up = np.array([0, -1, 0])  # 相机Y轴反方向，指向上方

        # 如果有IMU数据，使用IMU计算相机姿态
        camera_rotation = np.eye(3)  # 默认为单位矩阵
        gravity_dir = np.array([0, -1, 0])  # 默认重力方向

        if accel_sample is not None:
            try:
                # 计算相机的姿态角
                roll, pitch = get_camera_orientation_from_imu(accel_sample)

                # 使用IMU变化检测器判断是否需要打印
                accel_data = np.array(
                    [accel_sample.x, accel_sample.y, accel_sample.z])
                should_print = imu_detector.update(roll, pitch, accel_data)

                if should_print:
                    print(
                        f"相机姿态: Roll={np.degrees(roll):.2f}°, Pitch={np.degrees(pitch):.2f}°"
                    )

                # 创建相机姿态旋转矩阵
                # 先绕X轴旋转(roll)
                roll_mat = np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)],
                    ]
                )

                # 再绕Y轴旋转(pitch)
                pitch_mat = np.array(
                    [
                        [np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)],
                    ]
                )

                # 组合旋转
                camera_rotation = np.dot(pitch_mat, roll_mat)

                # 重力方向（加速度计测量值的反方向）
                gravity_dir = -np.array(
                    [accel_sample.x, accel_sample.y, accel_sample.z]
                )
                gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)
            except Exception as e:
                print(f"IMU姿态计算错误: {e}")
                import traceback

                traceback.print_exc()

        # 建立世界坐标系
        # 1. Z轴: 垂直于地面，与重力方向相反
        z_axis = -gravity_dir  # Z轴向上
        z_axis = z_axis / np.linalg.norm(z_axis)

        # 2. X轴: 定义为相机前方向在与Z轴垂直的平面上的投影
        # 相机前方向
        front_dir = np.dot(camera_rotation, default_front)
        # X轴需要与Z轴垂直
        x_axis = front_dir - np.dot(front_dir, z_axis) * z_axis
        # 如果x_axis接近零向量，使用默认方向
        if np.linalg.norm(x_axis) < 0.001:
            # 使用一个垂直于Z轴的向量作为X轴
            if abs(z_axis[0]) < 0.9:  # 如果Z轴不太接近X轴
                x_axis = (
                    np.array([1, 0, 0]) -
                    np.dot(np.array([1, 0, 0]), z_axis) * z_axis
                )
            else:
                x_axis = (
                    np.array([0, 1, 0]) -
                    np.dot(np.array([0, 1, 0]), z_axis) * z_axis
                )

        x_axis = x_axis / np.linalg.norm(x_axis)

        # 3. Y轴: X轴和Z轴的叉积，保证坐标系是右手系
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # 改动: 设置坐标系原点为相机原点 (0,0,0)
        origin = np.array([0.0, 0.0, 0.0])

        # 移除这个重复的print语句
        # print(f"坐标系原点设置为相机位置: {origin}")

        # 构建4x4变换矩阵
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = -np.dot(rotation, origin)

        # 应用变换
        homogeneous_points = np.ones((len(points), 4))
        homogeneous_points[:, :3] = points
        transformed_homogeneous = np.dot(homogeneous_points, transformation.T)
        transformed_points = transformed_homogeneous[:, :3]

        # 声明静态变量来跟踪上次的轴信息
        if not hasattr(transform_point_cloud_to_laser_coordinate_with_imu, "last_axes"):
            transform_point_cloud_to_laser_coordinate_with_imu.last_axes = None

        # 计算轴的变化程度
        current_axes = (x_axis, y_axis, z_axis)
        axes_changed = False

        if transform_point_cloud_to_laser_coordinate_with_imu.last_axes is not None:
            # 计算轴向量的差异
            x_diff = np.linalg.norm(np.array(
                x_axis) - np.array(transform_point_cloud_to_laser_coordinate_with_imu.last_axes[0]))
            y_diff = np.linalg.norm(np.array(
                y_axis) - np.array(transform_point_cloud_to_laser_coordinate_with_imu.last_axes[1]))
            z_diff = np.linalg.norm(np.array(
                z_axis) - np.array(transform_point_cloud_to_laser_coordinate_with_imu.last_axes[2]))

            # 如果任何轴的变化超过阈值，认为发生了明显变化
            if max(x_diff, y_diff, z_diff) > 0.05:  # 5%的变化阈值
                axes_changed = True
        else:
            # 首次运行，认为发生了变化
            axes_changed = True

        # 更新上次的轴信息
        transform_point_cloud_to_laser_coordinate_with_imu.last_axes = current_axes

        # 只有当轴发生明显变化或IMU检测器允许打印时才输出
        if imu_detector.has_significant_change:
            print(f"坐标系原点设置为相机位置: {origin}")
            if axes_changed:
                print(f"坐标系转换完成: X轴={x_axis}, Y轴={y_axis}, Z轴={z_axis}")

        return transformed_points, colors, transformation

    while running:  # 使用running变量代替True
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # 获取各部分图像数据
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 使用并行处理优化深度滤波链
        filtered_depth = process_depth_frame_parallel(depth_frame)

        # 获取滤波后的深度图像数据
        depth_image = np.asanyarray(filtered_depth.get_data())
        rgb_image = np.asanyarray(color_frame.get_data())

        # 获取深度相机内参
        depth_intrin = filtered_depth.profile.as_video_stream_profile().intrinsics

        # 获取图像尺寸，定义width和height变量
        height, width = rgb_image.shape[:2]

        # 直接使用预分配的内存
        display_image[:] = rgb_image

        # 每3帧更新一次点云 (减少计算负担)
        # 点云更新频率控制，提高性能
        if frame_count % 8 == 0 and show_pointcloud:  # 从每3帧改为每8帧更新一次点云
            try:
                # 重置深度过滤状态
                depth_filter_completed = False
                depth_filtered_mask = None
                depth_filtered_image = None
                display_distance_info = False

                # 生成激光区域点云数据，包括激光点标记
                points, colors, laser_mask = create_point_cloud(
                    depth_image,
                    rgb_image,
                    depth_scale,
                    filtered_depth.profile.as_video_stream_profile().intrinsics,
                    mask=current_mask,
                    laser_only=True,
                )

                # 检查是否有足够的点云数据
                if len(points) < 10:
                    if imu_detector.has_significant_change:
                        print("点云数据不足，无法进行深度过滤")
                    # 删除这个continue语句，改为设置一个默认的空点云数据，避免卡住
                    # 更新激光区域点云数据，即使是空数据
                    pointcloud_data["points"] = points
                    pointcloud_data["colors"] = colors
                    pointcloud_data["updated"] = True
                    # 如果在全场景模式下，确保也有全场景数据
                    if not export_laser_only and (pointcloud_data["full_points"] is None or len(pointcloud_data["full_points"]) == 0):
                        # 生成全场景点云数据
                        full_points, full_colors, _ = create_point_cloud(
                            depth_image,
                            rgb_image,
                            depth_scale,
                            depth_intrin,
                            mask=None,  # 不使用掩码
                            laser_only=False,  # 不限制为激光区域
                        )
                        # 更新全场景点云数据
                        pointcloud_data["full_points"] = full_points
                        pointcloud_data["full_colors"] = full_colors
                else:
                    # 原有的点云处理逻辑，只在有足够数据时执行
                    # 提取绿色激光点
                    if np.sum(laser_mask) > 10:  # 确保有足够的激光点
                        if imu_detector.has_significant_change:
                            print(f"检测到 {np.sum(laser_mask)} 个绿色激光点")
                        laser_points = points[laser_mask]
                        laser_colors = colors[laser_mask]
                    else:
                        if imu_detector.has_significant_change:
                            print(
                                f"绿色激光点不足 ({np.sum(laser_mask)}), 尝试使用掩码区域内的所有点"
                            )
                        laser_points = points
                        laser_colors = colors

                    # 使用转换的坐标系过滤点云
                    if (
                        middle_endpoints is not None
                        and detected_orientation is not None
                        and len(laser_points) > 10
                    ):
                        # 进行坐标系转换，使用IMU数据辅助
                        # 确保accel_sample已定义
                        local_accel_sample = (
                            accel_sample if "accel_sample" in locals() else None
                        )

                        transformed_points, filtered_colors, transformation_matrix = (
                            transform_point_cloud_to_laser_coordinate_with_imu(
                                laser_points,
                                laser_colors,
                                detected_orientation,
                                middle_endpoints,
                                local_accel_sample,
                            )
                        )

                        if transformed_points is not None and len(transformed_points) > 10:
                            # 获取Z轴值
                            z_values = transformed_points[:, 2]
                            z_min, z_max = np.min(z_values), np.max(z_values)
                            z_range = z_max - z_min

                            # 根据IMU变化检测结果决定是否打印
                            should_print = imu_detector.has_significant_change
                            if should_print:
                                print(
                                    f"进行简化的深度过滤，共有 {len(transformed_points)} 个点，深度值范围:{z_min:.2f}~{z_max:.2f}mm"
                                )

                            # SAM提供了多个激光区域，直接分析SAM的分割结果
                            if current_mask is not None:
                                # 转换为二值图像
                                binary_mask = (current_mask > 0).astype(
                                    np.uint8) * 255

                                # 找到连通区域
                                num_labels, labels, stats, centroids = (
                                    cv2.connectedComponentsWithStats(
                                        binary_mask)
                                )

                                if should_print:
                                    print(f"检测到 {num_labels-1} 个连通区域")

                                if num_labels <= 1:  # 如果没有连通区域
                                    if should_print:
                                        print("未检测到有效连通区域")
                                    continue

                                # 过滤掉太小的区域
                                min_area = 100
                                valid_regions = []

                                for i in range(1, num_labels):  # 跳过背景(0)
                                    area = stats[i, cv2.CC_STAT_AREA]
                                    if area >= min_area:
                                        # 为每个区域创建单独的掩码
                                        region_mask = (
                                            labels == i).astype(np.uint8)
                                        valid_regions.append(
                                            (i, area, region_mask,
                                             centroids[i])
                                        )
                                if should_print:
                                    print(
                                        f"面积筛选后保留 {len(valid_regions)} 个有效区域")

                                if len(valid_regions) == 0:
                                    if should_print:
                                        print("没有足够大的区域")
                                    continue

                                # 计算每个区域的深度统计信息
                                region_depth_stats = []  # 从region_z_stats改为region_depth_stats

                                # 从深度图直接获取深度数据
                                depth_data = np.asarray(depth_frame.get_data())
                                depth_values_mm = depth_data * depth_scale * 1000  # 转换为毫米

                                for region_id, area, region_mask, centroid in valid_regions:
                                    # 直接从深度图获取深度值
                                    # 提取区域掩码内的深度值
                                    depth_in_region = depth_values_mm[region_mask > 0]
                                    # 过滤掉0值（无效深度）
                                    valid_depth = depth_in_region[depth_in_region > 0]

                                    if len(valid_depth) >= 5:  # 至少有5个有效深度值
                                        # 从z_mean改为depth_mean
                                        depth_mean = np.mean(valid_depth)
                                        # 从z_median改为depth_median
                                        depth_median = np.median(valid_depth)
                                        # 从z_min改为depth_min
                                        depth_min = np.min(valid_depth)
                                        # 从z_max改为depth_max
                                        depth_max = np.max(valid_depth)

                                        region_depth_stats.append(  # 从region_z_stats改为region_depth_stats
                                            {
                                                "region_id": region_id,
                                                "area": area,
                                                "depth_mean": depth_mean,  # 从"z_mean"改为"depth_mean"
                                                "depth_median": depth_median,  # 从"z_median"改为"depth_median"
                                                "depth_min": depth_min,  # 从"z_min"改为"depth_min"
                                                "depth_max": depth_max,  # 从"z_max"改为"depth_max"
                                                "mask": region_mask,
                                                "centroid": centroid,
                                            }
                                        )

                                        if should_print:
                                            print(
                                                f"区域 {region_id}: 面积={area}, 深度范围={depth_min:.2f}~{depth_max:.2f}mm, 深度平均值={depth_mean:.2f}mm"
                                            )
                                    elif should_print:
                                        print(f"警告：区域 {region_id} 无法获取足够的深度数据")

                                # 区域选择逻辑
                                if region_depth_stats:  # 从region_z_stats改为region_depth_stats
                                    # 深度值越小，表示物体离相机越近（高度越高）
                                    # 从"z_mean"改为"depth_mean"
                                    region_depth_stats.sort(
                                        key=lambda x: x["depth_mean"])

                                    # 选择深度值最小（高度最高）的两个区域，而不是固定区域ID
                                    # 从region_z_stats改为region_depth_stats
                                    top_regions = region_depth_stats[:2]
                                    # 将区域ID列表进行排序后再输出
                                    sorted_region_ids = sorted(
                                        [region['region_id'] for region in top_regions])

                                    # 缓存上次输出的区域ID，只在变化时输出
                                    if not hasattr(transform_point_cloud_to_laser_coordinate_with_imu, "last_region_ids") or transform_point_cloud_to_laser_coordinate_with_imu.last_region_ids != sorted_region_ids:
                                        if imu_detector.has_significant_change:
                                            print(
                                                f"选择高度最高的区域：区域 {sorted_region_ids}")
                                        transform_point_cloud_to_laser_coordinate_with_imu.last_region_ids = sorted_region_ids

                                    # 创建包含所有区域的列表，用于显示
                                    all_regions = region_depth_stats.copy()  # 从region_z_stats改为region_depth_stats

                                    # 创建所选区域的ID集合，方便后续判断
                                    selected_region_ids = {
                                        region["region_id"] for region in top_regions}

                                    # 创建最终掩码（只包含所选区域）
                                    final_mask = np.zeros_like(binary_mask)
                                    for region in top_regions:
                                        final_mask = cv2.bitwise_or(
                                            final_mask, region["mask"] * 255
                                        )

                                    # 创建彩色区域标记图像（所有区域都显示，但亮度不同）
                                    region_label_image = np.zeros(
                                        (binary_mask.shape[0],
                                         binary_mask.shape[1], 3),
                                        dtype=np.uint8,
                                    )

                                    # 添加所有区域（非所选区域使用较暗的颜色）
                                    for region in all_regions:
                                        # 确定区域颜色
                                        if region["region_id"] in selected_region_ids:
                                            # 选中区域使用绿色
                                            color = (0, 255, 0)  # 绿色
                                            text_color = (
                                                255, 255, 255)  # 白色文本
                                        else:
                                            # 非选中区域使用红色
                                            color = (0, 0, 255)  # 红色 (BGR格式)
                                            text_color = (
                                                255, 255, 255)  # 白色文本

                                        # 在标记图像中为该区域着色
                                        color_mask = np.zeros_like(
                                            region_label_image)
                                        color_mask[region["mask"] > 0] = color
                                        region_label_image = cv2.add(
                                            region_label_image, color_mask
                                        )

                                        # 在中心位置添加原始区域ID标签
                                        cx, cy = int(region["centroid"][0]), int(
                                            region["centroid"][1]
                                        )
                                        # 使用原始区域ID
                                        label_text = f"{region['region_id']}"
                                        cv2.putText(
                                            region_label_image,
                                            label_text,
                                            (cx, cy),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.0,
                                            text_color,
                                            2,
                                        )

                                    # 创建掩码叠加图像
                                    display_with_mask = display_image.copy()

                                    # 将非掩码区域变暗
                                    mask_bool = final_mask > 0
                                    # 半透明处理，保留30%的原始图像
                                    display_with_mask[~mask_bool] = (
                                        display_with_mask[~mask_bool] * 0.3).astype(np.uint8)  # 半透明背景

                                    # 叠加区域标记
                                    display_with_mask = cv2.addWeighted(
                                        display_with_mask, 0.7, region_label_image, 0.3, 0
                                    )

                                    # 更新图像和掩码
                                    depth_filtered_image = display_with_mask
                                    depth_filtered_mask = final_mask

                                    # 标记深度过滤已完成
                                    # depth_filter_completed = False
                                    # display_distance_info = False     #（调试用，记得删）
                                    depth_filter_completed = True
                                    display_distance_info = True

                                    if imu_detector.has_significant_change:
                                        print(
                                            f"深度过滤完成，保留了 {len(top_regions)} 个最高区域，掩码包含 {np.sum(mask_bool)} 个像素")
                                # 重新计算深度过滤后保留区域的关键点和距离
                                if len(selected_region_ids) >= 2:
                                    try:
                                        # 创建一个仅包含保留区域的掩码
                                        retained_mask = np.zeros_like(
                                            current_mask)
                                        for region_id in selected_region_ids:
                                            retained_mask[labels ==
                                                          region_id] = 255

                                            # 清空之前的端点和距离信息，防止旧信息被显示
                                            calculated_distance = None
                                            middle_endpoints = None
                                            all_endpoints = None

                                            # 基于过滤后的区域重新计算距离和关键点
                                            new_distance_result = calculate_mask_distance(
                                                retained_mask)
                                            if new_distance_result[0] is not None:
                                                # 更新关键点和距离信息，用于在深度过滤视图中显示
                                                calculated_distance = new_distance_result[0]
                                                middle_endpoints = new_distance_result[1]
                                                all_endpoints = new_distance_result[2]
                                                # 获取检测到的方向
                                                detected_orientation = new_distance_result[3]

                                                # 不再重复打印
                                                # print(f"深度过滤后重新计算的距离：{calculated_distance:.2f} 像素")
                                                # print(f"深度过滤后中间端点：{middle_endpoints}")

                                                # 缓存上一次输出的距离和端点
                                                last_distance = getattr(
                                                    imu_detector, 'last_distance', -1)
                                                last_endpoints = getattr(
                                                    imu_detector, 'last_endpoints', None)

                                                # 检查距离是否发生显著变化（大于0.5）或者IMU检测到明显变化
                                                distance_changed = abs(
                                                    calculated_distance - last_distance) > 0.5 if last_distance != -1 else True
                                                endpoints_changed = middle_endpoints != last_endpoints if last_endpoints is not None else True

                                                if distance_changed or endpoints_changed or imu_detector.has_significant_change:
                                                    print(
                                                        f"深度过滤后重新计算的距离：{calculated_distance:.2f} 像素")
                                                    print(
                                                        f"深度过滤后中间端点：{middle_endpoints}")
                                                    # 更新缓存
                                                    imu_detector.last_distance = calculated_distance
                                                    imu_detector.last_endpoints = middle_endpoints

                                                try:

                                                    # 获取关键点
                                                    # 第一个中间点
                                                    point1 = middle_endpoints[0]
                                                    # 第二个中间点
                                                    point2 = middle_endpoints[1]

                                                    # 只绘制关键的两个端点
                                                    # 绘制第一个端点
                                                    cv2.circle(
                                                        depth_filtered_image,
                                                        (int(point1[0]), int(
                                                            point1[1])),
                                                        5,
                                                        (0, 255, 0),  # 绿色
                                                        -1,
                                                    )

                                                    # 绘制第二个端点
                                                    cv2.circle(
                                                        depth_filtered_image,
                                                        (int(point2[0]), int(
                                                            point2[1])),
                                                        5,
                                                        (0, 255, 0),  # 绿色
                                                        -1,
                                                    )

                                                    # 根据方向创建虚拟点
                                                    if detected_orientation == "horizontal":
                                                        # 水平方向
                                                        virtual_point = (
                                                            int(point2[0]), int(point1[1]))
                                                    else:
                                                        # 垂直方向
                                                        virtual_point = (
                                                            int(point1[0]), int(point2[1]))

                                                    # 绘制连接线
                                                    # 第一条线 - 两点之间的直接连线
                                                    cv2.line(
                                                        depth_filtered_image,
                                                        (int(point1[0]), int(
                                                            point1[1])),
                                                        (int(point2[0]), int(
                                                            point2[1])),
                                                        (255, 0, 0),
                                                        2,
                                                    )  # 蓝色

                                                    # 第二条线 - 第一个点到虚拟点
                                                    cv2.line(
                                                        depth_filtered_image,
                                                        (int(point1[0]), int(
                                                            point1[1])),
                                                        virtual_point,
                                                        (0, 0, 255),
                                                        2,
                                                    )  # 红色

                                                    # 第三条线 - 第二个点到虚拟点
                                                    cv2.line(
                                                        depth_filtered_image,
                                                        (int(point2[0]), int(
                                                            point2[1])),
                                                        virtual_point,
                                                        (255, 0, 0),
                                                        2,
                                                    )  # 蓝色

                                                    if imu_detector.has_significant_change:
                                                        print(
                                                            "已在深度过滤图像上绘制端点和连线")
                                                except Exception as e:
                                                    print(
                                                        f"在深度过滤图像上绘制端点和连线失败: {e}")
                                                    import traceback
                                                    traceback.print_exc()
                                    except Exception as e:
                                        print(f"深度过滤后重新计算关键点失败: {e}")
                                        import traceback
                                        traceback.print_exc()

                    # 更新激光区域点云数据
                    if depth_filter_completed and depth_filtered_mask is not None:
                        # 如果已完成深度过滤，只使用保留区域内的点
                        # 先创建一个全局掩码（所有点的布尔索引）
                        filtered_point_indices = []

                        # 将每个点投影回2D空间，检查是否在掩码区域内
                        for i, point in enumerate(points):
                            # 将毫米单位转回米
                            point_meters = point / 1000.0
                            # 投影到像素坐标
                            try:
                                pixel = rs.rs2_project_point_to_pixel(
                                    depth_intrin, point_meters
                                )

                                x, y = int(pixel[0]), int(pixel[1])

                                # 检查像素是否在图像范围内且在掩码内
                                if (
                                    0 <= x < depth_filtered_mask.shape[1]
                                    and 0 <= y < depth_filtered_mask.shape[0]
                                    and depth_filtered_mask[y, x] > 0
                                ):
                                    filtered_point_indices.append(i)
                            except Exception as e:
                                continue

                        if filtered_point_indices:
                            # 提取这些点的坐标和颜色
                            filtered_points = points[filtered_point_indices]
                            filtered_colors = colors[filtered_point_indices]

                            # 更新点云数据为过滤后的点
                            pointcloud_data["points"] = filtered_points
                            pointcloud_data["colors"] = filtered_colors

                            # 静态变量跟踪上次的点数
                            if not hasattr(pointcloud_data, "last_filtered_point_count"):
                                pointcloud_data["last_filtered_point_count"] = 0

                                # 检查点数变化或IMU变化
                            point_count_changed = abs(
                                len(filtered_points) - pointcloud_data["last_filtered_point_count"]) > 50
                            if imu_detector.has_significant_change or point_count_changed:
                                print(
                                    f"更新点云显示，保留 {len(filtered_points)} 个深度过滤后的点"
                                )
                                # 更新上次的点数
                                pointcloud_data["last_filtered_point_count"] = len(
                                    filtered_points)
                    else:
                        # 如果没有点，仍使用原始点
                        pointcloud_data["points"] = points
                        pointcloud_data["colors"] = colors

                    # 全场景点云保持不变
                    if (
                        not export_laser_only or frame_count % 30 == 0
                    ):  # 减少全场景更新频率，因为它更消耗资源
                        # 生成全场景点云数据
                        full_points, full_colors, _ = create_point_cloud(
                            depth_image,
                            rgb_image,
                            depth_scale,
                            depth_intrin,
                            mask=None,  # 不使用掩码
                            laser_only=False,  # 不限制为激光区域
                        )

                        # 更新全场景点云数据
                        pointcloud_data["full_points"] = full_points
                        pointcloud_data["full_colors"] = full_colors

                    # 设置更新标志
                    pointcloud_data["updated"] = True

                    # 如果没有可显示的点，显示提示
                    if (export_laser_only and len(points) == 0) or (
                        not export_laser_only
                        and (
                            pointcloud_data["full_points"] is None
                            or len(pointcloud_data["full_points"]) == 0
                        )
                    ):
                        try:
                            blank_image = np.zeros(
                                (480, 640, 3), dtype=np.uint8)
                            mode_text = "激光区域" if export_laser_only else "全场景"
                            cv2.putText(
                                blank_image,
                                f"No points detected in {mode_text} mode",
                                (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 255),
                                2,
                            )
                            pointcloud_data["render_image"] = blank_image
                        except Exception as e:
                            print(f"生成点云失败: {e}")
                            import traceback

                            traceback.print_exc()
            except Exception as e:
                print(f"更新点云时出错: {e}")
                import traceback
                traceback.print_exc()

        # 如果等待点击，绘制已有的点
        if waiting_for_clicks:
            display_image = draw_points(
                display_image, prompt_points, prompt_labels)

        # 检查分割处理是否超时
        if (
            segmentation_in_progress
            and time.time() - segmentation_start_time > segmentation_timeout
        ):
            print(f"分割处理超时 (>{segmentation_timeout}秒)，已中止")
            segmentation_in_progress = False
            do_segmentation = False
            waiting_for_clicks = False

        # 仅当用户触发分割处理时才执行SAM2处理
        if do_segmentation and len(prompt_points) > 0 and not segmentation_in_progress:
            print("开始执行SAM2分割处理...")
            segmentation_in_progress = True
            segmentation_start_time = time.time()

            try:
                # 如果是第一帧，需要初始化
                if is_first_frame:
                    predictor.load_first_frame(rgb_image)
                    is_first_frame = False

                # 使用用户指定的点
                points = np.array(prompt_points, dtype=np.float32)
                labels = np.array(prompt_labels, np.int32)

                # 添加目标点
                result = predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

                # 打印返回值的类型和长度，帮助调试
                # print(
                #     f"add_new_prompt返回值类型: {type(result)}, 长度: {len(result) if isinstance(result, tuple) else 'N/A'}"
                # )

                # 检查返回值
                if isinstance(result, tuple) and len(result) >= 3:
                    _, out_obj_ids, out_mask_logits = result[:3]

                    # 获取当前帧的掩码
                    if out_mask_logits is not None and len(out_mask_logits) > 0:
                        # 确保掩码是二维的
                        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask.squeeze(0)

                        # 处理掩码 - 使用连通区域分析替代形态学操作
                        if mask is not None:
                            # 转换为二值图像
                            binary_mask = (mask > 0).astype(np.uint8) * 255

                            # 找到连通区域
                            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                                binary_mask)

                            # 创建输出掩码
                            output_mask = np.zeros_like(binary_mask)

                            # 保留大于最小面积的区域
                            min_area = 100
                            for i in range(1, num_labels):  # 跳过背景(0)
                                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                                    component = (labels == i).astype(
                                        np.uint8) * 255
                                    output_mask = cv2.bitwise_or(
                                        output_mask, component)

                            mask = output_mask

                        current_mask = mask

                        obj_id += 1  # 增加对象ID
                        tracking = True  # 开始跟踪
                        show_pointcloud = True  # 分割后自动显示点云

                        # 计算距离
                        if len(prompt_points) >= 2:  # 确保有足够的点来分割两条线
                            try:
                                # 计算距离
                                distance_result = calculate_mask_distance(mask)
                                if distance_result[0] is not None:  # 检查是否有有效结果
                                    # 存储结果供后续帧使用
                                    calculated_distance = distance_result[0]
                                    middle_endpoints = distance_result[1]
                                    all_endpoints = distance_result[2]
                                    new_orientation = distance_result[3]

                                    # 只有当方向发生变化时才输出提示
                                    if new_orientation != last_orientation:
                                        if new_orientation == "horizontal":
                                            print("自动检测到水平方向激光线")
                                        else:
                                            print("自动检测到垂直方向激光线")
                                        last_orientation = (
                                            new_orientation
                                        )  # 更新上一次的方向

                                    detected_orientation = (
                                        new_orientation
                                    )  # 更新当前方向
                            except Exception as e:
                                import traceback

                                traceback.print_exc()
                    else:
                        print("警告：未生成有效的掩码")
                else:
                    print("警告：分割返回值格式不正确")

                print("分割处理完成，开始跟踪")
                waiting_for_clicks = False  # 分割完成后退出点击模式
            except Exception as e:
                print(f"分割处理失败: {e}")
                import traceback

                traceback.print_exc()  # 打印完整的错误堆栈
                tracking = False  # 错误时停止跟踪
            finally:
                segmentation_in_progress = False  # 无论成功失败，都标记为处理完成
                do_segmentation = False  # 重置标志
                torch.cuda.empty_cache()  # 清理GPU内存

        if current_mask is not None:
            try:
                # 1. 首先显示SAM2分割结果 - 不包含距离信息
                display_result, _ = add_mask(display_image, current_mask, 0)
                display_image = display_result

                # 添加SAM分割标识
                cv2.putText(
                    display_image,
                    "SAM Segmentation View",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # 2. 如果深度过滤已完成，更新显示
                if (
                    depth_filter_completed
                    and depth_filtered_image is not None
                    and depth_filtered_mask is not None
                ):
                    # 使用深度过滤后的图像
                    display_image = depth_filtered_image

            except Exception as e:
                print(f"显示掩码失败: {e}")
                import traceback

                traceback.print_exc()

        # 如果正在跟踪，更新掩码
        if tracking and not segmentation_in_progress:
            try:
                # 使用track方法更新掩码
                track_result = predictor.track(rgb_image)

                # 检查返回值类型和长度
                if not isinstance(track_result, tuple):
                    print("跟踪返回结果不是元组")
                    tracking = False
                    continue

                if len(track_result) < 2:
                    print("跟踪返回结果长度不足")
                    tracking = False

                # 获取掩码
                out_mask_logits = track_result[1]  # 第二个返回值是掩码
                if out_mask_logits is None or len(out_mask_logits) == 0:
                    print("未获取到有效的掩码")
                    tracking = False
                    continue

                # 处理掩码
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                if len(mask.shape) == 3:
                    mask = mask.squeeze(0)

                # 处理掩码 - 使用连通区域分析替代形态学操作
                if mask is not None:
                    # 转换为二值图像
                    binary_mask = (mask > 0).astype(np.uint8) * 255

                    # 找到连通区域
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                        binary_mask)

                    # 创建输出掩码
                    output_mask = np.zeros_like(binary_mask)

                    # 保留大于最小面积的区域
                    min_area = 100
                    for i in range(1, num_labels):  # 跳过背景(0)
                        if stats[i, cv2.CC_STAT_AREA] >= min_area:
                            component = (labels == i).astype(np.uint8) * 255
                            output_mask = cv2.bitwise_or(
                                output_mask, component)

                    mask = output_mask

                current_mask = mask

                # 如果有新的掩码，重新计算距离
                try:
                    distance_result = calculate_mask_distance(current_mask)
                    if distance_result[0] is not None:
                        calculated_distance = distance_result[0]
                        middle_endpoints = distance_result[1]
                        all_endpoints = distance_result[2]
                        new_orientation = distance_result[3]  # 获取检测到的方向

                        # 只有当方向发生变化时才输出提示
                        if new_orientation != last_orientation:
                            if new_orientation == "horizontal":
                                print("自动检测到水平方向激光线")
                            else:
                                print("自动检测到垂直方向激光线")
                            last_orientation = new_orientation  # 更新上一次的方向

                        detected_orientation = new_orientation  # 更新当前方向
                except Exception as e:
                    print(f"更新距离计算失败: {e}")

            except Exception as e:
                print(f"跟踪失败: {e}")
                import traceback

                traceback.print_exc()  # 打印完整的错误堆栈
                tracking = False  # 跟踪失败时停止跟踪

        # 如果有分割结果，则显示分割结果
        if current_mask is not None and not do_segmentation:
            try:
                # 修改为处理add_mask返回的多个值
                display_result, _ = add_mask(display_image, current_mask, 0)
                display_image = display_result
            except Exception as e:
                print(f"显示掩码失败: {e}")
                import traceback

                traceback.print_exc()
                current_mask = None  # 重置掩码
                tracking = False  # 停止跟踪

        # 显示FPS信息
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        fps_list.append(fps)
        if len(fps_list) > 30:  # 保留最近30帧的FPS
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        prev_time = current_time

        # 显示FPS和IMU数据（Roll和Pitch）
        cv2.putText(display_image, f"FPS: {avg_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 如果有有效的Roll和Pitch数据
        if global_roll is not None and global_pitch is not None:
            cv2.putText(
                display_image,
                f"Roll: {np.degrees(global_roll):.1f}",
                (10, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display_image,
                f"Pitch: {np.degrees(global_pitch):.1f}",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # 绘制重力方向指示器（每帧都更新）
            if global_gravity_dir is not None:
                # 在图像右下角显示重力方向指示器
                arrow_length = 50  # 箭头长度
                # 获取图像尺寸
                image_height, image_width = display_image.shape[:2]
                center_x, center_y = image_width - 60, image_height - 40  # 右下角位置
                # 反向显示更直观
                end_x = int(center_x - global_gravity_dir[0] * arrow_length)
                end_y = int(center_y - global_gravity_dir[1] * arrow_length)

                # 绘制重力方向指示器
                cv2.circle(
                    display_image, (center_x, center_y), 30, (0, 255, 255), 1
                )  # 圆
                cv2.line(
                    display_image,
                    (center_x, center_y),
                    (end_x, end_y),
                    (0, 255, 255),
                    2,
                )  # 线
                cv2.putText(
                    display_image,
                    "G",
                    (end_x - 5, end_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )  # 标签

        # 如果深度过滤已完成，使用深度过滤后的图像替换display_image
        if depth_filter_completed and depth_filtered_image is not None:
            # 重要：只复制图像数据，不包含之前可能添加的文本
            display_image = depth_filtered_image.copy()

            # 在深度过滤后的图像上绘制距离信息
            if (
                display_distance_info
                and all_endpoints is not None
                and middle_endpoints is not None
                and calculated_distance is not None
            ):
                try:
                    # 注意：此处使用的端点和距离信息应该是深度过滤后重新计算的结果
                    # 获取关键点
                    point1 = middle_endpoints[0]  # 第一个中间点
                    point2 = middle_endpoints[1]  # 第二个中间点

                    # 根据方向创建虚拟点
                    if detected_orientation == "horizontal":
                        # 水平方向 - (第二块板子左端点的x，第一块板子右端点的y值)
                        virtual_point = (int(point2[0]), int(point1[1]))
                    else:
                        # 垂直方向 - (第一块板子底部端点的x，第二块板子顶部端点的y)
                        virtual_point = (int(point1[0]), int(point2[1]))

                    # 获取深度信息和3D坐标
                    # 确保坐标在图像范围内
                    point1_x, point1_y = min(
                        int(point1[0]), depth_image.shape[1] - 1
                    ), min(int(point1[1]), depth_image.shape[0] - 1)
                    point2_x, point2_y = min(
                        int(point2[0]), depth_image.shape[1] - 1
                    ), min(int(point2[1]), depth_image.shape[0] - 1)

                    # 获取深度值（毫米）
                    point1_depth = depth_image[point1_y, point1_x]
                    point2_depth = depth_image[point2_y, point2_x]

                    # 如果深度值有效，计算3D坐标
                    point1_3d = None
                    point2_3d = None
                    real_distance_3d = None

                    if point1_depth > 0 and point2_depth > 0:
                        try:
                            # 将像素坐标和深度值转换为3D坐标
                            point1_3d = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [
                                    point1_x, point1_y], point1_depth
                            )
                            point2_3d = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [
                                    point2_x, point2_y], point2_depth
                            )

                            # 计算3D空间中的欧几里得距离（毫米）
                            real_distance_3d = np.sqrt(
                                sum(
                                    [
                                        (point2_3d[i] - point1_3d[i]) ** 2
                                        for i in range(3)
                                    ]
                                )
                            )

                            # 根据方向计算主轴方向的距离
                            if detected_orientation == "horizontal":
                                main_axis_distance = abs(
                                    point2_3d[0] - point1_3d[0]
                                )  # X轴距离
                                axis_name = "X"
                            else:
                                main_axis_distance = abs(
                                    point2_3d[1] - point1_3d[1]
                                )  # Y轴距离
                                axis_name = "Y"

                        except Exception as e:
                            print(f"3D转换错误: {e}")
                            import traceback

                            traceback.print_exc()
                            point1_3d = None
                            point2_3d = None

                    # 显示深度过滤信息
                    # 使用简化的提示，不显示具体区域ID
                    cv2.putText(
                        display_image,
                        f"Depth-Filtered View ",
                        (width - 380, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                     # 显示方向信息
                    orientation_text = (
                        "Horizontal"
                        if detected_orientation == "horizontal"
                        else "Vertical"
                    )
                    cv2.putText(
                        display_image,
                        f"Direction: {orientation_text}",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                     # 显示2D距离（像素）
                    cv2.putText(
                        display_image,
                        f"2D Distance: {calculated_distance:.2f} pixels",
                        (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                except Exception as e:
                    print(f"深度过滤后绘制端点和连线失败: {e}")
                    import traceback
                    traceback.print_exc()

            # 在深度过滤图像上重新显示Roll、Pitch和重力指示器
            if global_roll is not None and global_pitch is not None:
                # 显示深度过滤信息
                if sorted_region_ids is not None:
                    cv2.putText(
                        display_image,
                        f"Depth-Filtered View",
                        (width - 380, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                # 显示Roll和Pitch信息
                cv2.putText(
                    display_image,
                    f"Roll: {np.degrees(global_roll):.1f}",
                    (10, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_image,
                    f"Pitch: {np.degrees(global_pitch):.1f}",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # 重新绘制重力方向指示器
                if global_gravity_dir is not None:
                    # 在图像右下角显示重力方向指示器
                    arrow_length = 50  # 箭头长度
                    image_height, image_width = display_image.shape[:2]
                    center_x, center_y = image_width - 60, image_height - 40  # 右下角位置
                    # 反向显示更直观
                    end_x = int(
                        center_x - global_gravity_dir[0] * arrow_length)
                    end_y = int(
                        center_y - global_gravity_dir[1] * arrow_length)

                    # 绘制重力方向指示器
                    cv2.circle(
                        display_image, (center_x,
                                        center_y), 30, (0, 255, 255), 1
                    )  # 圆
                    cv2.line(
                        display_image,
                        (center_x, center_y),
                        (end_x, end_y),
                        (0, 255, 255),
                        2,
                    )  # 线
                    cv2.putText(
                        display_image,
                        "G",
                        (end_x - 5, end_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )  # 标签

            # 清除深度图显示中可能重叠的文本 - 创建一个干净的图像用于重新叠加文本信息
            if depth_filtered_mask is not None and filtered_depth is not None:
                # 获取深度图数据
                depth_data = np.asanyarray(filtered_depth.get_data())

                # 将深度图转换为可视化的彩色图
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(
                        depth_data, alpha=0.04), cv2.COLORMAP_JET
                )

                # 应用深度过滤掩码到深度图 - 这是一个干净的无文本版本
                depth_mask_bool = depth_filtered_mask > 0
                masked_depth_colormap = np.zeros_like(depth_colormap)
                masked_depth_colormap[depth_mask_bool] = depth_colormap[depth_mask_bool]

                # 为非掩码区域应用半透明效果，保留原始深度图中的信息
                non_mask_region = ~depth_mask_bool
                if np.any(non_mask_region):
                    masked_depth_colormap[non_mask_region] = (
                        depth_colormap[non_mask_region] * 0.4).astype(np.uint8)

                # 更新深度图显示
                depth_colormap_image = masked_depth_colormap

                # 确保深度窗口显示更新后的图像
                cv2.imshow(WINDOW_DEPTH, depth_colormap_image)

        # 显示信息
        cv2.putText(
            display_image,
            f"FPS: {avg_fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # 显示操作提示 - 放在FPS下方
        if waiting_for_clicks:
            cv2.putText(
                display_image,
                "Click mode: Left=positive, Right=negative, Enter=process",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        elif segmentation_in_progress:
            cv2.putText(
                display_image,
                f"Processing segmentation... Please wait ({int(time.time() - segmentation_start_time)}s)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                display_image,
                "Press 'c'=click mode 'r'=reset 'e'=export 'd'=measure 'm'=mode",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # 只在没有掩码时显示自动检测模式状态 - 放在屏幕右上角
        if current_mask is None:
            auto_mode_color = (0, 255, 0) if auto_detect_mode else (0, 0, 255)
            cv2.putText(
                display_image,
                f"Auto Detect: {'ON' if auto_detect_mode else 'OFF'}",
                (display_image.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                auto_mode_color,
                2,
            )

        # 显示点云状态 - 放在底部
        if show_pointcloud:
            cv2.putText(
                display_image,
                "Point Cloud: Enabled",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            # 显示当前导出模式 - 放在点云状态的右侧
            mode_text = (
                "Export Mode: Laser Only"
                if export_laser_only
                else "Export Mode: Full Scene"
            )
            cv2.putText(
                display_image,
                mode_text,
                (320, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # # 跟踪状态放在屏幕右侧
        # if tracking:
        #     cv2.putText(
        #         display_image,
        #         "Tracking: Active",
        #         (display_image.shape[1] - 320, 120),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.6,
        #         (255, 0, 0),
        #         2,
        #     )

        # 显示深度图
        depth_colormap_image = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_JET
        )

        # 如果有分割掩码，将掩码应用到深度图
        masked_depth_image = depth_colormap_image.copy()
        if current_mask is not None:
            try:
                # 增亮所有分割区域在深度图上的显示
                int_mask = current_mask.astype(np.uint8)

                # 为掩码区域创建高亮效果 - 使用白色边框勾勒
                kernel = np.ones((3, 3), np.uint8)
                mask_border = cv2.dilate(
                    int_mask, kernel, iterations=1) - int_mask

                # 在边缘处添加白色边框
                masked_depth_image[mask_border == 1] = [255, 255, 255]

                # 对掩码内区域增强对比度
                for c in range(3):
                    masked_depth_image[:, :, c] = np.where(
                        int_mask > 0,
                        np.clip(
                            masked_depth_image[:, :, c].astype(
                                np.int32) * 1.3, 0, 255
                        ).astype(np.uint8),
                        masked_depth_image[:, :, c],
                    )

                # 显示分割区域的标签
                cv2.putText(
                    masked_depth_image,
                    "Segmented Region",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                # 如果深度过滤已完成，也相应更新深度图
                if depth_filter_completed and depth_filtered_mask is not None:
                    # 创建深度图副本
                    depth_filtered_depth = masked_depth_image.copy()

                    # 获取深度过滤掩码（布尔数组）
                    final_mask_bool = depth_filtered_mask > 0

                    # 将非深度过滤掩膜区域变暗
                    # 原代码：depth_filtered_depth[~final_mask_bool] = (depth_filtered_depth[~final_mask_bool] // 4)
                    # 修改为半透明处理，保留50%的原始亮度
                    depth_filtered_depth[~final_mask_bool] = (
                        depth_filtered_depth[~final_mask_bool] * 0.5).astype(np.uint8)  # 半透明背景

                    # 为掩膜区域添加亮色边框
                    # 找到掩膜边缘
                    contours, _ = cv2.findContours(
                        depth_filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(depth_filtered_depth,
                                     contours, -1, (0, 255, 255), 2)

                    # 更新深度图显示
                    masked_depth_image = depth_filtered_depth

                    # 标记已过滤
                    cv2.putText(
                        masked_depth_image,
                        "Z-Filtered Depth",
                        (width - 220, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
            except Exception as e:
                print(f"应用掩码到深度图失败: {e}")

        # 显示图像前检查是否有测量结果需要展示
        current_time = time.time()
        if 'last_measurement_result' in globals() and last_measurement_result is not None:
            if current_time - last_measurement_time < MEASUREMENT_DISPLAY_DURATION:
                # 显示测量结果
                cv2.putText(
                    display_image,
                    f"Measurement: {last_measurement_result:.2f} mm (sigma={last_measurement_std:.2f})",
                    (width - 450, height - 60),  # 放在右下角
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),  # 黄色
                    2,
                )
                # 展示测量结果稳定性
                stability_desc = "High" if last_measurement_std < 0.5 else (
                    "Medium" if last_measurement_std < 1.5 else "Low")
                cv2.putText(
                    display_image,
                    f"Stability: {stability_desc} ({last_measurement_count} valid measurements)",
                    (width - 450, height - 30),  # 放在右下角
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),  # 黄色
                    2,
                )

        # 显示图像
        cv2.imshow(WINDOW_RGB, display_image)
        cv2.imshow(WINDOW_DEPTH, masked_depth_image)

        # 处理点云窗口
        if show_pointcloud:
            if not pointcloud_window_created:
                # 只有在窗口未创建时才创建窗口
                cv2.namedWindow(WINDOW_POINTCLOUD)
                pointcloud_window_created = True

            # 显示点云视图
            if pointcloud_data["render_image"] is not None:
                # 检查点云窗口是否长时间未更新（超时检测）
                current_time = time.time()
                if "last_update_time" in pointcloud_data:
                    time_since_update = current_time - \
                        pointcloud_data.get("last_update_time", current_time)
                    # 如果超过5秒没有更新，显示警告
                    if time_since_update > 5.0:
                        # 如果render_image为None，创建一个空白图像
                        if pointcloud_data["render_image"] is None:
                            pointcloud_data["render_image"] = np.zeros(
                                (480, 640, 3), dtype=np.uint8)

                        # 在现有图像上添加警告信息
                        warning_image = pointcloud_data["render_image"].copy()
                        # 添加半透明黑色背景
                        overlay = warning_image.copy()
                        cv2.rectangle(overlay, (50, 180),
                                      (590, 300), (0, 0, 0), -1)
                        # 应用半透明效果
                        cv2.addWeighted(
                            overlay, 0.5, warning_image, 0.5, 0, warning_image)

                        # 添加警告文本
                        cv2.putText(
                            warning_image,
                            "警告: 点云生成可能遇到问题",
                            (100, 220),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),  # 红色
                            2,
                        )
                        cv2.putText(
                            warning_image,
                            "尝试按'R'重置或'M'切换模式",
                            (100, 260),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),  # 黄色
                            2,
                        )

                        # 强制更新UI
                        pointcloud_data["render_image"] = warning_image
                        cv2.imshow(WINDOW_POINTCLOUD, warning_image)

                        # 尝试恢复 - 设置updated标志为True，强制渲染线程尝试更新
                        pointcloud_data["updated"] = True
                        print(f"点云窗口超过{time_since_update:.1f}秒未更新，尝试恢复")

                # 使用Open3D渲染的图像
                cv2.imshow(WINDOW_POINTCLOUD, pointcloud_data["render_image"])
            else:
                # 如果还没有点云数据，显示提示信息
                blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank_image,
                    "Generating point cloud...",
                    (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(WINDOW_POINTCLOUD, blank_image)
        elif pointcloud_window_created:
            # 只有在窗口已创建且需要关闭时才销毁窗口
            cv2.destroyWindow(WINDOW_POINTCLOUD)
            pointcloud_window_created = False
            print("点云窗口已关闭")

        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键退出
            running = False
        elif key == ord("c") and not segmentation_in_progress:  # 'c'键进入点击模式
            waiting_for_clicks = True
            tracking = False  # 停止跟踪
            prompt_points = []  # 清空点
            prompt_labels = []
            print("进入点击模式，请使用鼠标添加提示点")
        elif (
            key == 13
            and waiting_for_clicks
            and len(prompt_points) > 0
            and not segmentation_in_progress
        ):  # Enter键执行分割
            do_segmentation = True
            print("将在下一帧执行分割处理")
        elif key == ord("r") and not segmentation_in_progress:  # 'r'键重置
            prompt_points = []
            prompt_labels = []
            waiting_for_clicks = False
            tracking = False  # 停止跟踪
            current_mask = None
            obj_id = 1  # 重置对象ID
            is_first_frame = True  # 重置第一帧标志
            # 清除距离计算结果
            calculated_distance = None
            all_endpoints = None
            middle_endpoints = None
            # 如果点云窗口已开启，关闭它
            if show_pointcloud:
                show_pointcloud = False
                if pointcloud_window_created:
                    cv2.destroyWindow(WINDOW_POINTCLOUD)
                    pointcloud_window_created = False
            print("重置所有状态")
        elif key == ord("p"):  # 'p'键切换点云显示
            show_pointcloud = not show_pointcloud
            if not show_pointcloud and pointcloud_window_created:
                # 如果关闭点云显示且窗口已创建，销毁窗口
                cv2.destroyWindow(WINDOW_POINTCLOUD)
                pointcloud_window_created = False
            print(f"点云显示: {'已开启' if show_pointcloud else '已关闭'}")
        # 添加导出模式切换按键
        elif key == ord("m"):  # 'm'键切换导出模式
            export_laser_only = not export_laser_only
            print(f"导出模式: {'仅激光区域' if export_laser_only else '全场景'}")

            # 切换模式后立即更新点云显示
            if show_pointcloud:
                # 确保我们有两种模式的点云数据
                if export_laser_only and pointcloud_data["points"] is not None:
                    # 已经有激光区域数据，只需触发重新渲染
                    pointcloud_data["updated"] = True
                elif not export_laser_only:
                    if pointcloud_data["full_points"] is None:
                        # 如果还没有全场景点云数据，立即生成
                        try:
                            full_points, full_colors, _ = create_point_cloud(
                                depth_image,
                                rgb_image,
                                depth_scale,
                                depth_intrin,
                                mask=None,
                                laser_only=False,
                            )

                            pointcloud_data["full_points"] = full_points
                            pointcloud_data["full_colors"] = full_colors
                            pointcloud_data["updated"] = True
                        except Exception as e:
                            print(f"生成全场景点云失败: {e}")
                    else:
                        # 已经有全场景数据，只需触发重新渲染
                        pointcloud_data["updated"] = True
        # 添加视角控制
        elif key == ord("1"):  # 前视图
            render_options["front"] = [0, 0, -1]
            render_options["up"] = [0, -1, 0]
            print("点云视角: 前视图")
        elif key == ord("2"):  # 侧视图
            render_options["front"] = [1, 0, 0]
            render_options["up"] = [0, -1, 0]
            print("点云视角: 侧视图")
        elif key == ord("3"):  # 俯视图
            render_options["front"] = [0, -1, 0]
            render_options["up"] = [0, 0, -1]
            print("点云视角: 俯视图")
        elif key == ord("+") or key == ord("="):  # 放大
            render_options["zoom"] *= 0.7  # 减小zoom值以放大点云
            print(f"点云缩放: {render_options['zoom']:.2f}")
        elif key == ord("-") or key == ord("_"):  # 缩小
            render_options["zoom"] *= 1.3  # 增加zoom值以缩小点云
            print(f"点云缩放: {render_options['zoom']:.2f}")
        elif key == ord("e") and show_pointcloud:  # 'e'键导出点云
            if (
                pointcloud_data["points"] is not None
                and len(pointcloud_data["points"]) > 0
            ):
                # 根据导出模式生成不同的文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                if export_laser_only:
                    # 导出激光区域点云
                    export_filename = os.path.join(
                        export_dir, f"laser_pointcloud_{timestamp}.ply"
                    )

                    # 检查是否已完成深度过滤
                    if depth_filter_completed and depth_filtered_mask is not None:
                        # 如果已完成深度过滤，使用深度过滤后的点
                        print("使用深度过滤后的点云数据进行导出")
                        # 重新筛选点云数据，确保只包含深度过滤区域内的点
                        filtered_point_indices = []
                        points_to_filter = pointcloud_data["points"]
                        colors_to_filter = pointcloud_data["colors"]

                        # 将每个点投影回2D空间，检查是否在深度过滤掩码区域内
                        for i, point in enumerate(points_to_filter):
                            # 将毫米单位转回米
                            point_meters = point / 1000.0
                            # 投影到像素坐标
                            try:
                                pixel = rs.rs2_project_point_to_pixel(
                                    depth_intrin, point_meters
                                )
                                x, y = int(pixel[0]), int(pixel[1])

                                # 检查像素是否在图像范围内且在深度过滤掩码内
                                if (
                                    0 <= x < depth_filtered_mask.shape[1]
                                    and 0 <= y < depth_filtered_mask.shape[0]
                                    and depth_filtered_mask[y, x] > 0
                                ):
                                    filtered_point_indices.append(i)
                            except Exception as e:
                                continue

                        if filtered_point_indices:
                            # 使用深度过滤后的点
                            export_points = points_to_filter[filtered_point_indices]
                            export_colors = colors_to_filter[filtered_point_indices]
                            print(f"导出包含 {len(export_points)} 个深度过滤区域内的点")
                        else:
                            # 如果没有找到符合条件的点，使用原始点云
                            export_points = pointcloud_data["points"]
                            export_colors = pointcloud_data["colors"]
                            print("未找到深度过滤区域内的点，使用原始点云数据")
                    else:
                        # 如果未完成深度过滤，使用普通点云数据
                        export_points = pointcloud_data["points"]
                        export_colors = pointcloud_data["colors"]

                    file_type = "激光区域"
                else:
                    # 导出全场景点云
                    export_filename = os.path.join(
                        export_dir, f"full_scene_pointcloud_{timestamp}.ply"
                    )

                    # 生成全场景点云
                    full_points, full_colors, _ = create_point_cloud(
                        depth_image,
                        rgb_image,
                        depth_scale,
                        depth_intrin,
                        mask=None,  # 不使用掩码
                        laser_only=False,  # 不限制为激光区域
                    )

                    export_points = full_points
                    export_colors = full_colors
                    file_type = "全场景"

                # 导出点云
                if export_point_cloud(export_points, export_colors, export_filename):
                    # 显示相对于Desktop的路径，更直观
                    rel_path = os.path.relpath(
                        export_filename, os.path.dirname(script_dir)
                    )

                    # 在RGB图像上显示导出成功消息
                    export_msg = (
                        f"Exported {file_type}: {os.path.basename(export_filename)}"
                    )
                    cv2.putText(
                        display_image,
                        export_msg,
                        (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    print(f"{file_type}点云导出成功: {export_filename}")

                    # 添加自动测距功能
                    if HAS_CLOUDCOMPY:
                        print("正在执行自动交点检测和测距...")
                        intersections, distances = calculate_laser_lines_distance_exclude_bends(
                            export_filename
                        )

                        if distances and len(distances) > 0:
                            # 显示测量结果
                            distance_mm = distances[0]
                            cv2.putText(
                                display_image,
                                f"Auto Measurement: {distance_mm:.2f} mm",
                                (10, 450),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )
                            print(f"自动测量距离: {distance_mm:.2f} mm")

                    # 自动用CloudCompare打开文件
                    try:
                        # 尝试使用CloudCompare打开文件
                        cloudcompare_path = "CloudCompare"  # 如果CloudCompare在PATH中
                        # 设置LANG环境变量以避免语言文件警告
                        my_env = os.environ.copy()
                        my_env["LANG"] = "en_US.UTF-8"

                        subprocess.Popen(
                            [cloudcompare_path, export_filename], env=my_env
                        )
                        print(f"已使用CloudCompare打开文件: {export_filename}")
                    except Exception as e:
                        print(f"启动CloudCompare失败: {e}")
            else:
                print("没有点云数据可导出")
        # 添加测距快捷键
        elif key == ord("d") and show_pointcloud:  # 'd'键直接进行测距
            try:
                if (
                    pointcloud_data["points"] is not None
                    and len(pointcloud_data["points"]) > 0
                    and HAS_CLOUDCOMPY
                ):
                    print("正在执行增强型自动测距（多次测量取平均值）...")

                    # 临时保存点云到文件
                    temp_filename = os.path.join(
                        export_dir, f"temp_for_measure_{time.strftime('%Y%m%d_%H%M%S')}.ply"
                    )

                    if export_laser_only:
                        # 检查是否已完成深度过滤
                        if depth_filter_completed and depth_filtered_mask is not None:
                            # 如果已完成深度过滤，使用深度过滤后的点
                            print("使用深度过滤后的点云数据进行测距")
                            # 重新筛选点云数据，确保只包含深度过滤区域内的点
                            filtered_point_indices = []
                            points_to_filter = pointcloud_data["points"]
                            colors_to_filter = pointcloud_data["colors"]

                            # 将每个点投影回2D空间，检查是否在深度过滤掩码区域内
                            for i, point in enumerate(points_to_filter):
                                # 将毫米单位转回米
                                point_meters = point / 1000.0
                                # 投影到像素坐标
                                try:
                                    pixel = rs.rs2_project_point_to_pixel(
                                        depth_intrin, point_meters
                                    )
                                    x, y = int(pixel[0]), int(pixel[1])

                                    # 检查像素是否在图像范围内且在深度过滤掩码内
                                    if (
                                        0 <= x < depth_filtered_mask.shape[1]
                                        and 0 <= y < depth_filtered_mask.shape[0]
                                        and depth_filtered_mask[y, x] > 0
                                    ):
                                        filtered_point_indices.append(i)
                                except Exception as e:
                                    continue

                            if filtered_point_indices:
                                # 使用深度过滤后的点
                                export_points = points_to_filter[filtered_point_indices]
                                export_colors = colors_to_filter[filtered_point_indices]
                                print(f"测距使用 {len(export_points)} 个深度过滤区域内的点")
                            else:
                                # 如果没有找到符合条件的点，使用原始点云
                                export_points = pointcloud_data["points"]
                                export_colors = pointcloud_data["colors"]
                                print("未找到深度过滤区域内的点，使用原始点云数据进行测距")
                        else:
                            # 如果未完成深度过滤，使用普通点云数据
                            export_points = pointcloud_data["points"]
                            export_colors = pointcloud_data["colors"]
                    else:
                        export_points = pointcloud_data["full_points"]
                        export_colors = pointcloud_data["full_colors"]

                    # 先确保点云数据有效
                    if export_points is None or len(export_points) < 10:
                        print("警告: 点云数据不足，无法进行测距")
                        continue

                    # 导出点云数据
                    if export_point_cloud(export_points, export_colors, temp_filename):
                        try:
                            # 执行多次测量和统计处理
                            # 传递当前帧的点云数据用于实时测量
                            final_distance, std_dev, all_measures = calculate_multiple_measures(
                                temp_filename,
                                num_measures=5,
                                real_time=True,  # 始终保持实时模式为True，确保每次测量都获取新帧
                                time_between=0.5,  # 每次测量之间等待0.5秒
                                depth_scale=depth_scale,
                                depth_intrin=depth_intrin
                            )

                            if final_distance is not None:
                                # 保存测量结果到全局变量 - 这里不需要使用global语句，因为我们只是读取变量
                                # 修改全局变量
                                last_measurement_result = final_distance
                                last_measurement_std = std_dev
                                last_measurement_count = len(all_measures)
                                last_measurement_time = time.time()

                                # 显示测量结果
                                cv2.putText(
                                    display_image,
                                    f"Multiple Measurements: {final_distance:.2f} mm (sigma={std_dev:.2f})",
                                    (10, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2,
                                )
                                # 展示测量结果稳定性
                                stability_desc = "High" if std_dev < 0.5 else (
                                    "Medium" if std_dev < 1.5 else "Low")
                                cv2.putText(
                                    display_image,
                                    f"Stability: {stability_desc} ({len(all_measures)} valid measurements)",
                                    (10, 480),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2,
                                )
                                print(
                                    f"多次测量最终结果: {final_distance:.2f} mm, 标准差: {std_dev:.2f}")

                                # 立即显示图像，确保结果及时更新到窗口
                                cv2.imshow(WINDOW_RGB, display_image)
                                cv2.waitKey(1)  # 立即处理窗口事件

                                # 可选：绘制测量值分布直方图
                                try:
                                    # 创建测量结果分布图并保存
                                    if len(all_measures) >= 5:
                                        histogram_file = os.path.join(
                                            export_dir, f"measurement_histogram_{time.strftime('%Y%m%d_%H%M%S')}.png"
                                        )
                                        # 抑制matplotlib中文字体警告
                                        with warnings.catch_warnings():
                                            warnings.filterwarnings(
                                                "ignore", category=UserWarning, message=".*CJK.*")
                                            warnings.filterwarnings(
                                                "ignore", category=UserWarning, message=".*Glyph.*missing from font.*")

                                            plt.figure(figsize=(8, 4))
                                            plt.hist(all_measures, bins=5,
                                                     alpha=0.7, color='blue')
                                            plt.axvline(final_distance, color='red', linestyle='dashed',
                                                        linewidth=2, label=f'最终值: {final_distance:.2f}mm')
                                            plt.xlabel('距离 (mm)')
                                            plt.ylabel('频次')
                                            plt.title(
                                                f'测量值分布 (标准差: {std_dev:.2f}mm)')
                                            plt.legend()
                                            plt.tight_layout()
                                            plt.savefig(histogram_file)
                                            plt.close()
                                        print(f"测量分布直方图已保存到: {histogram_file}")
                                except Exception as e:
                                    print(f"生成测量分布图失败: {e}")
                            else:
                                print("测量失败，未能获得有效结果")
                        except Exception as e:
                            print(f"执行测量过程中出错: {e}")
                            import traceback
                            traceback.print_exc()
                        finally:
                            # 无论成功与否，都尝试删除临时文件
                            try:
                                if os.path.exists(temp_filename):
                                    os.remove(temp_filename)
                            except Exception:
                                pass

                            # 确保相机资源得到释放
                            try:
                                # 清空相机缓冲区
                                if 'pipeline' in globals() and pipeline:
                                    for _ in range(5):  # 清空几帧缓冲
                                        try:
                                            frames = pipeline.wait_for_frames(
                                                timeout_ms=100)
                                        except Exception:
                                            break
                                print("相机资源已重置，可以继续使用'd'键进行测量")
                            except Exception as e:
                                print(f"重置相机状态时出错: {e}")
                else:
                    print("没有点云数据可测距或CloudComPy未安装")
            except Exception as e:
                print(f"测量过程中发生未知错误: {e}")
                import traceback
                traceback.print_exc()
        # 在主循环中，处理按键部分添加新的case
        # 在处理按键处添加以下代码
        elif key == ord("a"):  # 'a'键切换自动检测模式
            auto_detect_mode = not auto_detect_mode
            print(f"自动检测模式: {'已开启' if auto_detect_mode else '已关闭'}")
            if auto_detect_mode:
                # 重置一些状态
                prompt_points = []
                prompt_labels = []
                waiting_for_clicks = False
                last_auto_detect_time = time.time()
                auto_detect_counter = 0

        # 主循环中，执行自动检测
        # 如果自动检测模式开启
        if auto_detect_mode and not segmentation_in_progress and not tracking:
            current_time = time.time()

            if (
                auto_detect_counter >= auto_detect_interval
                or (current_time - last_auto_detect_time) > 2.0
            ):
                # 只在状态变化或间隔较长时才输出检测信息
                should_print = (
                    current_time - last_status_print_time
                ) > 5.0  # 至少5秒输出一次状态

                if should_print:
                    print("执行自动绿色激光检测...")
                    last_status_print_time = current_time

                auto_points, auto_labels, laser_mask, detection_success = (
                    detect_green_laser_for_sam(rgb_image)
                )

                # 检测状态改变时输出信息
                if detection_success != last_detect_status or should_print:
                    if detection_success:
                        print("自动检测到激光，将执行分割")
                    else:
                        print("未检测到有效的绿色激光，继续等待...")

                    last_detect_status = detection_success

                if detection_success:
                    # 更新提示点
                    prompt_points = auto_points
                    prompt_labels = auto_labels

                    # 绘制检测到的提示点
                    display_image = draw_points(
                        display_image, prompt_points, prompt_labels
                    )

                    # 对激光区域绘制框选效果 - 使用绿色细线
                    if laser_mask is not None:
                        # 找到激光掩码的轮廓
                        contours, _ = cv2.findContours(
                            laser_mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        # 绘制绿色细线边框
                        cv2.drawContours(
                            display_image, contours, -1, (0, 255, 0), 1
                        )  # 绿色(BGR)，线宽1

                    # 显示当前帧以便用户看到提示点
                    cv2.imshow(WINDOW_RGB, display_image)
                    cv2.waitKey(500)  # 延时500毫秒，让用户看清提示点

                    # 立即执行分割
                    do_segmentation = True
                else:
                    # 清空之前的提示点，不进行分割
                    prompt_points = []
                    prompt_labels = []
                    do_segmentation = False

                    # 在图像上显示等待信息
                    waiting_msg = "等待激光检测中..."
                    cv2.putText(
                        display_image,
                        waiting_msg,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow(WINDOW_RGB, display_image)

                    # 重置检测计数器，但不要完全重置时间，以便更快地进行下一次检测
                    # 只减少间隔时间而不是完全重置
                    last_auto_detect_time = (
                        current_time - 1.5
                    )  # 减少等待时间，0.5秒后再次检测

                auto_detect_counter = 0
                # 只有在成功检测到激光时才完全重置时间
                if detection_success:
                    last_auto_detect_time = current_time

        frame_count += 1

        # 在主循环中获取并处理IMU数据
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # 获取各部分图像数据
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 获取IMU数据
        accel_frame = None
        gyro_frame = None
        try:
            # 尝试获取最新的IMU帧
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)
        except Exception as e:
            if frame_count % 300 == 0:  # 避免频繁打印错误
                print(f"获取IMU数据失败: {e}")
            accel_frame = None
            gyro_frame = None

        # 提取加速度计和陀螺仪数据
        accel_sample = None
        gyro_sample = None
        if accel_frame is not None:
            try:
                accel_sample = accel_frame.as_motion_frame().get_motion_data()
            except Exception as e:
                if frame_count % 300 == 0:
                    print(f"处理加速度计数据失败: {e}")
                accel_sample = None

        if gyro_frame is not None:
            try:
                gyro_sample = gyro_frame.as_motion_frame().get_motion_data()
            except Exception as e:
                if frame_count % 300 == 0:
                    print(f"处理陀螺仪数据失败: {e}")
                gyro_sample = None

        # # 调试输出IMU数据和相机姿态
        # if accel_sample is not None:  # 移除frame_count % 300条件，每帧都处理
        #     # 只在每300帧时打印调试信息
        #     if frame_count % 300 == 0:
        #         # 输出加速度计和陀螺仪数据
        #         print(
        #             f"加速度计: x={accel_sample.x:.2f}, y={accel_sample.y:.2f}, z={accel_sample.z:.2f}"
        #         )
        #         if gyro_sample is not None:
        #             print(
        #                 f"陀螺仪: x={gyro_sample.x:.2f}, y={gyro_sample.y:.2f}, z={gyro_sample.z:.2f}"
        #             )

            # 计算并显示相机姿态 - 每帧都更新
            try:
                roll, pitch = get_camera_orientation_from_imu(accel_sample)
                # 更新全局变量
                global_roll = roll
                global_pitch = pitch
                global_gravity_dir = - \
                    np.array([accel_sample.x, accel_sample.y, accel_sample.z])
                global_gravity_dir = global_gravity_dir / \
                    np.linalg.norm(global_gravity_dir)

                # 只在每300帧时打印调试信息
                if frame_count % 300 == 0:
                    print(
                        f"相机姿态估计: Roll={np.degrees(roll):.2f}°, Pitch={np.degrees(pitch):.2f}°"
                    )

                # 每帧都更新显示
                cv2.putText(
                    display_image,
                    f"Roll: {np.degrees(roll):.1f}",
                    (10, height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_image,
                    f"Pitch: {np.degrees(pitch):.1f}",
                    (10, height - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            except Exception as e:
                if frame_count % 300 == 0:  # 只在每300帧时打印错误信息
                    print(f"计算相机姿态失败: {e}")

        # 使用并行处理优化深度滤波链
        filtered_depth = process_depth_frame_parallel(depth_frame)
except KeyboardInterrupt:
    print("用户手动中断程序")
except Exception as e:
    print(f"发生错误: {e}")
    import traceback

    traceback.print_exc()
finally:
    print("正在关闭并释放设备资源...")

    # 关闭多核处理资源
    try:
        if "thread_pool" in globals() and thread_pool is not None:
            thread_pool.shutdown(wait=False)
            print("线程池已关闭")
    except Exception as e:
        print(f"关闭线程池时出错: {e}")

    try:
        if "process_pool" in globals() and process_pool is not None:
            process_pool.terminate()
            process_pool.join(timeout=1.0)
            print("进程池已关闭")
    except Exception as e:
        print(f"关闭进程池时出错: {e}")

    # 停止点云线程
    if "use_open3d_visualization" in locals() and use_open3d_visualization:
        pointcloud_thread_running = False
    try:
        if "pointcloud_thread" in locals() and pointcloud_thread is not None:
            pointcloud_thread.join(timeout=1.0)
            print("点云线程已停止")
    except:
        pass

    # 确保关闭所有窗口
    try:
        cv2.destroyAllWindows()
        print("所有窗口已销毁")
    except Exception as e:
        print(f"销毁窗口时出错: {e}")

    # 确保停止pipeline
    try:
        if "pipeline" in locals():
            pipeline.stop()
            print("Pipeline已停止")
    except Exception as e:
        print(f"停止pipeline时出错: {e}")

    # 尝试手动释放设备
    try:
        if "pipeline" in locals():
            del pipeline
            gc.collect()  # 强制垃圾回收
            print("已尝试手动释放设备资源")
    except:
        pass

    print("程序已完全退出")


def merge_multiple_frames(num_frames=3):  # 减少合并帧数从5帧降为3帧
    all_points = []
    all_colors = []

    # 设置采样间隔，减少CPU消耗
    sample_interval = 0.05  # 从0.1减少到0.05秒

    for i in range(num_frames):
        # 获取一帧数据并提取点云 - 使用缓存功能
        points, colors, _ = create_point_cloud(
            depth_image,
            rgb_image,
            depth_scale,
            depth_intrin,
            mask=current_mask,  # 使用当前激光掩码
            laser_only=True,  # 只显示激光区域
        )

        # 如果点数过多，进行随机下采样
        if len(points) > 5000:
            # 随机选择5000个点
            indices = np.random.choice(len(points), 5000, replace=False)
            points = points[indices]
            colors = colors[indices]

        all_points.append(points)
        all_colors.append(colors)
        time.sleep(sample_interval)  # 等待下一帧

    # 合并所有点云
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    # 如果点数过多，进行体素下采样
    if len(merged_points) > 10000:
        # 随机下采样到10000个点
        indices = np.random.choice(len(merged_points), 10000, replace=False)
        merged_points = merged_points[indices]
        merged_colors = merged_colors[indices]

    return merged_points, merged_colors


# 清理资源，关闭实时点云线程
pointcloud_thread_running = False
if (
    "pointcloud_thread" in globals()
    and pointcloud_thread is not None
    and pointcloud_thread.is_alive()
):
    pointcloud_thread.join()

# 清理句柄
if "pipeline" in globals():
    pipeline.stop()

cv2.destroyAllWindows()

torch.cuda.empty_cache()
gc.collect()

# 删除临时文件
if "temp_frame_dir" in globals():
    for f in temp_frame_dir.glob("*.jpg"):
        f.unlink()

sys.exit(0)

# 简单的测试，确保代码通过语法检查
if __name__ == "__test__":
    import ast

    # 检查源代码是否有语法错误
    with open(__file__, 'r') as f:
        source = f.read()

    try:
        ast.parse(source)
        print("代码通过语法检查")
    except SyntaxError as e:
        print(f"语法错误在第{e.lineno}行: {e.text}")
        print(f"错误信息: {e.msg}")

# 添加一个函数用于将3D点云点映射回RGB图像


def project_point_to_image(point_3d, color_intrinsics):
    """
    将3D点投影回2D图像坐标

    参数:
        point_3d: 3D点坐标(mm)
        color_intrinsics: 相机内参

    返回:
        (u, v): 2D图像坐标
    """
    import pyrealsense2 as rs

    # 将点从毫米转为米
    point_m = np.array(point_3d) / 1000.0

    # 使用RS2库进行投影
    pixel = rs.rs2_project_point_to_pixel(color_intrinsics, point_m)

    # 转为整数坐标
    u, v = int(round(pixel[0])), int(round(pixel[1]))

    return u, v
