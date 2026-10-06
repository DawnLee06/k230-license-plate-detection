# -*- coding: utf-8 -*-
'''
Script: target_tracking.py
脚本名称：靶心追踪系统（精准适配 DetectionApp）

Description:
    实时检测靶心（label="target"），并追踪最大靶心
    - 使用 K230 硬件 PWM 直接驱动舵机
    - 水平舵机：360°连续旋转（速度控制）
    - 垂直舵机：标准180°舵机（角度控制）
'''

# 导入必要的库
import os, gc
from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine, ScopedTiming
from libs.Utils import *
from machine import FPIOA, PWM
import ulab.numpy as np
import time

display_mode = "lcd"

# 定义RGB888P视频帧的输入尺寸
rgb888p_size = [1280, 720]

# 设置模型和配置文件的根目录路径
root_path = "/sdcard/mp_deployment_source/"

# 从JSON配置文件读取部署配置
deploy_conf = read_json(root_path + "/deploy_config.json")
kmodel_path = root_path + deploy_conf["kmodel_path"]              # KModel文件路径
labels = deploy_conf["categories"]                                # 类别标签列表
confidence_threshold = deploy_conf["confidence_threshold"]        # 置信度阈值
nms_threshold = deploy_conf["nms_threshold"]                      # NMS(非极大值抑制)阈值
model_input_size = deploy_conf["img_size"]                        # 模型输入尺寸
nms_option = deploy_conf["nms_option"]                            # NMS策略选项
model_type = deploy_conf["model_type"]                            # 检测模型类型
anchors = []  # 初始化anchors列表
# 如果模型类型是AnchorBaseDet(基于锚点的目标检测)，则合并所有anchors
if model_type == "AnchorBaseDet":
    anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]

# 推理配置
inference_mode = "video"                                          # 推理模式：'video'(视频模式)
debug_mode = 0                                                    # 调试模式标志(0:关闭, 1:开启)

# 创建并初始化视频/显示管道
pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
pl.create()  # 初始化管道
display_size = pl.get_display_size()  # 获取显示尺寸

# ================== 3. 舵机 PWM 初始化 ==================
fpioa = FPIOA()
fpioa.set_function(42, FPIOA.PWM0)  # GPIO42 -> PWM0（水平舵机）
fpioa.set_function(43, FPIOA.PWM1)  # GPIO43 -> PWM1（垂直舵机）

# 创建PWM实例
pwm_x = PWM(0, 50, enable=True)  # 水平舵机：360°连续旋转，50Hz
pwm_y = PWM(1, 50, enable=True)  # 垂直舵机：标准180°舵机，50Hz

# --- PWM 转换函数 ---
def speed_to_duty(speed_percent):
    """360°舵机：速度 → 占空比"""
    pulse_ms = 1.5 + (speed_percent / 100.0) * 0.2  # ±0.2ms
    return (pulse_ms / 20.0) * 100

def angle_to_duty(angle):
    """标准舵机：角度 → 占空比"""
    pulse_ms = 0.5 + (angle / 180.0) * 2.0  # 0.5~2.5ms
    return (pulse_ms / 20.0) * 100

# --- 垂直舵机初始位置 ---
y_angle = 90  # 初始居中
pwm_y.duty(angle_to_duty(y_angle))

# ================== 4. PID 控制器 ==================
class PID:
    def __init__(self, p=0.05, i=0.0, d=0.01):
        self.kp, self.ki, self.kd = p, i, d
        self.target = 0
        self.error = 0
        self.last_error = 0
        self.integral = 0

    def update(self, current):
        self.error = self.target - current
        if abs(self.error) < 15:  # 死区（像素）
            self.integral = 0
            return 0
        self.integral += self.error
        derivative = self.error - self.last_error
        output = self.kp * self.error + self.ki * self.integral + self.kd * derivative
        self.last_error = self.error
        return output

    def set_target(self, target):
        self.target = target
        self.integral = 0
        self.last_error = 0

# 初始化PID控制器
x_pid = PID(p=0.5, i=0.0, d=0.05)  # 水平：速度控制（需调试）
y_pid = PID(p=0.01, i=0.0, d=0.001) # 垂直：角度控制

# ================== 5. 初始化视频管道和检测模型 ==================
pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
pl.create()
display_size = pl.get_display_size()

# 初始化目标检测应用实例
det_app = DetectionApp(
    inference_mode,           # 推理模式
    kmodel_path,              # KModel路径
    labels,                   # 标签列表
    model_input_size,         # 模型输入尺寸
    anchors,                  # 锚点配置
    model_type,               # 模型类型
    confidence_threshold,     # 置信度阈值
    nms_threshold,            # NMS阈值
    rgb888p_size,             # RGB888P输入尺寸
    display_size,             # 显示尺寸
    debug_mode=debug_mode     # 调试模式
)
det_app.config_preprocess()

# 设置 PID 目标为中心
x_pid.set_target(display_size[0] / 2)
y_pid.set_target(display_size[1] / 2)

clock = time.clock()

# ================== 6. 主循环：检测 + 追踪 ==================
while True:
    clock.tick()
    img = pl.get_frame()
    det_result = det_app.run(img)  # 返回: {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "idx": [...]}

    # --- 清除上一帧 OSD ---
    pl.osd_img.clear()

    # --- 查找靶心目标（label == TARGET_LABEL）---
    target_center = None
    max_area = 0

    if det_result["boxes"]:
        for i in range(len(det_result["boxes"])):
            class_id = det_result["idx"][i]
            label_name = det_app.labels[class_id]

            # 🎯 检查是否是靶心
            if label_name == TARGET_LABEL:
                x1, y1, x2, y2 = det_result["boxes"][i]

                # 映射到显示坐标
                x = int(x1 * display_size[0] // det_app.rgb888p_size[0])
                y = int(y1 * display_size[1] // det_app.rgb888p_size[1])
                w = int((x2 - x1) * display_size[0] // det_app.rgb888p_size[0])
                h = int((y2 - y1) * display_size[1] // det_app.rgb888p_size[1])
                area = w * h

                # 选择面积最大的靶心
                if area > max_area:
                    max_area = area
                    cx = x + w // 2
                    cy = y + h // 2
                    target_center = (cx, cy)

                    # 绘制当前选中的靶心（红色，更粗）
                    pl.osd_img.draw_rectangle(x, y, w, h, color=(255, 0, 0, 255), thickness=4)
                    pl.osd_img.draw_cross(cx, cy, color=(255, 255, 0, 255), thickness=2)
                    pl.osd_img.draw_string_advanced(x, y-40, 24, f"TARGET {det_result['scores'][i]:.2f}", color=(255, 0, 0, 255))

    # --- 舵机控制逻辑 ---
    if target_center is not None:
        cx, cy = target_center

        # 水平控制
        speed_percent = x_pid.update(cx)
        speed_percent = max(-100, min(100, speed_percent))
        pwm_x.duty(speed_to_duty(speed_percent))

        # 垂直控制
        y_output = y_pid.update(cy)
        y_angle = max(0, min(180, y_angle - y_output))  # 注意方向
        pwm_y.duty(angle_to_duty(y_angle))

        print(f"🎯 Tracking: ({cx}, {cy}) | Speed: {speed_percent:+.1f}% | Angle: {y_angle:.1f}°")
    else:
        # 无靶心时停止水平舵机
        pwm_x.duty(speed_to_duty(0))
        print("🔍 No target detected.")

    # 显示 OSD 图像
    pl.show_image()
    gc.collect()  # 手动触发垃圾回收
    print(f"📊 FPS: {clock.fps():.2f}")
