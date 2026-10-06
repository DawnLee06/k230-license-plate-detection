# K230 车牌检测与识别

基于 **嘉楠 CanMV K230** 开发板的实时车牌检测与识别系统。使用 KPU 硬件加速推理车牌检测 + 车牌识别双模型，识别结果通过串口按自定义协议对外输出。

## 功能

- **实时车牌检测**：YOLO 系检测模型定位车牌区域（`LicenceDetectionApp`）
- **车牌字符识别**：识别模型对检测框裁剪区域做 OCR（`LicenceRec`）
- **结果串口输出**：通过 `YbUart`（115200）按 `YbProtocol` 自定义协议封装车牌数据发出
- **多目标跟踪**：跨帧 IOU 匹配，同一车牌复用识别结果，跳过冗余 OCR
- **结果投票**：指数衰减加权投票，抑制单帧误识别
- **检测框平滑**：移动平均，减少框体抖动
- **舵机追踪**（附）：以最大靶心为目标，PID 控制水平/垂直舵机跟随

## 硬件

| 部件 | 说明 |
|---|---|
| 主控 | CanMV K230（K230D 核心板） |
| 显示 | LCD / HDMI |
| 通讯 | UART0，115200 8N1 |
| 舵机（追踪例程） | GPIO42 → PWM0（水平 360° 连续旋转）、GPIO43 → PWM1（垂直 180°） |

## 目录结构

```
.
├── src/
│   ├── lic_det_1.py              # 车牌检测 + 识别（基础版，含串口输出）
│   ├── lic_det_1_fixed.py        # 修复版（串口/资源管理修正）
│   ├── lic_det_1_optimized.py    # 优化版 v2：帧跳跃 + IOU 跟踪 + 衰减投票 + 框平滑
│   └── target_tracking.py        # 靶心追踪：PID + K230 硬件 PWM 驱舵机
└── docs/                         # 说明与效果图（自行补充）
```

## 运行环境

- **固件**：CanMV K230 MicroPython 固件（含 `libs`、`media`、`nncase_runtime`、`aidemo` 模块）
- **模型**：需自行准备并放入 SD 卡

```
/sdcard/mp_deployment_source/
├── deploy_config.json      # 部署配置：kmodel_path / categories / 阈值 / anchors
├── *.kmodel                # 检测模型 + 识别模型
```

`deploy_config.json` 关键字段：

```json
{
  "kmodel_path": "your_detect_model.kmodel",
  "categories": ["license_plate"],
  "confidence_threshold": 0.5,
  "nms_threshold": 0.45,
  "img_size": [224, 224],
  "nms_option": false,
  "model_type": "AnchorBaseDet",
  "anchors": [[], [], []]
}
```

## 使用

1. 将 `src/lic_det_1_optimized.py` 与 `deploy_config.json`、模型文件拷入 SD 卡
2. 将脚本在 CanMV IDE 中打开并运行（或重命名为 `main.py` 开机自启）
3. 画面中车牌会被框出，识别结果经 UART 发往下位机

## 关键实现

### 1. 帧跳跃 + IOU 跟踪

```python
DET_SKIP         = 3       # 每 3 帧做一次检测，中间帧复用跟踪结果
TRACK_MAX_AGE    = 3000    # 跟踪目标最大存活时间(ms)
TRACK_DIST_RATIO = 0.3     # 中心点匹配阈值（相对框宽）
VOTE_DECAY_TAU   = 2000    # 投票指数衰减时间常数(ms)
```

检测帧之间存在空档，靠跟踪补位；同一车牌跨帧只做一次 OCR，实测可显著降负载。

### 2. 指数衰减投票

识别结果按时间加权累计，近期结果权重更高，兼顾稳定性与响应速度。

### 3. 自定义串口协议

```python
uart = YbUart(baudrate=115200)
pto  = YbProtocol()
# 识别完成后将车牌数据封装发送
```

### 4. 舵机 PID 控制（target_tracking.py）

```python
def speed_to_duty(speed_percent):        # 360° 舵机：速度 → 占空比
    pulse_ms = 1.5 + (speed_percent / 100.0) * 0.2
    return (pulse_ms / 20.0) * 100

def angle_to_duty(angle):                # 标准舵机：角度 → 占空比
    pulse_ms = 0.5 + (angle / 180.0) * 2.0
    return (pulse_ms / 20.0) * 100
```

PID 带 15 像素死区，水平轴走速度控制、垂直轴走角度增量控制。

## 已知问题

- `target_tracking.py` 中 `TARGET_LABEL` 常量未定义，运行前需按实际模型类别名补上（如 `TARGET_LABEL = "target"`）
- 需与配套硬件（串口下位机）配合才能完整验证通讯链路

## 依赖

- CanMV K230 官方固件与 `libs` 包（`PipeLine` / `AIBase` / `AI2D` / `YbProtocol`）
- 模型需自行训练或获取，**本仓库不含模型权重**

## License

MIT（仅覆盖本人编写的业务逻辑代码）
