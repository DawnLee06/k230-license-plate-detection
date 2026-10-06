# -*- coding: utf-8 -*-
# 车牌检测与识别 - 算法优化版（v2 带调试）
# 优化点：
#   1. 帧跳跃检测   —— 每N帧做一次检测，中间帧复用结果
#   2. IOU目标跟踪   —— 同一车牌跨帧匹配，跳过冗余OCR
#   3. 指数衰减投票  —— 近期识别结果权重更高，响应更快
#   4. 检测框平滑    —— 移动平均减少抖动

from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import ujson
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import image
import aidemo
import gc

from libs.YbProtocol import YbProtocol
from ybUtils.YbUart import YbUart

# ─── 可调参数 ─────────────────────────────────────────────
DET_SKIP         = 3       # 检测帧跳跃
TRACK_MAX_AGE    = 3000    # 跟踪目标最大存活时间(ms)
TRACK_DIST_RATIO = 0.3     # 中心点匹配阈值
VOTE_DECAY_TAU   = 2000    # 指数衰减时间常数(ms)
DEBUG            = True    # 调试开关

# ─── 初始化串口 ───────────────────────────────────────────
uart = YbUart(baudrate=115200)
pto = YbProtocol()
lr = None


def _box_center(box):
    return ((box[0] + box[2] + box[4] + box[6]) / 4.0,
            (box[1] + box[3] + box[5] + box[7]) / 4.0)


def _box_diag(box):
    cx, cy = _box_center(box)
    dx = max(abs(box[0] - cx), abs(box[2] - cx), abs(box[4] - cx), abs(box[6] - cx))
    dy = max(abs(box[1] - cy), abs(box[3] - cy), abs(box[5] - cy), abs(box[7] - cy))
    return (dx * dx + dy * dy) ** 0.5


def _exp_weight(age_ms, tau=VOTE_DECAY_TAU):
    x = -age_ms / float(tau)
    if x < -6.0:
        return 0.001
    if x > 0:
        return 1.0
    return max(0.001, 1.0 + x + x * x / 2.0 + x * x * x / 6.0)


# ╔══════════════════════════════════════════════════════════╗
# ║              车牌检测类                                  ║
# ╚══════════════════════════════════════════════════════════╝

class LicenceDetectionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size,
                 confidence_threshold=0.6, nms_threshold=0.4,
                 rgb888p_size=[224, 224], display_size=[1920, 1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build(
            [1, 3, ai2d_input_size[1], ai2d_input_size[0]],
            [1, 3, self.model_input_size[1], self.model_input_size[0]]
        )

    def postprocess(self, results):
        return aidemo.licence_det_postprocess(
            results,
            [self.rgb888p_size[1], self.rgb888p_size[0]],
            self.model_input_size,
            self.confidence_threshold,
            self.nms_threshold
        )


# ╔══════════════════════════════════════════════════════════╗
# ║              车牌识别类（OCR）                           ║
# ╚══════════════════════════════════════════════════════════╝

class LicenceRecognitionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size,
                 rgb888p_size=[1920, 1080], display_size=[1920, 1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.dict_rec = [
            "挂","使","领","澳","港","皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
            "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云",
            "藏","陕","甘","青","宁","新","警","学",
            "0","1","2","3","4","5","6","7","8","9",
            "A","B","C","D","E","F","G","H","J","K","L","M",
            "N","P","Q","R","S","T","U","V","W","X","Y","Z","_","-"
        ]
        self.dict_size = len(self.dict_rec)
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        self._last_input_size = None

    def config_preprocess(self, input_image_size=None):
        ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
        if self._last_input_size is not None:
            if (self._last_input_size[0] == ai2d_input_size[0] and
                    self._last_input_size[1] == ai2d_input_size[1]):
                return
        self._last_input_size = [ai2d_input_size[0], ai2d_input_size[1]]
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build(
            [1, 3, ai2d_input_size[1], ai2d_input_size[0]],
            [1, 3, self.model_input_size[1], self.model_input_size[0]]
        )

    def postprocess(self, results):
        output_data = results[0].reshape((-1, self.dict_size))
        max_indices = np.argmax(output_data, axis=1)
        result_str = ""
        for i in range(max_indices.shape[0]):
            index = max_indices[i]
            if index > 0 and (i == 0 or index != max_indices[i - 1]):
                result_str += self.dict_rec[index - 1]
        return result_str


# ╔══════════════════════════════════════════════════════════╗
# ║              目标跟踪器                                  ║
# ╚══════════════════════════════════════════════════════════╝

class TargetTracker:
    def __init__(self, max_age=TRACK_MAX_AGE, dist_ratio=TRACK_DIST_RATIO):
        self.max_age = max_age
        self.dist_ratio = dist_ratio
        self.tracks = []
        self.next_id = 0

    def update(self, boxes):
        current_time = time.ticks_ms()
        n = len(boxes)

        box_info = []
        for b in boxes:
            cx, cy = _box_center(b)
            diag = _box_diag(b)
            box_info.append((cx, cy, diag))

        matched_track = [-1] * len(self.tracks)
        matched_box   = [-1] * n
        ocr_from_cache = [''] * n

        for j in range(n):
            cx, cy, diag = box_info[j]
            best_track = -1
            best_dist = float('inf')
            for i, t in enumerate(self.tracks):
                if matched_track[i] != -1:
                    continue
                tc = t['center']
                dist = ((cx - tc[0]) ** 2 + (cy - tc[1]) ** 2) ** 0.5
                threshold = self.dist_ratio * max(diag, t['diag'])
                if dist < threshold and dist < best_dist:
                    best_dist = dist
                    best_track = i

            if best_track >= 0:
                matched_track[best_track] = j
                matched_box[j] = best_track
                ocr_from_cache[j] = self.tracks[best_track].get('ocr', '')

        new_tracks = []
        need_ocr = []

        for i, t in enumerate(self.tracks):
            age = time.ticks_diff(current_time, t['last_seen'])
            if age > self.max_age:
                continue
            j = matched_track[i]
            if j >= 0:
                alpha = 0.6
                bx, by, bd = box_info[j]
                t['center'] = (alpha * bx + (1 - alpha) * t['center'][0],
                               alpha * by + (1 - alpha) * t['center'][1])
                t['diag'] = alpha * bd + (1 - alpha) * t['diag']
                t['box'] = boxes[j]
                t['last_seen'] = current_time
            new_tracks.append(t)

        for j in range(n):
            if matched_box[j] < 0:
                cx, cy, diag = box_info[j]
                new_tracks.append({
                    'id': self.next_id,
                    'box': boxes[j],
                    'center': (cx, cy),
                    'diag': diag,
                    'ocr': '',
                    'last_seen': current_time
                })
                self.next_id += 1
                need_ocr.append(j)

        self.tracks = new_tracks
        return need_ocr, ocr_from_cache


# ╔══════════════════════════════════════════════════════════╗
# ║              指数衰减投票器                              ║
# ╚══════════════════════════════════════════════════════════╝

class DecayVoter:
    def __init__(self, vote_interval=5000):
        self.vote_interval = vote_interval
        self.buffer = []

    def add(self, plate, timestamp):
        if plate and len(plate.strip()) >= 5:
            self.buffer.append((timestamp, plate.strip()))

    def clean(self, current_time):
        self.buffer[:] = [
            (ts, p) for ts, p in self.buffer
            if time.ticks_diff(current_time, ts) < self.vote_interval
        ]

    def decide(self, current_time, min_count=2):
        self.clean(current_time)
        if len(self.buffer) < min_count:
            return None

        scores = {}
        counts = {}
        for ts, plate in self.buffer:
            age = time.ticks_diff(current_time, ts)
            w = _exp_weight(age)
            scores[plate] = scores.get(plate, 0.0) + w
            counts[plate] = counts.get(plate, 0) + 1

        best, best_score = None, 0.0
        for plate, sc in scores.items():
            if counts[plate] >= min_count and sc > best_score:
                best_score = sc
                best = plate
        return best


# ╔══════════════════════════════════════════════════════════╗
# ║              主流程类                                    ║
# ╚══════════════════════════════════════════════════════════╝

class LicenceRec:
    def __init__(self, licence_det_kmodel, licence_rec_kmodel,
                 det_input_size, rec_input_size,
                 confidence_threshold=0.6, nms_threshold=0.4,
                 rgb888p_size=[1920, 1080], display_size=[1920, 1080],
                 debug_mode=0, vote_interval=5000, det_skip=DET_SKIP):

        self.det_input_size = det_input_size
        self.rec_input_size = rec_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]

        self.det_skip = det_skip
        self.frame_count = 0
        self.cached_det_boxes = []
        self.cached_rec_res = []

        self.tracker = TargetTracker()
        self.voter = DecayVoter(vote_interval=vote_interval)

        self.reported_plate = None
        self.last_report_time = 0
        self.boot_time = time.ticks_ms()

        self.licence_det = LicenceDetectionApp(
            licence_det_kmodel,
            model_input_size=det_input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            rgb888p_size=self.rgb888p_size,
            display_size=self.display_size,
            debug_mode=0
        )
        self.licence_rec = LicenceRecognitionApp(
            licence_rec_kmodel,
            model_input_size=rec_input_size,
            rgb888p_size=self.rgb888p_size
        )
        self.licence_det.config_preprocess()

    def run(self, input_np):
        self.frame_count += 1

        # ── 帧跳跃检测 ──
        if self.frame_count % self.det_skip == 1:
            det_boxes = self.licence_det.run(input_np)
        else:
            det_boxes = self.cached_det_boxes

        if not det_boxes:
            self.cached_det_boxes = []
            self.cached_rec_res = []
            return [], []

        # OCR 预处理
        imgs_array_boxes = aidemo.ocr_rec_preprocess(
            input_np,
            [self.rgb888p_size[1], self.rgb888p_size[0]],
            det_boxes
        )
        imgs_array = imgs_array_boxes[0]

        # ── 目标跟踪：获取缓存OCR + 新目标列表 ──
        need_ocr, ocr_from_cache = self.tracker.update(det_boxes)
        rec_res = list(ocr_from_cache)

        # 对新目标做 OCR
        for idx in need_ocr:
            if idx < len(imgs_array):
                self.licence_rec.config_preprocess(
                    input_image_size=[imgs_array[idx].shape[3], imgs_array[idx].shape[2]]
                )
                plate = self.licence_rec.run(imgs_array[idx])
                rec_res[idx] = plate

                # 回写 tracker 缓存
                for t in self.tracker.tracks:
                    if t['box'] is det_boxes[idx]:
                        t['ocr'] = plate
                        break

                if DEBUG:
                    print(f"[OCR #{idx}] '{plate}'")

                gc.collect()

        self.cached_det_boxes = det_boxes
        self.cached_rec_res = rec_res

        return det_boxes, rec_res

    def draw_result(self, pl, det_res, rec_res):
        pl.osd_img.clear()
        current_ticks = time.ticks_ms()
        runtime = time.ticks_diff(current_ticks, self.boot_time)
        is_warmup = runtime < 3000

        if det_res:
            point_8 = np.zeros((8), dtype=np.int16)
            for det_index in range(len(det_res)):
                for i in range(4):
                    x = det_res[det_index][i * 2 + 0] / self.rgb888p_size[0] * self.display_size[0]
                    y = det_res[det_index][i * 2 + 1] / self.rgb888p_size[1] * self.display_size[1]
                    point_8[i * 2 + 0] = int(x)
                    point_8[i * 2 + 1] = int(y)

                for i in range(4):
                    pl.osd_img.draw_line(
                        point_8[i * 2 + 0], point_8[i * 2 + 1],
                        point_8[(i + 1) % 4 * 2 + 0], point_8[(i + 1) % 4 * 2 + 1],
                        color=(0, 255, 0, 255),
                        thickness=4
                    )

                plate_text = rec_res[det_index] if det_index < len(rec_res) else ''
                if plate_text:
                    pl.osd_img.draw_string_advanced(
                        point_8[6], point_8[7] + 20, 40, plate_text,
                        color=(255, 0, 255, 255)
                    )

                if not is_warmup:
                    self.voter.add(plate_text, current_ticks)

        if not is_warmup:
            winner = self.voter.decide(current_ticks, min_count=3)
            if winner and winner != self.reported_plate:
                if time.ticks_diff(current_ticks, self.last_report_time) >= self.voter.vote_interval:
                    self.reported_plate = winner
                    self.last_report_time = current_ticks
                    pto_data = pto.get_licence_rec_data(winner)
                    uart.send(pto_data)
                    if DEBUG:
                        print(f"[UART SEND] {winner}")


# ╔══════════════════════════════════════════════════════════╗
# ║              主函数                                      ║
# ╚══════════════════════════════════════════════════════════╝

def exce_demo(pl):
    global lr

    licence_det_kmodel = "/sdcard/kmodel/LPD_640.kmodel"
    licence_rec_kmodel = "/sdcard/kmodel/licence_reco.kmodel"

    lr = LicenceRec(
        licence_det_kmodel,
        licence_rec_kmodel,
        det_input_size=[640, 640],
        rec_input_size=[220, 32],
        confidence_threshold=0.6,
        nms_threshold=0.4,
        rgb888p_size=pl.rgb888p_size,
        display_size=pl.display_size,
        vote_interval=5000,
        det_skip=DET_SKIP
    )
    while True:
        with ScopedTiming("total", 0):
            img = pl.get_frame()
            det_res, rec_res = lr.run(img)
            lr.draw_result(pl, det_res, rec_res)
            pl.show_image()
            gc.collect()


def exit_demo():
    global lr
    if lr:
        lr.licence_det.deinit()
        lr.licence_rec.deinit()


if __name__ == "__main__":
    pl = PipeLine(rgb888p_size=[320, 240], display_size=[640, 480], display_mode="lcd")
    pl.create()
    try:
        exce_demo(pl)
    except Exception as e:
        import sys
        sys.print_exception(e)
    finally:
        exit_demo()
