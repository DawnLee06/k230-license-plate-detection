# -*- coding: utf-8 -*-
# 导入必要的模块
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

# 初始化串口通信（用于发送识别结果）
uart = YbUart(baudrate=115200)
pto = YbProtocol()

lr = None  # 全局变量，用于保存 LicenceRec 实例


# 车牌检测类（负责定位车牌区域）
class LicenceDetectionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, confidence_threshold=0.6, nms_threshold=0.4,
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
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            det_res = aidemo.licence_det_postprocess(
                results,
                [self.rgb888p_size[1], self.rgb888p_size[0]],
                self.model_input_size,
                self.confidence_threshold,
                self.nms_threshold
            )
            return det_res


# 车牌字符识别类（OCR）
class LicenceRecognitionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[1920, 1080],
                 display_size=[1920, 1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.dict_rec = [
            "挂", "使", "领", "澳", "港", "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
            "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云",
            "藏", "陕", "甘", "青", "宁", "新", "警", "学", "0", "1", "2", "3", "4", "5", "6", "7", "8",
            "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z", "_", "-"
        ]
        self.dict_size = len(self.dict_rec)
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        # 缓存上次配置的输入尺寸，避免重复 build
        self._last_input_size = None

    def config_preprocess(self, input_image_size=None):
        ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size

        # 如果尺寸与上次相同，跳过重复 build，节省时间与内存
        if self._last_input_size is not None:
            if (self._last_input_size[0] == ai2d_input_size[0] and
                    self._last_input_size[1] == ai2d_input_size[1]):
                return

        self._last_input_size = [ai2d_input_size[0], ai2d_input_size[1]]

        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            output_data = results[0].reshape((-1, self.dict_size))
            max_indices = np.argmax(output_data, axis=1)
            result_str = ""
            for i in range(max_indices.shape[0]):
                index = max_indices[i]
                if index > 0 and (i == 0 or index != max_indices[i - 1]):
                    result_str += self.dict_rec[index - 1]
            return result_str


# 车牌识别主流程类（整合检测 + 识别 + 投票机制 + 启动静默期）
class LicenceRec:
    def __init__(self, licence_det_kmodel, licence_rec_kmodel, det_input_size, rec_input_size,
                 confidence_threshold=0.6, nms_threshold=0.4, rgb888p_size=[1920, 1080],
                 display_size=[1920, 1080], debug_mode=0, vote_interval=5000):
        self.licence_det_kmodel = licence_det_kmodel
        self.licence_rec_kmodel = licence_rec_kmodel
        self.det_input_size = det_input_size
        self.rec_input_size = rec_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode

        # 投票机制参数
        self.vote_interval = vote_interval          # 毫秒
        self.plate_buffer = []                      # [(timestamp, plate), ...]
        self.reported_in_window = None               # 当前窗口已上报车牌
        self.last_report_time = 0                    # 上次上报时间（防抖）
        self.boot_time = time.ticks_ms()            # 记录启动时间

        # 初始化子模块
        self.licence_det = LicenceDetectionApp(
            self.licence_det_kmodel,
            model_input_size=self.det_input_size,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            rgb888p_size=self.rgb888p_size,
            display_size=self.display_size,
            debug_mode=0
        )
        self.licence_rec = LicenceRecognitionApp(
            self.licence_rec_kmodel,
            model_input_size=self.rec_input_size,
            rgb888p_size=self.rgb888p_size
        )

        self.licence_det.config_preprocess()

    def _clean_buffer(self, current_time):
        """清理过期数据"""
        self.plate_buffer[:] = [
            (ts, plate) for ts, plate in self.plate_buffer
            if time.ticks_diff(current_time, ts) < self.vote_interval
        ]

    def _get_most_frequent_plate(self, min_count=2):
        """返回出现次数最多的有效车牌（至少 min_count 次）"""
        if not self.plate_buffer:
            return None
        valid_plates = [plate.strip() for _, plate in self.plate_buffer
                        if len(plate.strip()) >= 5]
        if not valid_plates:
            return None

        freq = {}
        for p in valid_plates:
            freq[p] = freq.get(p, 0) + 1

        best_plate, max_count = None, 0
        for plate, cnt in freq.items():
            if cnt >= min_count and cnt > max_count:
                max_count = cnt
                best_plate = plate
        return best_plate

    def run(self, input_np):
        """执行完整车牌识别流程"""
        det_boxes = self.licence_det.run(input_np)
        imgs_array_boxes = aidemo.ocr_rec_preprocess(
            input_np,
            [self.rgb888p_size[1], self.rgb888p_size[0]],
            det_boxes
        )
        imgs_array = imgs_array_boxes[0]
        boxes = imgs_array_boxes[1]

        rec_res = []
        for img_array in imgs_array:
            self.licence_rec.config_preprocess(
                input_image_size=[img_array.shape[3], img_array.shape[2]]
            )
            licence_str = self.licence_rec.run(img_array)
            rec_res.append(licence_str)
            gc.collect()
        return det_boxes, rec_res

    def draw_result(self, pl, det_res, rec_res):
        """绘制结果，并将车牌加入缓冲区；尝试投票上报"""
        pl.osd_img.clear()
        current_ticks = time.ticks_ms()

        runtime = time.ticks_diff(current_ticks, self.boot_time)
        is_warmup_phase = runtime < 3000  # 前3秒为静默期

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
                        point_8[i * 2 + 0],
                        point_8[i * 2 + 1],
                        point_8[(i + 1) % 4 * 2 + 0],
                        point_8[(i + 1) % 4 * 2 + 1],
                        color=(255, 0, 255, 0),
                        thickness=4
                    )

                plate_text = rec_res[det_index]
                pl.osd_img.draw_string_advanced(
                    point_8[6], point_8[7] + 20, 40, plate_text,
                    color=(255, 255, 153, 18)
                )

                # 加入缓冲区（仅在非静默期且有效时）
                if not is_warmup_phase and plate_text and len(plate_text.strip()) >= 5:
                    self.plate_buffer.append((current_ticks, plate_text.strip()))

        # 清理过期数据并尝试投票
        self._clean_buffer(current_ticks)

        if not is_warmup_phase:
            top_plate = self._get_most_frequent_plate(min_count=3)

            if top_plate and top_plate != self.reported_in_window:
                if time.ticks_diff(current_ticks, self.last_report_time) >= self.vote_interval:
                    self.reported_in_window = top_plate
                    self.last_report_time = current_ticks
                    pto_data = pto.get_licence_rec_data(top_plate)
                    uart.send(pto_data)
                    print(f"[UART VOTE SEND] {top_plate}")


# 主运行函数
def exce_demo(pl):
    global lr

    display_mode = pl.display_mode
    rgb888p_size = pl.rgb888p_size
    display_size = pl.display_size

    # 模型路径与参数
    licence_det_kmodel_path = "/sdcard/kmodel/LPD_640.kmodel"
    licence_rec_kmodel_path = "/sdcard/kmodel/licence_reco.kmodel"
    licence_det_input_size = [640, 640]
    licence_rec_input_size = [220, 32]
    confidence_threshold = 0.6
    nms_threshold = 0.4

    try:
        lr = LicenceRec(
            licence_det_kmodel_path,
            licence_rec_kmodel_path,
            det_input_size=licence_det_input_size,
            rec_input_size=licence_rec_input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            rgb888p_size=rgb888p_size,
            display_size=display_size,
            vote_interval=5000
        )
        while True:
            with ScopedTiming("total", 0):
                img = pl.get_frame()
                det_res, rec_res = lr.run(img)
                lr.draw_result(pl, det_res, rec_res)
                pl.show_image()
                gc.collect()
    except Exception as e:
        print("车牌识别功能退出:", e)
    finally:
        exit_demo()


# 退出清理
def exit_demo():
    global lr
    if lr:
        lr.licence_det.deinit()
        lr.licence_rec.deinit()


# 程序入口
if __name__ == "__main__":
    rgb888p_size = [320, 240]
    display_size = [640, 480]
    display_mode = "lcd"

    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create()
    exce_demo(pl)
