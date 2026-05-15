# polar_h10_emotion_realtime.py (最终完美版)
import asyncio
import numpy as np
from bleak import BleakClient, BleakScanner
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
import time
import traceback
import warnings
import json
import sys

sys.path.append('D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM')
import data_utils as du

warnings.filterwarnings('ignore')

custom_objects = {}


class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            del kwargs['batch_shape']
        if 'optional' in kwargs:
            del kwargs['optional']
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) == 3:
                query, key, value = inputs
            elif len(inputs) == 2:
                query, value = inputs
                key = value
            else:
                query = inputs
                key = inputs
                value = inputs
        else:
            query = inputs
            key = inputs
            value = inputs

        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output

    def get_config(self):
        return super(DotProductAttention, self).get_config()


def load_model_compatible(model_path):
    try:
        print("尝试直接加载模型...")
        model = load_model(
            model_path,
            custom_objects={'DotProductAttention': DotProductAttention},
            compile=False
        )
        print("✅ 模型加载成功")
        return model
    except Exception as e1:
        print(f"直接加载失败: {e1}")
        try:
            print("尝试重建模型...")
            import Model_selection as Ms
            beat_length = 163
            rhythm_length = 1280
            model = Ms.fusion_model(beat_length, rhythm_length)
            model.load_weights(model_path)
            print("✅ 模型加载成功（重建）")
            return model
        except Exception as e2:
            raise Exception(f"模型加载失败: {e2}")


class ECGProcessor:
    def __init__(self, sampling_rate=130):
        self.sampling_rate = sampling_rate
        self.beat_length = 163
        self.rhythm_length = 1280
        self.target_rate = 256

    def preprocess_like_training(self, ecg_signal):
        """完全模仿训练时的预处理"""
        # 1. 重采样到256Hz
        from scipy import signal
        target_length = int(len(ecg_signal) * self.target_rate / self.sampling_rate)
        ecg_resampled = signal.resample(ecg_signal, target_length)

        # 2. 关键：使用neurokit2的R峰检测（与训练时完全相同）
        try:
            r_peaks = du.get_rpeak(ecg_resampled, sampling_rate=self.target_rate)
        except Exception as e:
            print(f"R峰检测失败: {e}")
            return None, None, None

        if len(r_peaks) < 5:
            return None, None, None

        # 3. 提取beat和rhythm（与训练时完全相同）
        beat_segments = du.load_BEAT_data(ecg_resampled, r_peaks, left=61, right=102)
        rhythm_segments = du.load_RHYTHM_data(ecg_resampled, r_peaks, window=1280, right=102)

        if len(beat_segments) == 0 or len(rhythm_segments) == 0:
            return None, None, None

        # 取平均
        beat = np.mean(beat_segments, axis=0)
        rhythm = rhythm_segments[0]  # 取第一个rhythm窗口

        return beat, rhythm, r_peaks


class EmotionRecognizer:
    def __init__(self, model_path):
        print(f"加载模型: {model_path}")
        self.model = load_model_compatible(model_path)
        self.processor = ECGProcessor(sampling_rate=130)
        self.emotion_labels = {0: "😆 兴奋", 1: "😐 中性", 2: "😰 压力"}
        self.prediction_history = []  # 用于平滑预测
        print("✅ 模型加载完成")

    def smooth_prediction(self, new_pred):
        """平滑预测，避免抖动"""
        self.prediction_history.append(new_pred)
        if len(self.prediction_history) > 3:
            self.prediction_history.pop(0)
        # 返回最近3次预测的平均
        avg = np.mean(self.prediction_history, axis=0)
        return avg

    def predict_from_ecg(self, ecg_raw_data):
        if len(ecg_raw_data) < 3000:
            return None

        ecg_signal = np.array(ecg_raw_data[-3000:])

        # 使用训练时的预处理
        beat, rhythm, r_peaks = self.processor.preprocess_like_training(ecg_signal)

        if beat is None or rhythm is None or len(r_peaks) < 5:
            return None

        try:
            beat_input = beat.reshape(1, -1, 1)
            rhythm_input = rhythm.reshape(1, -1, 1)

            prediction = self.model.predict([beat_input, rhythm_input], verbose=0)[0]

            # 平滑预测
            smoothed_pred = self.smooth_prediction(prediction)
            emotion_idx = np.argmax(smoothed_pred)
            confidence = np.max(smoothed_pred)

            # 计算心率和HRV
            rr_intervals = np.diff(r_peaks) / 256
            hr = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
            hrv = np.std(rr_intervals) * 1000 if len(rr_intervals) > 1 else 0

            # 调试输出
            print(f"\n📊 模型原始输出: 兴奋={prediction[0]:.3f}, 中性={prediction[1]:.3f}, 压力={prediction[2]:.3f}")
            print(
                f"📊 平滑后输出: 兴奋={smoothed_pred[0]:.3f}, 中性={smoothed_pred[1]:.3f}, 压力={smoothed_pred[2]:.3f}")

            return {
                'emotion': self.emotion_labels[emotion_idx],
                'confidence': confidence,
                'probabilities': {
                    '兴奋': float(smoothed_pred[0]),
                    '中性': float(smoothed_pred[1]),
                    '压力': float(smoothed_pred[2])
                },
                'r_peaks_count': len(r_peaks),
                'hr': hr,
                'hrv': hrv
            }
        except Exception as e:
            print(f"预测失败: {e}")
            traceback.print_exc()
            return None


# Polar H10 连接部分（保持不变）
ECG_UUID = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL_UUID = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"


class PolarH10RealtimeEmotion:
    def __init__(self, model_path, device_address=None):
        self.recognizer = EmotionRecognizer(model_path)
        self.device_address = device_address
        self.client = None
        self.ecg_buffer = deque(maxlen=6000)
        self.prediction_count = 0
        self.start_time = None
        self.on_prediction = None

    def set_websocket_callback(self, callback):
        self.on_prediction = callback

    def parse_ecg_packet(self, data):
        samples = []
        for i in range(10, len(data), 3):
            if i + 3 <= len(data):
                sample = int.from_bytes(data[i:i + 3], byteorder='little', signed=True)
                samples.append(sample)
        return samples

    def ecg_callback(self, sender, data):
        try:
            samples = self.parse_ecg_packet(data)
            self.ecg_buffer.extend(samples)
            buffer_sec = len(self.ecg_buffer) / 130
            print(f"\r📊 ECG数据: {buffer_sec:.1f}秒", end="", flush=True)
        except Exception as e:
            print(f"\n回调错误: {e}")

    async def scan_device(self, timeout=5):
        print(f"扫描 Polar H10 设备（{timeout}秒）...")
        devices = await BleakScanner.discover(timeout=timeout)
        for device in devices:
            if device.name and "Polar" in device.name:
                print(f"✅ 找到设备: {device.name} ({device.address})")
                return device
        return None

    async def connect_and_stream(self):
        if not self.device_address:
            device = await self.scan_device()
            if not device:
                print("❌ 未找到设备")
                return
            self.device_address = device.address

        print(f"\n连接到 {self.device_address}...")
        self.client = BleakClient(self.device_address, timeout=15.0)

        try:
            await self.client.connect()
            print("✅ 连接成功")

            print("激活ECG数据流...")
            try:
                await self.client.write_gatt_char(
                    PMD_CONTROL_UUID,
                    bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00]),
                    response=True
                )
                print("✅ ECG激活成功")
            except Exception as e:
                print(f"❌ ECG激活失败: {e}")
                return

            await asyncio.sleep(1.0)
            await self.client.start_notify(ECG_UUID, self.ecg_callback)

            self.start_time = time.time()

            print("\n" + "=" * 55)
            print("   🫀 实时情绪监测已启动")
            print("   每5秒进行一次情绪预测...")
            print("   按 Ctrl+C 停止")
            print("=" * 55 + "\n")

            last_prediction_time = 0
            prediction_interval = 5

            while True:
                await asyncio.sleep(0.5)
                current_time = time.time()
                if current_time - last_prediction_time >= prediction_interval:
                    if len(self.ecg_buffer) >= 4000:
                        self.prediction_count += 1
                        print(f"\n\n{'─' * 50}")
                        print(f"📋 第 {self.prediction_count} 次情绪预测")
                        print(f"{'─' * 50}")

                        result = self.recognizer.predict_from_ecg(list(self.ecg_buffer))

                        if result:
                            elapsed = time.time() - self.start_time
                            print(f"⏱  运行时间: {elapsed:.0f}秒")
                            print(f"💓 心率: {result['hr']:.0f} bpm")
                            print(f"📊 HRV: {result['hrv']:.1f} ms")
                            print(f"🔍 R峰: {result['r_peaks_count']}个")
                            print(f"🎯 情绪: {result['emotion']}")
                            print(f"📊 置信度: {result['confidence']:.1%}")
                            print(f"   兴奋: {result['probabilities']['兴奋']:.1%} | "
                                  f"中性: {result['probabilities']['中性']:.1%} | "
                                  f"压力: {result['probabilities']['压力']:.1%}")

                            if self.on_prediction:
                                try:
                                    await self.on_prediction({
                                        "type": "emotion_prediction",
                                        "data": {
                                            "emotion": result["emotion"],
                                            "confidence": float(result["confidence"]),
                                            "probabilities": result["probabilities"],
                                            "hr": float(result["hr"]),
                                            "hrv": float(result["hrv"])
                                        }
                                    })
                                    print("   ✅ 已广播")
                                except Exception as e:
                                    print(f"广播失败: {e}")
                        else:
                            print("⚠️ 预测失败")
                        print(f"{'─' * 50}\n")
                    else:
                        print(f"\n⏳ 收集数据... {len(self.ecg_buffer)}/4000点")
                    last_prediction_time = current_time

        except KeyboardInterrupt:
            print("\n\n⏹️ 停止监测")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            traceback.print_exc()
        finally:
            if self.client and self.client.is_connected:
                try:
                    await self.client.stop_notify(ECG_UUID)
                    await self.client.disconnect()
                    print("✅ 已断开连接")
                except:
                    pass


async def main():
    MODEL_PATH = "ECG_Model/Fusion_model_Attention_7Convlayer_FINAL_MODEL.h5"
    DEVICE_ADDRESS = "A0:9E:1A:E9:FD:44"

    print("=" * 55)
    print("     Polar H10 实时情绪识别系统 v3.0")
    print("     使用训练时相同的预处理")
    print("=" * 55)

    app = PolarH10RealtimeEmotion(MODEL_PATH, DEVICE_ADDRESS)
    await app.connect_and_stream()


if __name__ == "__main__":
    asyncio.run(main())