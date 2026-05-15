# polar_h10_emotion_fixed.py
import asyncio
import numpy as np
from bleak import BleakClient, BleakScanner
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
from scipy.signal import butter, filtfilt


# ==================== 自定义注意力层 ====================
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
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
        config = super(DotProductAttention, self).get_config()
        return config


# ==================== ECG信号处理器（修复版）====================
class ECGProcessor:
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.rr_buffer = []
        self.beat_length = 163  # left(61) + right(102) = 163
        self.rhythm_length = 1280  # 5秒 * 256Hz = 1280

    def add_rr_interval(self, rr_ms):
        self.rr_buffer.append(rr_ms)
        if len(self.rr_buffer) > 300:
            self.rr_buffer.pop(0)

    def reconstruct_ecg_from_rr(self, duration_seconds=5):
        """重建足够长度的ECG信号"""
        if len(self.rr_buffer) < 30:
            return None

        # 需要的数据长度：beat + rhythm + 额外空间
        needed_samples = self.beat_length + self.rhythm_length + 200
        ecg_signal = []

        # 使用平均RR间期
        avg_rr = np.mean(self.rr_buffer[-30:])
        beat_samples = int(self.sampling_rate * (avg_rr / 1000))
        if beat_samples < 50:
            beat_samples = 100

        # 创建单个心跳模板
        t = np.linspace(0, 1, beat_samples)
        # 模拟ECG波形
        p_wave = 0.1 * np.exp(-((t - 0.1) / 0.03) ** 2)
        qrs_wave = 0.8 * np.exp(-((t - 0.25) / 0.02) ** 2) - 0.2 * np.exp(-((t - 0.27) / 0.01) ** 2)
        t_wave = 0.3 * np.exp(-((t - 0.45) / 0.05) ** 2)
        beat_template = p_wave + qrs_wave + t_wave

        # 拼接足够长度
        while len(ecg_signal) < needed_samples:
            ecg_signal.extend(beat_template)

        return np.array(ecg_signal[:needed_samples])

    def preprocess_ecg(self, ecg_signal):
        """完整的信号处理流程"""

        def butter_highpass(data, cutoff=0.5, order=2):
            nyq = 0.5 * self.sampling_rate
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high')
            return filtfilt(b, a, data)

        def butter_bandstop(data, low=57, high=63, order=2):
            nyq = 0.5 * self.sampling_rate
            low = low / nyq
            high = high / nyq
            b, a = butter(order, [low, high], btype='bandstop')
            return filtfilt(b, a, data)

        def butter_lowpass(data, cutoff=100, order=2):
            nyq = 0.5 * self.sampling_rate
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low')
            return filtfilt(b, a, data)

        def minmax_norm(data):
            min_val, max_val = data.min(), data.max()
            if max_val - min_val == 0:
                return data
            return (data - min_val) / (max_val - min_val)

        filtered = butter_highpass(ecg_signal)
        filtered = butter_bandstop(filtered)
        filtered = butter_lowpass(filtered)
        normalized = minmax_norm(filtered)
        return normalized

    def get_r_peaks(self, ecg_signal):
        """R峰检测"""
        diff = np.diff(ecg_signal)
        abs_diff = np.abs(diff)
        threshold = np.mean(abs_diff) * 3
        peaks = []
        for i in range(1, len(abs_diff) - 1):
            if abs_diff[i] > threshold and abs_diff[i] > abs_diff[i - 1] and abs_diff[i] > abs_diff[i + 1]:
                peaks.append(i)
        return np.array(peaks)

    def extract_beat_feature(self, ecg_signal, r_peaks):
        """提取beat特征 (163,)"""
        if len(r_peaks) < 3:
            return None

        left, right = 61, 102
        segments = []
        for i in range(1, min(len(r_peaks) - 1, 10)):  # 取最多10个心跳
            start = r_peaks[i] - left
            end = r_peaks[i] + right
            if start >= 0 and end < len(ecg_signal):
                segments.append(ecg_signal[start:end])

        if len(segments) == 0:
            return None

        # 取平均并确保长度正确
        beat = np.mean(segments, axis=0)
        if len(beat) < self.beat_length:
            # 补齐
            beat = np.pad(beat, (0, self.beat_length - len(beat)))
        elif len(beat) > self.beat_length:
            # 截断
            beat = beat[:self.beat_length]

        return beat

    def extract_rhythm_feature(self, ecg_signal, r_peaks):
        """提取rhythm特征 (1280,)"""
        if len(r_peaks) < 3:
            return None

        right = 102
        segments = []
        for i in range(1, min(len(r_peaks) - 1, 5)):  # 取最多5个心跳
            start = r_peaks[i] + right
            end = start + self.rhythm_length
            if end <= len(ecg_signal):
                segments.append(ecg_signal[start:end])

        if len(segments) == 0:
            return None

        # 取平均并确保长度正确
        rhythm = np.mean(segments, axis=0)
        if len(rhythm) < self.rhythm_length:
            rhythm = np.pad(rhythm, (0, self.rhythm_length - len(rhythm)))
        elif len(rhythm) > self.rhythm_length:
            rhythm = rhythm[:self.rhythm_length]

        return rhythm


# ==================== 情绪识别器 ====================
class EmotionRecognizer:
    def __init__(self, model_path):
        print(f"加载模型: {model_path}")
        self.model = load_model(model_path, custom_objects={'DotProductAttention': DotProductAttention})
        self.processor = ECGProcessor()
        self.emotion_labels = {0: "兴奋", 1: "中性", 2: "压力"}
        print("✅ 模型加载完成")

    def predict_from_rr(self, rr_intervals):
        """从RR间期预测情绪"""
        if len(rr_intervals) < 50:
            return None

        # 清空并重新添加RR间期
        self.processor.rr_buffer = []
        for rr in rr_intervals:
            self.processor.add_rr_interval(rr)

        # 重建ECG信号
        ecg_signal = self.processor.reconstruct_ecg_from_rr(duration_seconds=6)
        if ecg_signal is None or len(ecg_signal) < 1500:
            print(f"ECG重建失败: {len(ecg_signal) if ecg_signal is not None else 0}")
            return None

        # 预处理
        try:
            processed = self.processor.preprocess_ecg(ecg_signal)
        except Exception as e:
            print(f"预处理失败: {e}")
            return None

        # 检测R峰
        r_peaks = self.processor.get_r_peaks(processed)
        if len(r_peaks) < 3:
            print(f"R峰检测失败: {len(r_peaks)}")
            return None

        # 提取特征
        beat = self.processor.extract_beat_feature(processed, r_peaks)
        rhythm = self.processor.extract_rhythm_feature(processed, r_peaks)

        if beat is None or rhythm is None:
            print(f"特征提取失败: beat={beat is not None}, rhythm={rhythm is not None}")
            return None

        # 确保形状正确
        if len(beat) != 163:
            print(f"beat长度错误: {len(beat)}")
            return None
        if len(rhythm) != 1280:
            print(f"rhythm长度错误: {len(rhythm)}")
            return None

        # 预测
        beat_input = beat.reshape(1, -1, 1)
        rhythm_input = rhythm.reshape(1, -1, 1)

        prediction = self.model.predict([beat_input, rhythm_input], verbose=0)
        emotion_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        return {
            'emotion': self.emotion_labels[emotion_idx],
            'confidence': confidence,
            'probabilities': {
                '兴奋': float(prediction[0][0]),
                '中性': float(prediction[0][1]),
                '压力': float(prediction[0][2])
            }
        }


# ==================== Polar H10 连接 ====================
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


class PolarH10Emotion:
    def __init__(self, model_path, device_address=None):
        self.recognizer = EmotionRecognizer(model_path)
        self.device_address = device_address
        self.client = None
        self.rr_intervals = []
        self.last_analysis_time = 0
        self.analysis_interval = 8  # 每8秒分析一次
        self.data_count = 0

    async def scan_device(self, timeout=5):
        """扫描 Polar H10 设备"""
        print(f"扫描 Polar H10 设备（{timeout}秒）...")
        devices = await BleakScanner.discover(timeout=timeout)

        for device in devices:
            if device.name and "polar" in device.name.lower():
                print(f"✅ 找到设备: {device.name}")
                print(f"   地址: {device.address}")
                return device
        return None

    async def connect(self):
        """连接设备"""
        if not self.device_address:
            device = await self.scan_device()
            if not device:
                print("❌ 未找到 Polar H10 设备")
                return False
            self.device_address = device.address

        print(f"正在连接 {self.device_address}...")

        try:
            self.client = BleakClient(self.device_address, timeout=30.0, cache_address=False)
            await self.client.connect()

            if not self.client.is_connected:
                print("❌ 连接失败")
                return False

            print(f"✅ 连接成功")
            return True

        except Exception as e:
            print(f"❌ 连接错误: {e}")
            return False

    def data_handler(self, sender, data):
        """处理心率数据"""
        try:
            self.data_count += 1

            rr_index = 2
            while rr_index < len(data):
                if rr_index + 1 >= len(data):
                    break

                try:
                    rr_bytes = data[rr_index:rr_index + 2]
                    if len(rr_bytes) == 2:
                        rr_raw = int.from_bytes(rr_bytes, byteorder='little', signed=False)

                        if rr_raw > 0:
                            rr_interval_ms = (rr_raw / 1024.0) * 1000

                            if 300 <= rr_interval_ms <= 2000:
                                self.rr_intervals.append(rr_interval_ms)

                                if len(self.rr_intervals) <= 10:
                                    hr = 60000 / rr_interval_ms
                                    print(f"[RR {len(self.rr_intervals)}] {rr_interval_ms:.0f} ms (心率: {hr:.0f} bpm)")
                except Exception:
                    pass

                rr_index += 2

        except Exception as e:
            print(f"数据处理错误: {e}")

    async def start_monitoring(self):
        """开始监测"""
        if not self.client or not self.client.is_connected:
            print("设备未连接")
            return

        try:
            await self.client.start_notify(HEART_RATE_MEASUREMENT_UUID, self.data_handler)
            print("🎧 开始接收心率数据...")
            print("等待收集RR间期（需要50个）...")
            print("-" * 50)

            while True:
                await asyncio.sleep(1)

                current_time = asyncio.get_event_loop().time()
                if current_time - self.last_analysis_time >= self.analysis_interval:
                    rr_count = len(self.rr_intervals)

                    if rr_count >= 50:
                        print(f"\n📊 分析 {rr_count} 个RR间期...")
                        result = self.recognizer.predict_from_rr(self.rr_intervals)
                        if result:
                            print(f"🎯 情绪: {result['emotion']} (置信度: {result['confidence']:.1%})")
                            print(f"   兴奋: {result['probabilities']['兴奋']:.1%} | "
                                  f"中性: {result['probabilities']['中性']:.1%} | "
                                  f"压力: {result['probabilities']['压力']:.1%}")
                        else:
                            print("预测失败，继续收集...")
                    else:
                        print(f"\r收集RR间期: {rr_count}/50", end="")

                    self.last_analysis_time = current_time

        except KeyboardInterrupt:
            print("\n⏹️ 停止监测")
        finally:
            if self.client and self.client.is_connected:
                try:
                    await self.client.stop_notify(HEART_RATE_MEASUREMENT_UUID)
                except:
                    pass
                await self.client.disconnect()
                print("已断开连接")

    async def run(self):
        if not await self.connect():
            return
        await self.start_monitoring()


async def main():
    MODEL_PATH = "Model/Fusion_model_Attention_7Convlayer_1(92.556).h5"
    DEVICE_ADDRESS = "A0:9E:1A:E9:FD:44"  # 你的 Polar H10 地址

    print("=" * 50)
    print("Polar H10 情绪识别系统")
    print("=" * 50)

    app = PolarH10Emotion(MODEL_PATH, DEVICE_ADDRESS)
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())