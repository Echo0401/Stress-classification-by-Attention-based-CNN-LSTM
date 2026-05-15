# test_recorded_data_fixed.py
import pickle
import numpy as np
import sys

sys.path.append('D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM')
import Model_selection as Ms
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
from scipy.signal import butter, filtfilt, find_peaks
import os


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
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'DotProductAttention': DotProductAttention},
            compile=False
        )
        print("✅ 模型直接加载成功")
        return model
    except Exception as e:
        print(f"直接加载失败: {e}")
        try:
            beat_length = 163
            rhythm_length = 1280
            model = Ms.fusion_model(beat_length, rhythm_length)
            model.load_weights(model_path)
            print("✅ 模型重建加载成功")
            return model
        except Exception as e2:
            raise Exception(f"模型加载失败: {e2}")


def get_r_peaks_polar_style(ecg_signal, sampling_rate=256):
    """使用与主程序相同的R峰检测算法"""
    try:
        # 与 Polar_H10_Model_ECG.py 中相同的算法
        diff1 = np.diff(ecg_signal)
        diff2 = np.diff(diff1)
        diff2 = np.pad(diff2, (0, len(ecg_signal) - len(diff2)), 'constant')

        # 调整阈值 - 降低阈值以便检测更多R峰
        threshold = np.std(diff2) * 1.5  # 原来是2.5，降到1.5
        min_distance = int(0.3 * sampling_rate)

        peaks, _ = find_peaks(
            np.abs(diff2),
            height=threshold,
            distance=min_distance
        )

        # 微调R峰位置
        refined_peaks = []
        search_window = int(0.05 * sampling_rate)

        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            local_peak = start + np.argmax(ecg_signal[start:end])
            refined_peaks.append(local_peak)

        return np.array(refined_peaks)
    except Exception as e:
        print(f"R峰检测失败: {e}")
        return np.array([])


def get_r_peaks_multiple_methods(ecg_signal, sampling_rate=256):
    """多种方法尝试R峰检测"""

    # 方法1: 原始差分法
    diff1 = np.diff(ecg_signal)
    diff2 = np.diff(diff1)
    diff2 = np.pad(diff2, (0, len(ecg_signal) - len(diff2)), 'constant')

    thresholds = [1.0, 1.2, 1.5, 2.0, 2.5]
    min_distance = int(0.3 * sampling_rate)

    best_peaks = []
    best_count = 0

    for thresh in thresholds:
        threshold = np.std(diff2) * thresh
        peaks, _ = find_peaks(np.abs(diff2), height=threshold, distance=min_distance)

        if len(peaks) > best_count:
            best_count = len(peaks)
            best_peaks = peaks

    # 微调
    refined_peaks = []
    search_window = int(0.05 * sampling_rate)

    for peak in best_peaks:
        start = max(0, peak - search_window)
        end = min(len(ecg_signal), peak + search_window)
        local_peak = start + np.argmax(ecg_signal[start:end])
        refined_peaks.append(local_peak)

    return np.array(refined_peaks)


def extract_beat_feature(ecg_signal, r_peaks, left=61, right=102):
    """提取beat特征"""
    if len(r_peaks) < 3:
        return None

    segments = []
    for i in range(1, min(len(r_peaks) - 1, 20)):  # 增加到20个心跳
        start = r_peaks[i] - left
        end = r_peaks[i] + right
        if start >= 0 and end < len(ecg_signal):
            segment = ecg_signal[start:end]
            if len(segment) >= 163:
                segment = segment[:163]
                segments.append(segment)

    if len(segments) < 2:
        return None

    return np.mean(segments, axis=0)


def extract_rhythm_feature(ecg_signal, r_peaks, window=1280, right=102):
    """提取rhythm特征"""
    if len(r_peaks) < 3:
        return None

    # 从第二个R峰后开始
    start = r_peaks[1] + right
    end = start + window

    if end <= len(ecg_signal):
        return ecg_signal[start:end]

    # 如果不够，从末尾取
    if len(ecg_signal) >= window:
        return ecg_signal[-window:]

    return None


def test_ecg_file(filepath, model):
    """测试单个ECG文件"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    ecg_signal = data['ecg_signal']
    label = data['label']

    print(f"\n{'=' * 50}")
    print(f"测试文件: {os.path.basename(filepath)}")
    print(f"标签: {label}")
    print(f"原始信号: {len(ecg_signal)}点, {len(ecg_signal) / 130:.1f}秒")

    # 先放大信号（Polar H10的信号幅度较小）
    ecg_signal = ecg_signal.astype(np.float32)

    # 方法1: 不做重采样，直接用130Hz检测R峰
    print("\n方法1: 原始130Hz信号检测R峰")
    r_peaks_130 = get_r_peaks_polar_style(ecg_signal, sampling_rate=130)
    print(f"  检测到R峰: {len(r_peaks_130)}个")

    if len(r_peaks_130) < 10:
        # 方法2: 降低阈值再试
        print("方法2: 降低阈值检测")
        r_peaks_130 = get_r_peaks_multiple_methods(ecg_signal, sampling_rate=130)
        print(f"  检测到R峰: {len(r_peaks_130)}个")

    if len(r_peaks_130) < 10:
        print("❌ R峰太少，信号质量差")
        # 打印信号统计信息
        print(f"   信号范围: [{ecg_signal.min():.1f}, {ecg_signal.max():.1f}]")
        print(f"   信号标准差: {ecg_signal.std():.1f}")
        return None

    # 计算心率
    rr_intervals = np.diff(r_peaks_130) / 130
    hr = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
    hrv = np.std(rr_intervals) * 1000 if len(rr_intervals) > 1 else 0
    print(f"心率: {hr:.0f} bpm, HRV: {hrv:.1f} ms")

    # 重采样到256Hz用于模型输入
    from scipy import signal
    target_rate = 256
    target_length = int(len(ecg_signal) * target_rate / 130)
    ecg_256hz = signal.resample(ecg_signal, target_length)

    # 在256Hz信号上重新检测R峰
    r_peaks_256 = np.array([int(p * 256 / 130) for p in r_peaks_130])

    # 提取特征
    beat = extract_beat_feature(ecg_256hz, r_peaks_256, left=61, right=102)
    rhythm = extract_rhythm_feature(ecg_256hz, r_peaks_256, window=1280, right=102)

    if beat is None or rhythm is None:
        print("❌ 特征提取失败")
        return None

    # 预测
    beat_input = beat.reshape(1, -1, 1)
    rhythm_input = rhythm.reshape(1, -1, 1)

    prediction = model.predict([beat_input, rhythm_input], verbose=0)[0]
    emotion_idx = np.argmax(prediction)

    emotions = {0: "兴奋", 1: "中性", 2: "压力"}

    print(f"\n预测结果:")
    print(f"  兴奋: {prediction[0]:.4f}")
    print(f"  中性: {prediction[1]:.4f}")
    print(f"  压力: {prediction[2]:.4f}")
    print(f"  → 预测情绪: {emotions[emotion_idx]}")
    print(f"  → 实际标签: {label}")

    # 判断是否正确
    label_to_idx = {"neutral": 1, "excitement": 0, "stress": 2}
    is_correct = (emotion_idx == label_to_idx.get(label, -1))
    print(f"  → 预测{'正确' if is_correct else '错误'}")

    return {
        'prediction': prediction,
        'label': label,
        'hr': hr,
        'hrv': hrv,
        'r_peaks_count': len(r_peaks_130),
        'correct': is_correct
    }


def main():
    MODEL_PATH = "ECG_Model/Fusion_model_Attention_7Convlayer_FINAL_MODEL.h5"

    print("=" * 60)
    print("加载模型...")
    print("=" * 60)
    model = load_model_compatible(MODEL_PATH)

    # 测试所有录制的数据
    save_dir = "recorded_ecg_data"

    if not os.path.exists(save_dir):
        print(f"❌ 目录 {save_dir} 不存在")
        return

    files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]

    if len(files) == 0:
        print("❌ 没有找到录制数据")
        return

    print(f"\n找到 {len(files)} 个数据文件")

    results = []
    for file in sorted(files):
        filepath = os.path.join(save_dir, file)
        result = test_ecg_file(filepath, model)
        if result:
            results.append(result)

    if len(results) == 0:
        print("\n❌ 没有成功处理的数据文件")
        return

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    correct_count = sum(1 for r in results if r['correct'])
    print(f"准确率: {correct_count}/{len(results)} = {correct_count / len(results) * 100:.1f}%")
    print()

    for r in results:
        pred_class = np.argmax(r['prediction'])
        pred_label = {0: "兴奋", 1: "中性", 2: "压力"}[pred_class]
        status = "✅" if r['correct'] else "❌"
        print(f"{status} 实际: {r['label']:10} -> 预测: {pred_label:10} "
              f"(兴奋:{r['prediction'][0]:.2f}, 中性:{r['prediction'][1]:.2f}, 压力:{r['prediction'][2]:.2f}) "
              f"R峰:{r['r_peaks_count']} 心率:{r['hr']:.0f} HRV:{r['hrv']:.0f}")


if __name__ == "__main__":
    main()