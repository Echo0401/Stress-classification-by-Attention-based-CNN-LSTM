"""
V6.3 最终完整版：使用真实数据模式 + 完整类定义
h10_session_20260519_180134.json
h10_session_20260522_121327.json
h10_session_20260522_123358.json

共43个片段：平静6个、兴奋16个、恐惧/压力21个、
第1次：用视频2+3训练 → 测试视频1
第2次：用视频1+3训练 → 测试视频2
第3次：用视频1+2训练 → 测试视频3
...
共9次（9个独立视频）
这样确保同一个视频的数据不会同时出现在训练集和测试集，防止数据泄露。
"""
import joblib
import numpy as np
from collections import deque
import time
import json
import glob
from dreamer_hrv_extraction import calculate_hrv_features_25, FEATURE_NAMES_25


# ==================== 特征计算函数 ====================

def _calculate_sample_entropy(rr, m=2, r_factor=0.2):
    N = len(rr)
    if N < m + 2 or np.std(rr) == 0:
        return np.nan
    r = r_factor * np.std(rr)
    if r == 0:
        return np.nan

    def _count_matches(template_len):
        if N - template_len < 2:
            return 0
        templates = np.array([rr[i:i + template_len] for i in range(N - template_len)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dist < r) - 1
        return max(count, 0)

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return -np.log(A / B)


def _calculate_dfa(rr, scales=None):
    if scales is None:
        scales = np.array([4, 8, 16])
    N = len(rr)
    if N < 20:
        return np.nan, np.nan
    rr_integrated = np.cumsum(rr - np.mean(rr))
    fluctuations, valid_scales = [], []

    for scale in scales:
        if scale < 4 or scale > N // 4:
            continue
        n_segments = N // scale
        if n_segments < 2:
            continue
        fluct_sum = 0
        for i in range(n_segments):
            segment = rr_integrated[i * scale:(i + 1) * scale]
            if len(segment) < 4:
                continue
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            fluct_sum += np.mean((segment - trend) ** 2)
        if n_segments > 0:
            fluctuations.append(np.sqrt(fluct_sum / n_segments))
            valid_scales.append(scale)

    if len(fluctuations) < 2:
        return np.nan, np.nan
    try:
        log_scales = np.log(valid_scales)
        log_fluct = np.log(fluctuations)
        alpha1, _ = np.polyfit(log_scales[:2], log_fluct[:2], 1)
        alpha2, _ = np.polyfit(log_scales[1:], log_fluct[1:], 1)
        return alpha1, alpha2
    except:
        return np.nan, np.nan


def _calculate_poincare_features(rr):
    if len(rr) < 3:
        return np.nan, np.nan, np.nan, np.nan
    rr_n, rr_n1 = rr[:-1], rr[1:]
    sd1 = np.std(rr_n1 - rr_n) / np.sqrt(2)
    sd2 = np.std(rr_n1 + rr_n) / np.sqrt(2)
    sd_ratio = sd1 / sd2 if sd2 != 0 else np.nan
    area = np.pi * sd1 * sd2 if not np.isnan(sd1) and not np.isnan(sd2) else np.nan
    return sd1, sd2, sd_ratio, area


def _calculate_arousal_features(rr):
    if len(rr) < 5:
        return np.nan, np.nan, np.nan
    rr_diff = np.diff(rr)
    accelerating = np.sum(rr_diff > 0) / len(rr_diff)
    if len(rr_diff) >= 3:
        sudden_changes = np.sum(np.abs(np.diff(rr_diff)) > np.std(rr_diff)) / (len(rr_diff) - 1)
    else:
        sudden_changes = np.nan
    if len(rr) >= 8:
        window = min(4, len(rr) // 2)
        rolling_std = np.array([np.std(rr[i:i + window]) for i in range(len(rr) - window + 1)])
        symp_index = np.mean(rolling_std) / np.mean(rr) if np.mean(rr) > 0 else np.nan
    else:
        symp_index = np.nan
    return accelerating, sudden_changes, symp_index


def calculate_optimized_hrv_features(rr):
    """计算优化后的HRV特征"""
    base_features = calculate_hrv_features_25(rr)
    if base_features is None:
        return None

    feature_list = [base_features[name] for name in FEATURE_NAMES_25]
    feature_list.append(_calculate_sample_entropy(rr))

    dfa_alpha1, dfa_alpha2 = _calculate_dfa(rr)
    feature_list.append(dfa_alpha1)
    feature_list.append(dfa_alpha2)

    sd1, sd2, sd_ratio, area = _calculate_poincare_features(rr)
    feature_list.extend([sd1, sd2, sd_ratio, area])

    acc, sudden, symp = _calculate_arousal_features(rr)
    feature_list.extend([acc, sudden, symp])

    if len(rr) >= 4:
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
        cv_rr = np.std(rr) / np.mean(rr) * 100 if np.mean(rr) > 0 else np.nan
        extreme_points = np.sum(np.abs(rr - np.mean(rr)) > 2 * np.std(rr)) / len(rr)
    else:
        rmssd = cv_rr = extreme_points = np.nan

    feature_list.extend([rmssd, cv_rr, extreme_points])
    return feature_list


def predict_emotion_hierarchical(rr_intervals, model_pkg):
    """层次化预测情绪"""
    features = calculate_optimized_hrv_features(rr_intervals)
    if features is None:
        return None

    features_array = np.array(features).reshape(1, -1)

    if np.any(np.isnan(features_array)):
        features_imputed = model_pkg['imputer'].transform(features_array)
    else:
        features_imputed = features_array

    features_scaled = model_pkg['scaler'].transform(features_imputed)

    arousal_proba = model_pkg['arousal_model'].predict_proba(features_scaled)[0]
    arousal_pred = model_pkg['arousal_model'].predict(features_scaled)[0]

    if arousal_pred == 0:
        return {
            'emotion': '平静',
            'confidence': float(arousal_proba[0]),
            'arousal': '低唤醒'
        }
    else:
        valence_proba = model_pkg['valence_model'].predict_proba(features_scaled)[0]
        valence_pred = model_pkg['valence_model'].predict(features_scaled)[0]

        return {
            'emotion': '兴奋' if valence_pred == 1 else '恐惧/压力',
            'confidence': float(valence_proba[valence_pred]),
            'arousal': '高唤醒',
            'valence_proba': {
                '恐惧/压力': float(valence_proba[0]),
                '兴奋': float(valence_proba[1])
            }
        }


class RealTimeEmotionMonitor:
    """实时情绪监测系统"""

    def __init__(self, model_path='my_emotion_model_v5_hierarchical.pkl'):
        print(f"📂 加载模型: {model_path}")
        self.model_pkg = joblib.load(model_path)
        print(f"✅ 模型加载成功")

        self.rr_buffer = deque(maxlen=60)
        self.emotion_history = deque(maxlen=20)
        self.last_prediction_time = 0
        self.prediction_count = 0

    def add_rr_interval(self, rr_value):
        """添加新的RR间期"""
        self.rr_buffer.append(rr_value)

    def predict_current_emotion(self, force=False):
        """预测当前情绪状态"""
        if len(self.rr_buffer) < 20:
            return None

        current_time = time.time()
        if not force and current_time - self.last_prediction_time < 2:
            return None

        rr_array = np.array(list(self.rr_buffer))
        result = predict_emotion_hierarchical(rr_array, self.model_pkg)

        if result:
            self.emotion_history.append(result)
            self.last_prediction_time = current_time
            self.prediction_count += 1

        return result

    def get_emotion_trend(self):
        """获取情绪趋势"""
        if len(self.emotion_history) < 3:
            return "数据不足"
        recent = list(self.emotion_history)[-10:]
        emotions = [r['emotion'] for r in recent]

        if emotions.count('恐惧/压力') >= 5:
            return "持续压力"
        elif emotions.count('兴奋') >= 5:
            return "持续兴奋"
        elif emotions.count('平静') >= 5:
            return "持续平静"
        else:
            return "情绪波动"

    def get_statistics(self):
        """获取统计信息"""
        if len(self.emotion_history) == 0:
            return None
        emotions = [r['emotion'] for r in self.emotion_history]
        confidences = [r['confidence'] for r in self.emotion_history]
        return {
            'total_predictions': len(emotions),
            'avg_confidence': np.mean(confidences),
            'emotion_distribution': {
                '平静': emotions.count('平静'),
                '兴奋': emotions.count('兴奋'),
                '恐惧/压力': emotions.count('恐惧/压力')
            }
        }


def extract_real_rr_patterns(json_files):
    """从真实数据中提取RR间期模式"""
    patterns = {'平静': [], '兴奋': [], '恐惧/压力': []}

    for f in json_files:
        with open(f, 'r') as fp:
            data = json.load(fp)

        for seg in data.get('segments', []):
            emotion = seg.get('emotion_label', '')
            if emotion in patterns:
                rr = np.array(seg.get('rr_intervals', []))
                if len(rr) >= 30:
                    patterns[emotion].append(rr[:30])

    return patterns


def generate_rr_from_pattern(patterns, emotion, length=30):
    """基于真实数据模式生成RR序列"""
    if emotion not in patterns or len(patterns[emotion]) == 0:
        return None

    base = patterns[emotion][np.random.randint(len(patterns[emotion]))]
    noise = np.random.normal(0, 5, len(base))
    generated = base + noise

    if len(generated) > length:
        return generated[:length].astype(int)
    else:
        repeated = np.tile(generated, length // len(generated) + 1)
        return repeated[:length].astype(int)


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 实时情绪监测系统 V6.3")
    print("=" * 60)

    monitor = RealTimeEmotionMonitor()

    # 加载真实数据模式
    print("\n📂 加载真实HRV数据模式...")
    json_files = glob.glob("h10_session_*.json")
    patterns = extract_real_rr_patterns(json_files)

    for emotion, samples in patterns.items():
        print(f"   {emotion}: {len(samples)}个样本")

    print(f"\n💡 使用真实数据模式进行演示...")
    print("-" * 60)

    np.random.seed(42)


    # 测试函数
    def test_emotion(monitor, patterns, emotion_name, label):
        print(f"\n📊 模拟{label}状态")
        rr_seq = generate_rr_from_pattern(patterns, emotion_name, length=30)

        predictions = []
        for i in range(30):
            if rr_seq is not None and i < len(rr_seq):
                monitor.add_rr_interval(rr_seq[i])
            else:
                monitor.add_rr_interval(800)

            if i >= 20 and i % 5 == 0:
                result = monitor.predict_current_emotion(force=True)
                if result:
                    predictions.append(result)
                    print(f"  [{i:2d}s] {result['emotion']:6s} | "
                          f"置信度: {result['confidence']:.1%} | "
                          f"{result['arousal']}")
        return predictions


    pred_calm = test_emotion(monitor, patterns, '平静', '平静')
    pred_excite = test_emotion(monitor, patterns, '兴奋', '兴奋')
    pred_stress = test_emotion(monitor, patterns, '恐惧/压力', '恐惧/压力')

    # 统计
    print(f"\n{'=' * 60}")
    print(f"📈 监测总结")
    print(f"{'=' * 60}")

    stats = monitor.get_statistics()
    if stats:
        print(f"总预测次数: {stats['total_predictions']}")
        print(f"平均置信度: {stats['avg_confidence']:.1%}")

    for scene_name, preds in [("平静场景", pred_calm), ("兴奋场景", pred_excite), ("压力场景", pred_stress)]:
        if preds:
            emotions = [p['emotion'] for p in preds]
            print(f"\n  {scene_name}:")
            print(f"    平静: {emotions.count('平静')}次")
            print(f"    兴奋: {emotions.count('兴奋')}次")
            print(f"    恐惧/压力: {emotions.count('恐惧/压力')}次")

    print(f"\n✅ 演示完成")
