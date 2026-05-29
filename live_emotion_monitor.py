"""
V8.1 最终版 - 完整调试信息 + 数据记录
用于后续采集更多数据后重新训练
"""
import asyncio
import time
import numpy as np
from collections import deque
from datetime import datetime
from bleak import BleakClient
import joblib
import json
import os

HEART_RATE_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


# ==================== 数据记录器 ====================

class DataLogger:
    """记录所有原始数据和分析结果，供后续训练使用"""

    def __init__(self, session_name=None):
        if session_name is None:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.session_name = session_name
        self.raw_rr = []  # 所有RR间期 + 时间戳
        self.predictions = []  # 所有预测结果
        self.baseline_info = {}  # 基线信息
        self.start_time = None

    def set_baseline(self, hr, rmssd, sdnn):
        self.baseline_info = {
            'baseline_hr': hr,
            'baseline_rmssd': rmssd,
            'baseline_sdnn': sdnn,
            'timestamp': datetime.now().isoformat()
        }

    def log_rr(self, rr_value, timestamp):
        self.raw_rr.append({
            'rr_ms': rr_value,
            'timestamp': timestamp,
            'elapsed': timestamp - self.start_time if self.start_time else 0
        })

    def log_prediction(self, result):
        self.predictions.append(result)

    def save(self):
        """保存所有数据到JSON文件"""
        filename = f"session_{self.session_name}.json"

        data = {
            'session_info': {
                'session_name': self.session_name,
                'date': datetime.now().isoformat(),
                'baseline': self.baseline_info,
                'total_rr': len(self.raw_rr),
                'total_predictions': len(self.predictions)
            },
            'raw_rr': self.raw_rr,
            'predictions': self.predictions
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 数据已保存: {filename}")
        print(f"   RR间期: {len(self.raw_rr)}个")
        print(f"   预测结果: {len(self.predictions)}个")
        return filename


# ==================== 情绪监测器 ====================

class LiveEmotionMonitor:
    """V8.1 - 差分特征 + 规则 + 完整调试 + 数据记录"""

    def __init__(self, model_path='my_emotion_model_v7_diff.pkl', session_name=None):
        print(f"\n{'=' * 70}")
        print(f"  🎯 实时情绪监测系统 V8.1 (调试版)")
        print(f"{'=' * 70}")

        # 加载模型
        print(f"\n📂 加载模型: {model_path}")
        self.model_pkg = joblib.load(model_path)
        print(f"   ✅ 模型加载成功")
        print(f"   特征数量: {len(self.model_pkg['feature_names'])}")
        print(f"   特征名: {self.model_pkg['feature_names']}")
        print(f"   情绪标签: {self.model_pkg['emotion_labels']}")

        # 检查模型类型
        model = self.model_pkg['model']
        print(f"   模型类型: {type(model).__name__}")
        if hasattr(model, 'n_estimators'):
            print(f"   树数量: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"   最大深度: {model.max_depth}")

        # RR缓冲区
        self.rr_buffer = deque(maxlen=300)
        self.rr_timestamps = deque(maxlen=300)

        # 基线
        self.baseline_rr = None
        self.baseline_hr = None
        self.baseline_rmssd = None
        self.baseline_sdnn = None

        # 状态
        self.start_time = None
        self.current_emotion = None
        self.emotion_history = deque(maxlen=100)
        self.last_display_time = 0

        # 调试统计
        self.debug_stats = {
            'total_predictions': 0,
            'model_predictions': {'平静': 0, '兴奋': 0, '恐惧/压力': 0},
            'final_predictions': {'平静': 0, '兴奋': 0, '恐惧/压力': 0},
            'rule_corrections': 0,
            'avg_confidence': 0,
            'hr_range': [999, 0],
            'rmssd_range': [999, 0],
            'feature_values': [],
        }

        # 数据记录器
        self.logger = DataLogger(session_name)

    def hr_handler(self, sender, data):
        """解析H10的RR间期"""
        try:
            if len(data) >= 4:
                rr_index = 2
                current_time = time.time()

                while rr_index < len(data):
                    if rr_index + 1 >= len(data):
                        break
                    try:
                        rr_bytes = data[rr_index:rr_index + 2]
                        if len(rr_bytes) == 2:
                            rr_ms = int.from_bytes(rr_bytes, byteorder='little') / 1024.0 * 1000
                            if 300 <= rr_ms <= 2000:
                                self.rr_buffer.append(rr_ms)
                                self.rr_timestamps.append(current_time)

                                # 记录原始数据
                                self.logger.log_rr(rr_ms, current_time)
                    except:
                        pass
                    rr_index += 2
        except:
            pass

    def _extract_features(self, rr_window):
        """提取特征（与训练脚本完全一致）"""
        rr = np.array(rr_window)

        if len(rr) < 10:
            return None, None, None, None, None

        # 绝对特征
        hr = 60000 / np.mean(rr)
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
        sdnn = np.std(rr)
        rr_range = np.max(rr) - np.min(rr)
        cv_rr = sdnn / np.mean(rr) * 100
        pnn20 = np.sum(np.abs(np.diff(rr)) > 20) / len(rr) * 100

        features = [hr, rmssd, sdnn, rr_range, cv_rr, pnn20]

        # 差分特征
        if self.baseline_rr is not None and len(self.baseline_rr) >= 10:
            baseline_hr = 60000 / np.mean(self.baseline_rr)
            baseline_rmssd = np.sqrt(np.mean(np.diff(self.baseline_rr) ** 2))
            baseline_sdnn = np.std(self.baseline_rr)
            baseline_cv = baseline_sdnn / np.mean(self.baseline_rr) * 100

            features.extend([
                hr - baseline_hr,
                (hr - baseline_hr) / baseline_hr * 100 if baseline_hr > 0 else 0,
                rmssd - baseline_rmssd,
                (rmssd - baseline_rmssd) / baseline_rmssd * 100 if baseline_rmssd > 0 else 0,
                sdnn - baseline_sdnn,
                cv_rr - baseline_cv
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0])

        return np.array(features).reshape(1, -1), hr, rmssd, sdnn, cv_rr

    def _predict(self, rr_window):
        """预测情绪（模型 + 规则）"""
        features, current_hr, rmssd, sdnn, cv_rr = self._extract_features(rr_window)
        if features is None:
            return None

        # 更新统计
        self.debug_stats['hr_range'][0] = min(self.debug_stats['hr_range'][0], current_hr)
        self.debug_stats['hr_range'][1] = max(self.debug_stats['hr_range'][1], current_hr)
        self.debug_stats['rmssd_range'][0] = min(self.debug_stats['rmssd_range'][0], rmssd)
        self.debug_stats['rmssd_range'][1] = max(self.debug_stats['rmssd_range'][1], rmssd)
        self.debug_stats['feature_values'].append(features[0].tolist())

        # 模型预测
        features_imputed = self.model_pkg['imputer'].transform(features)
        features_scaled = self.model_pkg['scaler'].transform(features_imputed)

        model_pred = self.model_pkg['model'].predict(features_scaled)[0]
        model_proba = self.model_pkg['model'].predict_proba(features_scaled)[0]

        emotion_names = self.model_pkg['emotion_labels']
        model_emotion = emotion_names[model_pred]

        self.debug_stats['model_predictions'][model_emotion] += 1

        # 规则修正
        proba = model_proba.copy()
        rule_applied = None

        if self.baseline_hr:
            hr_change = current_hr - self.baseline_hr
            hr_change_pct = hr_change / self.baseline_hr * 100

            # 规则1：心率显著升高 → 偏向兴奋
            if hr_change > 10 or hr_change_pct > 15:
                old_proba = proba.copy()
                proba[1] += proba[2] * 0.5
                proba[2] *= 0.5
                if proba[1] < 0.3:
                    proba[1] += 0.2
                    proba[0] = max(0, proba[0] - 0.1)
                    proba[2] = max(0, proba[2] - 0.1)
                proba = proba / proba.sum()
                rule_applied = f"心率↑{hr_change:+.0f}bpm→偏向兴奋"

            # 规则2：心率接近基线 → 偏向平静
            elif abs(hr_change) < 3:
                old_proba = proba.copy()
                proba[0] += 0.15
                proba[1] = max(0, proba[1] - 0.08)
                proba[2] = max(0, proba[2] - 0.07)
                proba = proba / proba.sum()
                rule_applied = f"心率稳定→偏向平静"

            # 规则3：心率低于基线 → 偏向放松
            elif hr_change < -5:
                old_proba = proba.copy()
                proba[0] += 0.15
                proba[1] = max(0, proba[1] - 0.1)
                proba[2] = max(0, proba[2] - 0.05)
                proba = proba / proba.sum()
                rule_applied = f"心率↓{hr_change:+.0f}bpm→偏向放松"

        # 最终预测
        final_pred = np.argmax(proba)
        final_emotion = emotion_names[final_pred]
        confidence = proba[final_pred]

        self.debug_stats['final_predictions'][final_emotion] += 1
        self.debug_stats['total_predictions'] += 1
        if rule_applied:
            self.debug_stats['rule_corrections'] += 1

        # 计算平均置信度
        n = self.debug_stats['total_predictions']
        old_avg = self.debug_stats['avg_confidence']
        self.debug_stats['avg_confidence'] = old_avg + (confidence - old_avg) / n

        result = {
            'emotion': final_emotion,
            'confidence': confidence,
            'proba': {
                emotion_names[0]: float(proba[0]),
                emotion_names[1]: float(proba[1]),
                emotion_names[2]: float(proba[2])
            },
            'model_emotion': model_emotion,
            'model_proba': {
                emotion_names[0]: float(model_proba[0]),
                emotion_names[1]: float(model_proba[1]),
                emotion_names[2]: float(model_proba[2])
            },
            'current_hr': current_hr,
            'baseline_hr': self.baseline_hr,
            'hr_change': current_hr - self.baseline_hr if self.baseline_hr else 0,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'cv_rr': cv_rr,
            'rule_applied': rule_applied,
            'rr_count': len(rr_window),
        }

        # 记录预测
        self.logger.log_prediction(result)

        return result

    def _display(self, result):
        """完整调试显示"""
        emotion = result['emotion']
        confidence = result['confidence']
        proba = result['proba']

        emoji_map = {'平静': '😌', '兴奋': '😃', '恐惧/压力': '😰'}
        emoji = emoji_map.get(emotion, '❓')

        elapsed_str = ""
        if 'elapsed' in result:
            elapsed = result['elapsed']
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            elapsed_str = f"{mins}:{secs:02d}"

        hr = result['current_hr']
        hr_change = result['hr_change']
        model_emotion = result['model_emotion']
        rule = result['rule_applied']

        # 第1行：主显示
        if hr_change > 3:
            hr_str = f"HR:{hr:.0f}(↑{hr_change:+.0f})"
        elif hr_change < -3:
            hr_str = f"HR:{hr:.0f}(↓{hr_change:+.0f})"
        else:
            hr_str = f"HR:{hr:.0f}(→)"

        correction = f"[模型:{model_emotion}]" if model_emotion != emotion else ""
        rule_str = f"规则:{rule}" if rule else ""

        print(f"\r{emoji} {emotion:6s} | {elapsed_str} | "
              f"置信度:{confidence:.1%} | {hr_str} | "
              f"{correction} {rule_str}", end='', flush=True)

        # 第2行：概率详情
        print(f"\n  ├─ 模型概率: 平静{result['model_proba']['平静']:.1%} | "
              f"兴奋{result['model_proba']['兴奋']:.1%} | "
              f"压力{result['model_proba']['恐惧/压力']:.1%}")

        # 第3行：生理指标
        print(f"  ├─ 最终概率: 平静{proba['平静']:.1%} | "
              f"兴奋{proba['兴奋']:.1%} | "
              f"压力{proba['恐惧/压力']:.1%}")

        print(f"  └─ RMSSD:{result['rmssd']:.0f}ms | "
              f"SDNN:{result['sdnn']:.0f}ms | "
              f"CV:{result['cv_rr']:.1f}% | "
              f"RR数:{result['rr_count']}")

    async def start_monitoring(self, device_address, duration_minutes=None):
        """开始监测"""
        print(f"\n{'=' * 70}")
        print(f"  📡 连接设备: {device_address}")
        print(f"{'=' * 70}")

        print(f"\n📋 规则说明:")
        print(f"   心率↑ >10bpm 或 >15% → 偏向兴奋")
        print(f"   心率≈基线 ±3bpm     → 偏向平静")
        print(f"   心率↓ >5bpm          → 偏向放松")
        print(f"   其他                  → 保持模型预测")

        try:
            async with BleakClient(device_address, timeout=30.0) as client:
                print(f"\n✅ 已连接 Polar H10")
                await client.start_notify(HEART_RATE_UUID, self.hr_handler)

                # 基线校准
                print(f"\n⏳ 静息基线采集（60秒）...")
                print(f"   请保持静息状态，正常呼吸")

                for i in range(60, 0, -1):
                    rr_count = len(self.rr_buffer)
                    print(f"\r   {i:2d}秒 | RR间期: {rr_count}个", end='', flush=True)
                    await asyncio.sleep(1)

                if len(self.rr_buffer) >= 30:
                    self.baseline_rr = np.array(list(self.rr_buffer))[-30:]
                    self.baseline_hr = 60000 / np.mean(self.baseline_rr)
                    self.baseline_rmssd = np.sqrt(np.mean(np.diff(self.baseline_rr) ** 2))
                    self.baseline_sdnn = np.std(self.baseline_rr)

                    print(f"\n\n📊 基线建立完成:")
                    print(f"   基线心率:  {self.baseline_hr:.0f} bpm")
                    print(f"   基线RMSSD: {self.baseline_rmssd:.0f} ms")
                    print(f"   基线SDNN:  {self.baseline_sdnn:.0f} ms")

                    self.logger.set_baseline(self.baseline_hr, self.baseline_rmssd, self.baseline_sdnn)

                    if self.baseline_hr > 85:
                        print(f"   ⚠️ 基线心率偏高(>85bpm)，建议休息后重试")
                    elif self.baseline_hr > 75:
                        print(f"   ⚠️ 基线心率略高(>75bpm)，可能在紧张状态")
                    else:
                        print(f"   ✅ 基线心率正常")
                else:
                    print(f"\n⚠️ 基线数据不足，继续监测...")

                self.start_time = time.time()
                self.logger.start_time = self.start_time

                print(f"\n{'=' * 70}")
                print(f"  💓 开始实时监测")
                print(f"  按 Ctrl+C 停止")
                print(f"{'=' * 70}\n")

                last_prediction = 0
                last_status = 0
                prediction_count = 0

                try:
                    while True:
                        if duration_minutes:
                            elapsed_min = (time.time() - self.start_time) / 60
                            if elapsed_min >= duration_minutes:
                                print(f"\n\n⏰ 监测时长已到 ({duration_minutes}分钟)")
                                break

                        current_time = time.time()

                        # 每3秒预测一次
                        if (current_time - last_prediction >= 3 and
                                len(self.rr_buffer) >= 30):

                            rr_window = list(self.rr_buffer)[-30:]
                            result = self._predict(rr_window)

                            if result:
                                result['elapsed'] = current_time - self.start_time
                                self.current_emotion = result
                                self.emotion_history.append(result)
                                last_prediction = current_time
                                prediction_count += 1

                                # 每2秒显示一次
                                if current_time - self.last_display_time >= 2:
                                    self._display(result)
                                    self.last_display_time = current_time

                        # 每30秒状态摘要
                        elapsed = int(time.time() - self.start_time)
                        if elapsed // 30 > last_status:
                            last_status = elapsed // 30
                            mins = elapsed // 60
                            secs = elapsed % 60

                            if self.current_emotion:
                                recent_emotions = [e['emotion'] for e in
                                                   list(self.emotion_history)[-10:]]
                                from collections import Counter
                                counts = Counter(recent_emotions)
                                dominant = counts.most_common(1)[0][0]

                                print(f"\n[{mins}:{secs:02d}] 状态: {dominant} | "
                                      f"预测{len(self.emotion_history)}次 | "
                                      f"RR{len(self.rr_buffer)}个")
                            else:
                                print(f"\n[{mins}:{secs:02d}] 初始化中... | "
                                      f"RR{len(self.rr_buffer)}个")

                        await asyncio.sleep(0.1)

                except KeyboardInterrupt:
                    print(f"\n\n⏹️ 用户停止监测")

                # 停止H10
                await client.stop_notify(HEART_RATE_UUID)

                # 显示总结
                self._show_debug_summary()

                # 保存数据
                self.logger.save()

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    def _show_debug_summary(self):
        """显示完整调试总结"""
        stats = self.debug_stats
        n = stats['total_predictions']

        if n == 0:
            print(f"\n⚠️ 无预测数据")
            return

        print(f"\n{'=' * 70}")
        print(f"  📊 调试总结")
        print(f"{'=' * 70}")

        # 时间
        duration = time.time() - self.start_time if self.start_time else 0
        print(f"\n⏱️  监测时长: {duration:.0f}秒 ({duration / 60:.1f}分钟)")
        print(f"📊 总预测次数: {n}")
        print(f"📊 规则修正次数: {stats['rule_corrections']} ({stats['rule_corrections'] / n * 100:.0f}%)")

        # 模型预测分布
        print(f"\n🤖 模型原始预测:")
        for emotion in ['平静', '兴奋', '恐惧/压力']:
            count = stats['model_predictions'][emotion]
            pct = count / n * 100 if n > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"   {emotion:8s}: {bar} {pct:.0f}% ({count})")

        # 最终预测分布
        print(f"\n✅ 最终预测分布:")
        for emotion in ['平静', '兴奋', '恐惧/压力']:
            count = stats['final_predictions'][emotion]
            pct = count / n * 100 if n > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"   {emotion:8s}: {bar} {pct:.0f}% ({count})")

        # 生理指标
        print(f"\n💓 生理指标:")
        print(f"   心率范围: {stats['hr_range'][0]:.0f} - {stats['hr_range'][1]:.0f} bpm")
        print(f"   RMSSD范围: {stats['rmssd_range'][0]:.0f} - {stats['rmssd_range'][1]:.0f} ms")
        print(f"   基线心率: {self.baseline_hr:.0f} bpm" if self.baseline_hr else "   基线: 未建立")
        print(f"   平均置信度: {stats['avg_confidence']:.1%}")

        # 情绪变化时间线
        if len(self.emotion_history) >= 3:
            print(f"\n📈 情绪变化时间线:")
            emotions = list(self.emotion_history)
            for i in range(0, len(emotions), max(1, len(emotions) // 20)):
                e = emotions[i]
                elapsed = e.get('elapsed', 0)
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                hr = e.get('current_hr', 0)
                emotion = e['emotion']
                emoji = {'平静': '😌', '兴奋': '😃', '恐惧/压力': '😰'}[emotion]
                print(f"   [{mins:2d}:{secs:02d}] {emoji} {emotion:6s} HR:{hr:.0f}")

        # 特征统计
        if stats['feature_values']:
            features_array = np.array(stats['feature_values'])
            print(f"\n🔬 特征统计 (均值±标准差):")
            feature_names = self.model_pkg['feature_names']
            for i, name in enumerate(feature_names):
                if i < features_array.shape[1]:
                    mean = np.mean(features_array[:, i])
                    std = np.std(features_array[:, i])
                    print(f"   {name:18s}: {mean:8.2f} ± {std:8.2f}")

        print(f"\n{'=' * 70}")


# ==================== 入口 ====================

async def scan_devices():
    from bleak import BleakScanner
    print("🔍 扫描蓝牙设备...")
    devices = await BleakScanner.discover()
    print(f"\n找到 {len(devices)} 个设备:")
    for i, d in enumerate(devices, 1):
        if d.name:
            print(f"  {i}. {d.name}: {d.address}")
    return devices


if __name__ == "__main__":
    import argparse

    DEFAULT_DEVICE = "A0:9E:1A:E9:FD:44"

    parser = argparse.ArgumentParser(description='实时情绪监测 V8.1')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                        help=f'Polar H10 MAC地址 (默认: {DEFAULT_DEVICE})')
    parser.add_argument('--duration', type=int, default=None,
                        help='监测时长（分钟）')
    parser.add_argument('--model', type=str, default='my_emotion_model_v7_diff.pkl',
                        help='模型文件路径')
    parser.add_argument('--session', type=str, default=None,
                        help='会话名称（用于保存数据）')
    parser.add_argument('--scan', action='store_true',
                        help='扫描可用设备')

    args = parser.parse_args()

    if args.scan:
        asyncio.run(scan_devices())
    else:
        print(f"\n📡 设备: {args.device}")
        print(f"📂 模型: {args.model}")

        monitor = LiveEmotionMonitor(args.model, args.session)
        asyncio.run(monitor.start_monitoring(args.device, args.duration))