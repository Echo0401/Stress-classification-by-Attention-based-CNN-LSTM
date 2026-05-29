"""
V8.1 - 加入先验规则：心率显著高于基线时优先判兴奋
"""
import asyncio
import time
import numpy as np
from collections import deque
from bleak import BleakClient
import joblib

HEART_RATE_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


class LiveEmotionMonitorV8:
    """V8.1 - 差分特征 + 心率先验规则"""

    def __init__(self, model_path='my_emotion_model_v7_diff.pkl'):
        print(f"📂 加载模型: {model_path}")
        self.model_pkg = joblib.load(model_path)
        print(f"✅ 模型加载成功")

        self.rr_buffer = deque(maxlen=180)
        self.baseline_rr = None
        self.baseline_hr = None

        self.emotion_history = []
        self.start_time = None
        self.last_display_time = 0
        self.current_emotion = None

    def hr_handler(self, sender, data):
        try:
            if len(data) >= 4:
                rr_index = 2
                while rr_index < len(data):
                    if rr_index + 1 >= len(data):
                        break
                    try:
                        rr_bytes = data[rr_index:rr_index + 2]
                        if len(rr_bytes) == 2:
                            rr_ms = int.from_bytes(rr_bytes, byteorder='little') / 1024.0 * 1000
                            if 300 <= rr_ms <= 2000:
                                self.rr_buffer.append(rr_ms)
                    except:
                        pass
                    rr_index += 2
        except:
            pass

    def _extract_features(self, rr_current):
        rr = np.array(rr_current)
        if len(rr) < 10:
            return None

        hr = 60000 / np.mean(rr)
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
        sdnn = np.std(rr)
        rr_range = np.max(rr) - np.min(rr)
        cv_rr = sdnn / np.mean(rr) * 100
        pnn20 = np.sum(np.abs(np.diff(rr)) > 20) / len(rr) * 100

        features = [hr, rmssd, sdnn, rr_range, cv_rr, pnn20]

        if self.baseline_rr is not None and len(self.baseline_rr) >= 10:
            baseline_hr = 60000 / np.mean(self.baseline_rr)
            baseline_rmssd = np.sqrt(np.mean(np.diff(self.baseline_rr) ** 2))
            baseline_sdnn = np.std(self.baseline_rr)
            baseline_cv = baseline_sdnn / np.mean(self.baseline_rr) * 100

            features.extend([
                hr - baseline_hr,
                (hr - baseline_hr) / baseline_hr * 100,
                rmssd - baseline_rmssd,
                (rmssd - baseline_rmssd) / baseline_rmssd * 100 if baseline_rmssd > 0 else 0,
                sdnn - baseline_sdnn,
                cv_rr - baseline_cv
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0])

        return np.array(features).reshape(1, -1), hr

    def _predict(self, rr_window):
        features, current_hr = self._extract_features(rr_window)
        if features is None:
            return None

        # 处理NaN
        features = self.model_pkg['imputer'].transform(features)
        features = self.model_pkg['scaler'].transform(features)

        # 模型预测
        pred = self.model_pkg['model'].predict(features)[0]
        proba = self.model_pkg['model'].predict_proba(features)[0]

        emotion_names = self.model_pkg['emotion_labels']

        # === 先验规则修正 ===
        if self.baseline_hr:
            hr_change = current_hr - self.baseline_hr
            hr_change_pct = hr_change / self.baseline_hr * 100

            # 规则1：心率显著高于基线（>10bpm或>15%）→ 偏向兴奋
            if hr_change > 10 or hr_change_pct > 15:
                # 把"恐惧/压力"的概率部分转移给"兴奋"
                proba[1] += proba[2] * 0.5  # 转移50%的压力概率
                proba[2] *= 0.5

                # 如果兴奋概率不够高，适当提升
                if proba[1] < 0.3:
                    proba[1] += 0.2
                    proba[0] = max(0, proba[0] - 0.1)
                    proba[2] = max(0, proba[2] - 0.1)

            # 规则2：心率接近基线（±3bpm）→ 偏向平静
            elif abs(hr_change) < 3 and proba[0] > 0.2:
                proba[0] += 0.1
                proba[1] = max(0, proba[1] - 0.05)
                proba[2] = max(0, proba[2] - 0.05)

            # 规则3：心率低于基线较多 → 可能是放松
            elif hr_change < -5:
                proba[0] += 0.15
                proba[1] = max(0, proba[1] - 0.1)
                proba[2] = max(0, proba[2] - 0.05)

            # 归一化概率
            proba = proba / proba.sum()

        # 取最大概率的类别
        pred = np.argmax(proba)
        emotion = emotion_names[pred]
        confidence = proba[pred]

        return {
            'emotion': emotion,
            'confidence': confidence,
            'proba': {
                emotion_names[0]: proba[0],
                emotion_names[1]: proba[1],
                emotion_names[2]: proba[2]
            },
            'current_hr': current_hr,
            'baseline_hr': self.baseline_hr,
            'model_pred': emotion_names[self.model_pkg['model'].predict(features)[0]]  # 原始预测
        }

    def _display(self, result):
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
            elapsed_str = f"{mins}:{secs:02d} | "

        hr = result.get('current_hr', 0)
        baseline = result.get('baseline_hr', 0)
        model_pred = result.get('model_pred', '?')

        # 如果规则修正了模型预测，显示修正信息
        correction = f"(模型判:{model_pred})" if model_pred != emotion else ""

        print(f"\r{emoji} {emotion:6s} | {elapsed_str}"
              f"置信度:{confidence:.1%} | HR:{hr:.0f}(基线{baseline:.0f}) {correction}",
              end='', flush=True)

        print(f"\n  ├─ 平静:{proba.get('平静', 0):.1%} | "
              f"兴奋:{proba.get('兴奋', 0):.1%} | "
              f"压力:{proba.get('恐惧/压力', 0):.1%}")

    async def start_monitoring(self, device_address, duration_minutes=None):
        print(f"\n{'=' * 60}")
        print(f"🎯 实时情绪监测 V8.1 - 差分特征+先验规则")
        print(f"{'=' * 60}\n")

        print(f"📋 先验规则:")
        print(f"   心率↑ >10bpm → 偏向兴奋")
        print(f"   心率≈基线 ±3bpm → 偏向平静")
        print(f"   心率↓ >5bpm → 偏向放松")
        print()

        try:
            async with BleakClient(device_address, timeout=30.0) as client:
                print(f"✅ 已连接 Polar H10")
                await client.start_notify(HEART_RATE_UUID, self.hr_handler)

                print(f"\n⏳ 静息基线（60秒）...")
                for i in range(60, 0, -1):
                    print(f"\r   剩余 {i} 秒 | RR: {len(self.rr_buffer)}个", end='', flush=True)
                    await asyncio.sleep(1)

                if len(self.rr_buffer) >= 30:
                    self.baseline_rr = np.array(list(self.rr_buffer))[-30:]
                    self.baseline_hr = 60000 / np.mean(self.baseline_rr)
                    print(f"\n✅ 基线: HR={self.baseline_hr:.0f}bpm")

                    if self.baseline_hr > 85:
                        print(f"⚠️ 基线偏高，建议休息后再试")

                self.start_time = time.time()
                print(f"\n💓 开始监测...\n")

                last_prediction = 0

                try:
                    while True:
                        if duration_minutes:
                            if (time.time() - self.start_time) / 60 >= duration_minutes:
                                break

                        current_time = time.time()

                        if current_time - last_prediction >= 3 and len(self.rr_buffer) >= 30:
                            rr_window = list(self.rr_buffer)[-30:]
                            result = self._predict(rr_window)

                            if result:
                                result['elapsed'] = current_time - self.start_time
                                self.current_emotion = result
                                self.emotion_history.append(result)
                                last_prediction = current_time

                                if current_time - self.last_display_time >= 2:
                                    self._display(result)
                                    self.last_display_time = current_time

                        await asyncio.sleep(0.1)

                except KeyboardInterrupt:
                    print(f"\n\n⏹️ 停止")

                await client.stop_notify(HEART_RATE_UUID)

                if self.emotion_history:
                    emotions = [e['emotion'] for e in self.emotion_history]
                    from collections import Counter
                    counts = Counter(emotions)
                    print(f"\n📊 总结: {dict(counts)}")

        except Exception as e:
            print(f"❌ {e}")
            import traceback
            traceback.print_exc()


async def scan_devices():
    from bleak import BleakScanner
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name:
            print(f"  {d.name}: {d.address}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="A0:9E:1A:E9:FD:44")
    parser.add_argument('--duration', type=int, default=None)
    parser.add_argument('--model', type=str, default='my_emotion_model_v7_diff.pkl')
    parser.add_argument('--scan', action='store_true')
    args = parser.parse_args()

    if args.scan:
        asyncio.run(scan_devices())
    else:
        monitor = LiveEmotionMonitorV8()
        asyncio.run(monitor.start_monitoring(args.device, args.duration))