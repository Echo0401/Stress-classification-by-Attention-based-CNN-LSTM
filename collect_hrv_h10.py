"""
H10数据采集脚本 V4 - 自动筛选候选片段 + 人工确认
基于HRV变化自动检测可能的情绪反应点，减少回忆负担
"""
import asyncio
import time
import json
import numpy as np
from datetime import datetime
from bleak import BleakClient

HEART_RATE_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


class H10Recorder:
    def __init__(self):
        self.rr_intervals = []
        self.rr_timestamps = []
        self.session_data = []
        self.start_time = None

    def hr_handler(self, sender, data):
        """解析H10的RR间期，记录时间戳"""
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
                                self.rr_intervals.append(rr_ms)
                                self.rr_timestamps.append(current_time - self.start_time)
                    except:
                        pass
                    rr_index += 2
        except Exception as e:
            pass

    async def record_full_video(self, client, video_name, duration_minutes):
        """完整记录视频的RR数据"""
        duration_seconds = duration_minutes * 60
        self.rr_intervals = []
        self.rr_timestamps = []
        self.start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"🎬 开始观看: {video_name}")
        print(f"⏱️  视频时长: {duration_minutes}分钟")
        print(f"💡 全屏观看，数据后台自动记录")
        print(f"{'=' * 60}")

        start = time.time()
        last_progress = 0
        try:
            while time.time() - start < duration_seconds:
                elapsed = int(time.time() - start)
                if elapsed // 30 > last_progress:  # 每30秒提示
                    last_progress = elapsed // 30
                    mins = elapsed // 60
                    secs = elapsed % 60
                    print(f"⏱️  {mins}:{secs:02d} | RR: {len(self.rr_intervals)}个", end="\r")
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print(f"\n⏸️  用户中断")

        actual_duration = time.time() - start
        print(f"\n✅ 完成: {len(self.rr_intervals)}个RR间期，{actual_duration:.1f}秒")

        return {
            'video_name': video_name,
            'full_rr': self.rr_intervals.copy(),
            'full_timestamps': self.rr_timestamps.copy(),
            'duration_seconds': actual_duration
        }

    def auto_detect_emotion_segments(self, full_data, window_sec=30, step_sec=10):
        """
        自动检测可能的情绪变化片段
        基于心率变异性(HRV)的突变来识别

        window_sec: 分析窗口大小（秒）
        step_sec: 滑动步长（秒）
        """
        rr = np.array(full_data['full_rr'])
        ts = np.array(full_data['full_timestamps'])

        if len(rr) < 20:
            return []

        # 计算滑动窗口的HRV特征
        windows = []
        current_start = 0

        while current_start + window_sec <= ts[-1]:
            # 找到窗口内的RR数据
            mask = (ts >= current_start) & (ts < current_start + window_sec)
            window_rr = rr[mask]

            if len(window_rr) >= 10:
                # 计算多个HRV指标
                hr = 60000 / np.mean(window_rr)  # 平均心率
                rmssd = np.sqrt(np.mean(np.diff(window_rr) ** 2))  # 心率变异性
                sdnn = np.std(window_rr)  # 标准差

                windows.append({
                    'time_start': current_start,
                    'time_end': current_start + window_sec,
                    'mean_hr': hr,
                    'rmssd': rmssd,
                    'sdnn': sdnn,
                    'rr_count': len(window_rr)
                })

            current_start += step_sec

        if len(windows) < 3:
            return []

        # 找到HRV突变点（情绪可能变化的时刻）
        hrs = [w['mean_hr'] for w in windows]
        rmssds = [w['rmssd'] for w in windows]

        # 计算变化率
        hr_changes = np.abs(np.diff(hrs))
        rmssd_changes = np.abs(np.diff(rmssds))

        # 找到显著变化点（超过平均值+1个标准差）
        hr_threshold = np.mean(hr_changes) + np.std(hr_changes)
        rmssd_threshold = np.mean(rmssd_changes) + np.std(rmssd_changes)

        candidate_indices = []
        for i in range(len(hr_changes)):
            if hr_changes[i] > hr_threshold or rmssd_changes[i] > rmssd_threshold:
                candidate_indices.append(i + 1)  # +1因为diff后索引偏移

        # 合并相邻候选点
        if not candidate_indices:
            # 如果没有显著变化，选变化最大的3个点
            combined = hr_changes + rmssd_changes
            top_indices = np.argsort(combined)[-3:]
            candidate_indices = list(set([i + 1 for i in top_indices]))

        # 去重并排序
        candidate_indices = sorted(list(set(candidate_indices)))

        # 构建候选片段
        candidates = []
        for idx in candidate_indices:
            if idx < len(windows):
                w = windows[idx]
                candidates.append({
                    'time_start': max(0, w['time_start'] - 15),  # 往前扩展15秒
                    'time_end': min(ts[-1], w['time_end'] + 15),  # 往后扩展15秒
                    'mean_hr': w['mean_hr'],
                    'rmssd': w['rmssd'],
                    'hr_change': hr_changes[idx - 1] if idx > 0 else 0,
                    'rmssd_change': rmssd_changes[idx - 1] if idx > 0 else 0
                })

        return candidates[:8]  # 最多返回8个候选

    def review_candidates(self, full_data, candidates):
        """让用户确认自动检测的候选片段"""
        video_name = full_data['video_name']
        total_duration = full_data['duration_seconds']

        print(f"\n{'=' * 60}")
        print(f"🤖 自动检测到 {len(candidates)} 个可能的情绪变化点")
        print(f"   视频: {video_name} (总时长 {total_duration:.0f}秒)")
        print(f"{'=' * 60}")

        if not candidates:
            print(f"\n⚠️  未检测到明显变化，你可以手动标注")
            return self.manual_annotate(full_data)

        print(f"\n💡 这些时间点附近可能有情绪变化：")
        for i, c in enumerate(candidates, 1):
            start_m, start_s = divmod(c['time_start'], 60)
            end_m, end_s = divmod(c['time_end'], 60)
            print(f"   {i}. {int(start_m)}:{int(start_s):02d} - {int(end_m)}:{int(end_s):02d}")
            print(f"      心率: {c['mean_hr']:.0f} bpm, HRV变化: {c['rmssd_change']:.1f}ms")

        confirmed_segments = []

        print(f"\n现在逐个确认这些候选片段（按回车跳过，或标注情绪）")

        for i, candidate in enumerate(candidates, 1):
            print(f"\n{'─' * 40}")
            start_m, start_s = divmod(candidate['time_start'], 60)
            end_m, end_s = divmod(candidate['time_end'], 60)

            print(f"📍 候选片段 #{i}")
            print(f"   时间: {int(start_m)}:{int(start_s):02d} - {int(end_m)}:{int(end_s):02d}")
            print(f"   生理指标: 心率{candidate['mean_hr']:.0f}bpm, HRV突变{candidate['rmssd_change']:.1f}ms")

            print(f"\n   这个时间点你有情绪变化吗？")
            print(f"   1 = 兴奋 (开心+激动)")
            print(f"   2 = 恐惧/压力 (不适+紧张)")
            print(f"   3 = 其他情绪")
            print(f"   0 = 无感/跳过")

            choice = input(f"   选择 (0-3): ").strip()

            if choice == '0':
                continue

            if choice not in ['1', '2', '3']:
                print(f"   ⚠️  无效选择，跳过")
                continue

            # SAM评分
            try:
                print(f"\n   😊 效价 (愉快程度): 1=非常不愉快 ... 5=非常愉快")
                valence = int(input(f"   评分 (1-5): "))

                print(f"   ⚡ 唤醒度 (激动程度): 1=非常平静 ... 5=非常兴奋")
                arousal = int(input(f"   评分 (1-5): "))

                if not (1 <= valence <= 5 and 1 <= arousal <= 5):
                    print(f"   ⚠️  评分需在1-5之间，跳过")
                    continue

                # 提取该片段的RR数据
                segment_rr = []
                for rr, ts in zip(full_data['full_rr'], full_data['full_timestamps']):
                    if candidate['time_start'] <= ts <= candidate['time_end']:
                        segment_rr.append(rr)

                if len(segment_rr) < 10:
                    print(f"   ⚠️  数据太少，跳过")
                    continue

                # 情绪标签
                if choice == '1':
                    emotion_label = "兴奋"
                elif choice == '2':
                    emotion_label = "恐惧/压力"
                else:
                    emotion_label = "其他"

                segment = {
                    'video_name': video_name,
                    'detection_method': 'auto_hrv',
                    'emotion_label': emotion_label,
                    'time_start': candidate['time_start'],
                    'time_end': candidate['time_end'],
                    'duration': candidate['time_end'] - candidate['time_start'],
                    'valence': valence,
                    'arousal': arousal,
                    'rr_intervals': segment_rr,
                    'rr_count': len(segment_rr),
                    'mean_hr': candidate['mean_hr'],
                    'hrv_change': candidate['rmssd_change'],
                    'timestamp': datetime.now().isoformat()
                }

                confirmed_segments.append(segment)
                print(f"   ✅ 已确认: {emotion_label} (效价={valence}, 唤醒度={arousal})")

            except ValueError:
                print(f"   ⚠️  输入无效，跳过")
                continue

        # 询问是否还有遗漏
        print(f"\n{'─' * 40}")
        print(f"已确认 {len(confirmed_segments)} 个情绪片段")
        add_more = input(f"还有其他遗漏的情绪变化点吗？(y/n): ")
        if add_more.lower() == 'y':
            manual_segments = self.manual_annotate(full_data)
            confirmed_segments.extend(manual_segments)

        return confirmed_segments

    def manual_annotate(self, full_data):
        """手动标注补充"""
        video_name = full_data['video_name']
        total_duration = full_data['duration_seconds']

        print(f"\n📝 手动标注 - {video_name}")
        print(f"   视频时长: {total_duration:.0f}秒")

        segments = []
        while True:
            print(f"\n{'─' * 40}")
            print(f"添加情绪片段 (输入0退出)")

            try:
                start_sec = float(input(f"开始时间(秒): "))
                if start_sec == 0:
                    break
                end_sec = float(input(f"结束时间(秒): "))

                if start_sec < 0 or end_sec > total_duration or start_sec >= end_sec:
                    print(f"⚠️  时间范围无效")
                    continue

                print(f"情绪类型: 1=兴奋 2=恐惧/压力 3=其他")
                emotion_type = input(f"选择: ")

                valence = int(input(f"效价(1-5): "))
                arousal = int(input(f"唤醒度(1-5): "))

                segment_rr = []
                for rr, ts in zip(full_data['full_rr'], full_data['full_timestamps']):
                    if start_sec <= ts <= end_sec:
                        segment_rr.append(rr)

                if len(segment_rr) < 10:
                    print(f"⚠️  数据太少")
                    continue

                emotion_map = {'1': '兴奋', '2': '恐惧/压力', '3': '其他'}

                segments.append({
                    'video_name': video_name,
                    'detection_method': 'manual',
                    'emotion_label': emotion_map.get(emotion_type, '其他'),
                    'time_start': start_sec,
                    'time_end': end_sec,
                    'duration': end_sec - start_sec,
                    'valence': valence,
                    'arousal': arousal,
                    'rr_intervals': segment_rr,
                    'rr_count': len(segment_rr),
                    'timestamp': datetime.now().isoformat()
                })

                print(f"✅ 已添加")

            except ValueError:
                print(f"⚠️  输入无效")
                continue

        return segments


async def main():
    print(f"\n{'=' * 60}")
    print(f"🧠 H10情绪数据采集 V4 - 智能检测")
    print(f"   自动筛选 + 人工确认")
    print(f"{'=' * 60}")

    DEVICE_ADDRESS = "A0:9E:1A:E9:FD:44"

    recorder = H10Recorder()
    all_segments = []

    try:
        async with BleakClient(DEVICE_ADDRESS, timeout=30.0) as client:
            print(f"✅ 已连接 Polar H10")

            await client.start_notify(HEART_RATE_UUID, recorder.hr_handler)
            await asyncio.sleep(2)

            # 1. 静息基线（1分钟）
            print(f"\n📊 采集静息基线 (1分钟)")
            input(f"按回车开始，闭眼静坐...")
            baseline_data = await recorder.record_full_video(client, "静息基线", 1)

            # 2. 观看视频（5分钟）
            videos = [
                ("兴奋视频", 5),
                ("压力视频", 5),
            ]

            all_full_data = [baseline_data]

            for video_name, duration_min in videos:
                print(f"\n📹 下一个: {video_name} ({duration_min}分钟)")
                input(f"按回车开始观看...")

                full_data = await recorder.record_full_video(client, video_name, duration_min)
                all_full_data.append(full_data)

                print(f"\n☕ 休息2分钟...")
                await asyncio.sleep(120)

            await client.stop_notify(HEART_RATE_UUID)
            print(f"\n✅ 数据采集完成")

            # 3. 自动检测 + 人工确认
            print(f"\n{'=' * 60}")
            print(f"🤖 开始自动分析...")
            print(f"{'=' * 60}")

            for full_data in all_full_data:
                if full_data['video_name'] == "静息基线":
                    baseline_segment = {
                        'video_name': '静息基线',
                        'detection_method': 'baseline',
                        'emotion_label': '平静',
                        'time_start': 0,
                        'time_end': full_data['duration_seconds'],
                        'duration': full_data['duration_seconds'],
                        'valence': 3,
                        'arousal': 1,
                        'rr_intervals': full_data['full_rr'],
                        'rr_count': len(full_data['full_rr']),
                        'timestamp': datetime.now().isoformat()
                    }
                    all_segments.append(baseline_segment)
                    print(f"✅ 静息基线已标注")
                else:
                    # 自动检测候选片段
                    candidates = recorder.auto_detect_emotion_segments(full_data)

                    # 人工确认
                    segments = recorder.review_candidates(full_data, candidates)
                    all_segments.extend(segments)

            # 4. 保存
            filename = f"h10_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            output = {
                'session_info': {
                    'date': datetime.now().isoformat(),
                    'device': 'Polar H10',
                    'method': 'auto_detect + manual_confirm',
                    'total_segments': len(all_segments)
                },
                'segments': all_segments
            }

            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)

            # 5. 总结
            print(f"\n{'=' * 60}")
            print(f"📊 采集总结")
            print(f"{'=' * 60}")

            emotion_counts = {}
            for seg in all_segments:
                label = seg['emotion_label']
                emotion_counts[label] = emotion_counts.get(label, 0) + 1

            print(f"总片段: {len(all_segments)}")
            for label, count in emotion_counts.items():
                print(f"  {label}: {count}段")

            print(f"\n💾 已保存: {filename}")

    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())