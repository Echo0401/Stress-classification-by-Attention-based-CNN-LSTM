#创建时间2026--5-15
# 采集Ploar H10 兴奋 压力 平和的数据
# 只负责记录RR间期和标注，不做实时预测：
# A0:9E:1A:E9:FD:44
"""
H10数据采集脚本 - 记录RR间期 + 情绪标注
"""
import asyncio
import time
import json
import numpy as np
from datetime import datetime
from bleak import BleakClient

# Polar H10 心率服务UUID
HEART_RATE_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


class H10Recorder:
    def __init__(self):
        self.rr_intervals = []
        self.session_data = []  # 存储每段视频的数据
        self.current_label = None

    def hr_handler(self, sender, data):
        """解析H10的RR间期数据"""
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
                                self.rr_intervals.append(rr_ms)
                    except:
                        pass
                    rr_index += 2
        except Exception as e:
            pass

    async def record_segment(self, client, segment_name, duration):
        """记录一段视频的RR间期"""
        self.rr_intervals = []
        print(f"\n🎬 开始记录: {segment_name} ({duration}秒)")
        print(f"   观看视频...")

        start = time.time()
        while time.time() - start < duration:
            await asyncio.sleep(1)
            elapsed = time.time() - start
            if int(elapsed) % 10 == 0:
                print(f"   [{int(elapsed)}/{duration}秒] RR数据: {len(self.rr_intervals)}个", end="\r")

        print(f"\n✅ 记录完成: {len(self.rr_intervals)}个RR间期")

        # 立即标注
        print(f"\n📝 请对刚才这段视频打分（SAM量表）:")
        valence = int(input("   效价 (1=非常不愉快, 5=非常愉快): "))
        arousal = int(input("   唤醒度 (1=非常平静, 5=非常兴奋): "))

        segment = {
            'name': segment_name,
            'timestamp': datetime.now().isoformat(),
            'rr_intervals': self.rr_intervals.copy(),
            'valence': valence,
            'arousal': arousal,
            'duration': duration,
            'rr_count': len(self.rr_intervals)
        }

        self.session_data.append(segment)
        return segment


async def main():
    # Polar H10的蓝牙地址（需要改成你自己的）
    DEVICE_ADDRESS = input("请输入Polar H10的蓝牙地址 (如 XX:XX:XX:XX:XX:XX): ").strip()

    recorder = H10Recorder()

    try:
        async with BleakClient(DEVICE_ADDRESS, timeout=30.0) as client:
            print(f"✅ 已连接 Polar H10")

            await client.start_notify(HEART_RATE_UUID, recorder.hr_handler)
            await asyncio.sleep(2)  # 等待数据流稳定

            # 采集基线
            input("\n📊 按回车开始采集静息基线 (30秒)，请保持静坐...")
            await recorder.record_segment(client, "baseline", 30)

            # 依次采集各情绪视频
            videos = [
                ("兴奋视频1", 180),  # 180秒=3分钟，可自行调整
                ("中性视频1", 180),
                ("压力视频1", 180),
                # 如果时间充裕，可以加更多
                # ("兴奋视频2", 180),
                # ("中性视频2", 180),
                # ("压力视频2", 180),
            ]

            for name, duration in videos:
                input(f"\n🎬 按回车开始录制 '{name}'...")
                await recorder.record_segment(client, name, duration)
                input(f"\n⏸️  休息30秒后按回车继续...")

            await client.stop_notify(HEART_RATE_UUID)

            # 保存数据
            filename = f"h10_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(recorder.session_data, f,
                          default=lambda x: x if not isinstance(x, np.ndarray) else x.tolist())

            print(f"\n💾 数据已保存到: {filename}")
            print(f"   总共 {len(recorder.session_data)} 段记录")

    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())