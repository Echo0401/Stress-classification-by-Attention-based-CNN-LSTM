# record_ecg_data_simple.py
import asyncio
import numpy as np
from bleak import BleakClient, BleakScanner
from collections import deque
import time
import pickle
from datetime import datetime
import os
import traceback

ECG_UUID = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL_UUID = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"


class ECGRecorder:
    def __init__(self, device_address=None):
        self.device_address = device_address
        self.client = None
        self.ecg_buffer = deque(maxlen=20000)
        self.recording = False
        self.start_time = None
        self.prediction_count = 0

    def parse_ecg_packet(self, data):
        samples = []
        for i in range(10, len(data), 3):
            if i + 3 <= len(data):
                sample = int.from_bytes(data[i:i + 3], byteorder='little', signed=True)
                samples.append(sample)
        return samples

    def ecg_callback(self, sender, data):
        if self.recording:
            samples = self.parse_ecg_packet(data)
            self.ecg_buffer.extend(samples)
            duration = len(self.ecg_buffer) / 130
            print(f"\r📊 录制中: {duration:.1f}秒, {len(self.ecg_buffer)}个点", end="", flush=True)

    async def scan_device(self, timeout=5):
        print(f"扫描设备（{timeout}秒）...")
        devices = await BleakScanner.discover(timeout=timeout)
        for device in devices:
            if device.name and "Polar" in device.name:
                print(f"✅ 找到: {device.name} ({device.address})")
                return device
        return None

    async def connect_and_record(self, duration=60, label="unknown"):
        """直接使用主程序已验证的连接方式"""

        if not self.device_address:
            device = await self.scan_device()
            if not device:
                print("❌ 未找到设备")
                return None
            self.device_address = device.address

        print(f"\n连接到 {self.device_address}...")
        self.client = BleakClient(self.device_address, timeout=15.0)

        try:
            # 第一次连接
            await self.client.connect()
            print("✅ 第一次连接成功")

            # 配对
            print("正在配对...")
            try:
                await self.client.pair(protection_level=2)
                print("✅ 配对完成")
            except Exception as e:
                print(f"配对信息: {e}")

            # 断开重连（与主程序完全一致）
            await self.client.disconnect()
            await asyncio.sleep(2.0)
            await self.client.connect()
            print("✅ 重新连接成功")

            # 激活ECG
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
                return None

            await asyncio.sleep(1.0)

            # 开始录制
            self.recording = True
            self.ecg_buffer.clear()
            await self.client.start_notify(ECG_UUID, self.ecg_callback)

            print(f"\n🎤 开始录制 [{label}]，时长 {duration} 秒...")
            print("   按 Ctrl+C 可提前停止\n")

            # 等待指定时长
            for i in range(duration):
                await asyncio.sleep(1)
                if not self.recording:
                    break
                # 每10秒显示一次
                if (i + 1) % 10 == 0:
                    print(f"\n   ⏱️ 已录制 {i + 1}/{duration} 秒")

            # 停止录制
            self.recording = False
            await self.client.stop_notify(ECG_UUID)

            # 保存数据
            ecg_data = np.array(list(self.ecg_buffer))

            # 创建保存目录
            save_dir = "recorded_ecg_data"
            os.makedirs(save_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/ecg_{label}_{timestamp}.pkl"

            # 保存数据
            with open(filename, 'wb') as f:
                pickle.dump({
                    'label': label,
                    'duration': duration,
                    'sampling_rate': 130,
                    'ecg_signal': ecg_data,
                    'timestamp': timestamp,
                    'sample_count': len(ecg_data)
                }, f)

            print(f"\n\n✅ 数据已保存: {filename}")
            print(f"   采样点数: {len(ecg_data)}")
            print(f"   实际时长: {len(ecg_data) / 130:.1f}秒")

            return filename

        except KeyboardInterrupt:
            print("\n\n⏹️ 用户中断录制")
            self.recording = False
            if self.client and self.client.is_connected:
                try:
                    await self.client.stop_notify(ECG_UUID)
                except:
                    pass
            return None
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            traceback.print_exc()
            return None
        finally:
            if self.client and self.client.is_connected:
                try:
                    await self.client.disconnect()
                    print("✅ 已断开连接")
                except:
                    pass


async def main():
    print("=" * 60)
    print("   Polar H10 ECG 数据录制工具")
    print("=" * 60)
    print()
    print("请按顺序录制不同状态的数据:")
    print()

    # Polar H10 的MAC地址（你的设备地址）
    device_address = "A0:9E:1A:E9:FD:44"

    recorder = ECGRecorder(device_address)

    # 录制顺序
    recordings = [
        ("neutral", 60, "😐 中性状态 - 请放松，看平静视频或闭眼休息"),
        ("excitement", 60, "😆 兴奋状态 - 请观看搞笑/刺激/振奋的视频"),
        ("stress", 60, "😰 压力状态 - 请观看紧张/恐怖/压力视频"),
    ]

    results = []

    for label, duration, instruction in recordings:
        print("\n" + "=" * 60)
        print(f"下一步: {instruction}")
        print("=" * 60)
        input("准备好后按回车开始录制...")

        filename = await recorder.connect_and_record(duration=duration, label=label)

        if filename:
            results.append(filename)
            print(f"\n✅ {label} 状态录制完成！")
        else:
            print(f"\n❌ {label} 状态录制失败")

        # 录制间隔
        if label != recordings[-1][0]:
            print("\n请休息一下，准备下一个状态...")
            await asyncio.sleep(3)

    print("\n" + "=" * 60)
    print("录制完成！")
    print("=" * 60)
    print(f"共录制 {len(results)} 个文件:")
    for f in results:
        print(f"  - {f}")
    print("\n现在可以运行 test_recorded_data.py 来测试模型")


if __name__ == "__main__":
    asyncio.run(main())