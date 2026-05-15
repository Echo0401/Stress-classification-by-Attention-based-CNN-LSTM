import asyncio
import sys
from bleak import BleakScanner, BleakClient

ECG_UUID = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL_UUID = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"


def ecg_callback(sender, data):
    print(f"✅ 收到ECG数据 ({len(data)} bytes): {data.hex()[:60]}...")


async def activate_ecg(client):
    print("正在激活ECG数据流...")
    try:
        await client.write_gatt_char(
            PMD_CONTROL_UUID,
            bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00]),
            response=True
        )
        print("✅ ECG激活成功")
        return True
    except Exception as e:
        print(f"❌ ECG激活失败: {e}")
        return False


async def connect_and_stream(device):
    print(f"\n正在连接到 {device.name} ({device.address})...")

    client = BleakClient(device, timeout=15.0)

    try:
        # 第一次连接
        await client.connect()
        print(f"✅ 第一次连接成功: {client.is_connected}")

        # 尝试配对（如果已经配对过，这一步会很快完成）
        print("正在请求配对...")
        print("⚠️ 如果弹出系统配对窗口，请点击'允许'或'确认'")
        try:
            await client.pair(protection_level=2)
            print("✅ 配对请求完成")
        except Exception as e:
            error_msg = str(e)
            if "ALREADY" in error_msg.upper() or "Already" in error_msg:
                print("⚠️ 配对已在后台进行中，等待完成...")
                await asyncio.sleep(3)
            else:
                print(f"配对信息: {e}")

        # 关键：断开后重新连接，让配对加密生效
        print("断开连接，准备重新建立加密通道...")
        await client.disconnect()
        await asyncio.sleep(2.0)

        # 第二次连接（此时应该已配对）
        await client.connect()
        print(f"✅ 重新连接成功: {client.is_connected}")

        # 现在尝试激活ECG
        if not await activate_ecg(client):
            print("尝试用另一种方式激活...")
            # 备用激活指令
            try:
                await client.write_gatt_char(
                    PMD_CONTROL_UUID,
                    bytearray([0x01, 0x00]),
                    response=True
                )
                await asyncio.sleep(0.5)
                await client.write_gatt_char(
                    PMD_CONTROL_UUID,
                    bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00]),
                    response=True
                )
                print("✅ 备用激活方式成功")
            except Exception as e2:
                print(f"❌ 备用激活也失败: {e2}")
                await client.disconnect()
                return

        await asyncio.sleep(1.0)

        # 开始接收ECG数据
        print("开始接收ECG数据流...\n")
        await client.start_notify(ECG_UUID, ecg_callback)

        print("=" * 50)
        print("正在持续接收ECG数据，按 Ctrl+C 停止")
        print("=" * 50 + "\n")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n\n用户中断，正在停止...")

        await client.stop_notify(ECG_UUID)
        await client.disconnect()
        print("✅ 已停止ECG数据接收")

    except Exception as e:
        print(f"❌ 连接或通信出错: {e}")
        # 确保断开连接
        try:
            if client.is_connected:
                await client.disconnect()
        except:
            pass


async def scan_and_select():
    print("=" * 50)
    print("Polar H10 ECG 数据采集程序")
    print("=" * 50)
    print("\n请确保：")
    print("1. H10已经佩戴在胸前（电极湿润）")
    print("2. 电脑蓝牙已开启")
    print("3. 已从系统蓝牙设置中删除了旧的H10记录")
    print("\n正在扫描蓝牙设备（10秒）...")

    devices = await BleakScanner.discover(timeout=10.0)

    polar_devices = [d for d in devices if d.name and "Polar" in d.name]

    if not polar_devices:
        print("\n❌ 没有找到任何Polar设备！")
        return None

    print(f"\n✅ 找到 {len(polar_devices)} 个Polar设备：")
    for i, d in enumerate(polar_devices):
        print(f"  [{i}] {d.name} - {d.address}")

    if len(polar_devices) == 1:
        selected = polar_devices[0]
        print(f"\n自动选择: {selected.name}")
    else:
        while True:
            try:
                choice = input(f"\n请选择设备编号 (0-{len(polar_devices) - 1}): ").strip()
                idx = int(choice)
                if 0 <= idx < len(polar_devices):
                    selected = polar_devices[idx]
                    print(f"已选择: {selected.name}")
                    break
            except ValueError:
                pass

    return selected


async def main():
    try:
        device = await scan_and_select()
        if device:
            await connect_and_stream(device)
        else:
            print("\n程序退出，未找到可用设备")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        print("\n程序结束")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())