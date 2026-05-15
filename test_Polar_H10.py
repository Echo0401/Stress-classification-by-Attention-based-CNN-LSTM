# test_scan.py
import asyncio
from bleak import BleakScanner

async def scan():
    print("扫描中...")
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"{device.name} - {device.address}")
    print("扫描完成")

asyncio.run(scan())