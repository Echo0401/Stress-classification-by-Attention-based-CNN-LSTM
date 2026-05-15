import asyncio
import websockets
import json
from Polar_H10_Model_ECG import PolarH10RealtimeEmotion

connected_clients = set()


async def broadcast(data):
    if not connected_clients:
        return
    message = json.dumps(data, ensure_ascii=False)
    for client in list(connected_clients):
        try:
            await client.send(message)
            print(f"   📤 已发送到前端")
        except Exception as e:
            connected_clients.remove(client)


async def handle_client(websocket):
    connected_clients.add(websocket)
    print(f"✅ 客户端已连接 ({len(connected_clients)})")
    try:
        await websocket.send(json.dumps({
            "type": "welcome",
            "message": "已连接到 Polar H10 实时情绪监测服务器"
        }))
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "heartbeat":
                    await websocket.send(json.dumps({"type": "heartbeat_ack"}))
            except:
                pass
    except Exception as e:
        print(f"客户端断开: {e}")
    finally:
        connected_clients.remove(websocket)


async def main():
    MODEL_PATH = "ECG_Model/Fusion_model_Attention_7Convlayer_FINAL_MODEL.h5"
    DEVICE_ADDRESS = "A0:9E:1A:E9:FD:44"

    # 创建 Polar 应用
    app = PolarH10RealtimeEmotion(MODEL_PATH, DEVICE_ADDRESS)

    # 设置广播回调
    async def on_prediction(data):
        await broadcast(data)

    app.on_prediction = on_prediction

    # 启动 WebSocket 服务器
    server = await websockets.serve(handle_client, "0.0.0.0", 8765)
    print("=" * 60)
    print("Polar H10 + WebSocket 服务启动")
    print("=" * 60)
    print("🚀 WebSocket: ws://localhost:8765")
    print()

    # 启动 Polar 连接
    await app.connect_and_stream()


if __name__ == "__main__":
    asyncio.run(main())