import json
import os
import glob

# 找到所有JSON文件
json_files = glob.glob("h10_session_*.json")
print(f"找到 {len(json_files)} 个JSON文件\n")

all_segments = []
emotion_counts = {}

for f in json_files:
    with open(f, 'r') as fp:
        data = json.load(fp)
    segments = data['segments']
    all_segments.extend(segments)

    # 统计这个文件里的情绪
    file_counts = {}
    for seg in segments:
        label = seg['emotion_label']
        file_counts[label] = file_counts.get(label, 0) + 1
        emotion_counts[label] = emotion_counts.get(label, 0) + 1

    print(f"📁 {f}: {len(segments)}段")
    for label, count in file_counts.items():
        print(f"     {label}: {count}段")

print(f"\n{'=' * 50}")
print(f"📊 总计: {len(all_segments)}段")
for label, count in emotion_counts.items():
    print(f"   {label}: {count}段")

# 检查是否达标
print(f"\n{'=' * 50}")
print(f"达标检查:")
targets = {"兴奋": 15, "恐惧/压力": 15, "平静": 5}
for label, target in targets.items():
    count = emotion_counts.get(label, 0)
    status = "✅" if count >= target else f"❌ 还差{target - count}段"
    print(f"  {label}: {count}/{target} {status}")