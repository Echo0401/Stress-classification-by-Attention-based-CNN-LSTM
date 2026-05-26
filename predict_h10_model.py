"""
用 Dreamer HRV 模型预测 H10 V4 采集的情绪数据
适配新的数据结构：emotion_label, video_name, time_start/time_end
"""
import json
import joblib
import numpy as np
from dreamer_hrv_extraction import calculate_hrv_features_25, FEATURE_NAMES_25

# 1. 加载模型
model_pkg = joblib.load('dreamer_hrv_model.pkl')
model = model_pkg['model']
scaler = model_pkg['scaler']
label_map = model_pkg['label_map']

print(f"模型标签映射: {label_map}")

# 2. 加载你的V4数据
data_file = "h10_session_20260522_121327.json"  # 改成你刚才保存的文件名
with open(data_file, 'r') as f:
    data = json.load(f)

segments = data['segments']


# 3. 定义情绪映射（方便对比）
def get_true_emotion(valence, arousal):
    """根据效价和唤醒度判断真实情绪"""
    if valence >= 4 and arousal >= 4:
        return "兴奋"
    elif valence <= 2 and arousal >= 4:
        return "恐惧/压力"
    elif abs(valence - 3) <= 1 and arousal <= 2:
        return "平静"
    else:
        return "中性/混合"


# 4. 逐段预测
print("=" * 70)
print("Dreamer HRV 模型预测结果")
print("=" * 70)

results = []
correct = 0
total = 0

for i, seg in enumerate(segments):
    # 跳过基线
    if seg.get('emotion_label') == '平静':
        print(f"\n📊 基线片段: 静息数据（跳过预测）")
        print(f"   RR间期数量: {seg['rr_count']}")
        continue

    rr = np.array(seg['rr_intervals'])

    # 检查数据是否足够
    if len(rr) < 20:
        print(f"\n⚠️  片段 #{seg.get('segment_id', i + 1)}: RR数据不足 ({len(rr)}个)，跳过")
        continue

    # 提取HRV特征
    features = calculate_hrv_features_25(rr)

    if features is None:
        print(f"\n⚠️  片段 #{seg.get('segment_id', i + 1)}: 特征提取失败，跳过")
        continue

    # 转为特征向量并标准化
    X = np.array([[features[name] for name in FEATURE_NAMES_25]])
    X_scaled = scaler.transform(X)

    # 预测
    pred = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]

    # 获取真实标签
    true_valence = seg['valence']
    true_arousal = seg['arousal']
    true_emotion = seg['emotion_label']

    # 输出详细信息
    print(f"\n{'─' * 70}")
    print(f"📍 片段 #{seg.get('segment_id', i + 1)}")
    print(f"   视频: {seg.get('video_name', '未知')}")
    print(f"   时间: {seg['time_start']:.0f}s - {seg['time_end']:.0f}s")
    print(f"   时长: {seg['duration']:.0f}秒")
    print(f"   RR数据: {seg['rr_count']}个")

    print(f"\n   📝 你的标注:")
    print(f"      情绪: {true_emotion}")
    print(f"      效价: {true_valence}, 唤醒度: {true_arousal}")

    print(f"\n   🤖 模型预测:")
    print(f"      情绪: {label_map[pred]}")
    print(f"      概率分布:")
    for j, prob in enumerate(probs):
        bar = "█" * int(prob * 20)
        print(f"        {label_map[j]:12s}: {prob:.1%} {bar}")

    # 判断对错
    is_correct = (label_map[pred] == true_emotion)
    if is_correct:
        print(f"\n   ✅ 预测正确！")
        correct += 1
    else:
        print(f"\n   ❌ 预测错误（真实:{true_emotion}, 预测:{label_map[pred]}）")
        # 分析可能原因
        if true_emotion == "兴奋" and "恐惧/压力" in label_map[pred]:
            print(f"   💡 可能原因: 模型难以区分高效价+高唤醒 vs 低效价+高唤醒")
        elif true_emotion == "恐惧/压力" and "兴奋" in label_map[pred]:
            print(f"   💡 可能原因: 生理信号相似（心率都升高），但效价方向相反")

    total += 1

    # 保存结果用于后续分析
    results.append({
        'segment_id': seg.get('segment_id', i + 1),
        'video': seg.get('video_name', '未知'),
        'true_emotion': true_emotion,
        'predicted_emotion': label_map[pred],
        'true_valence': true_valence,
        'true_arousal': true_arousal,
        'probabilities': {label_map[j]: prob for j, prob in enumerate(probs)},
        'is_correct': is_correct
    })

# 5. 总体统计
print(f"\n{'=' * 70}")
print(f"📊 总体统计")
print(f"{'=' * 70}")
print(f"总片段数: {total}")
print(f"预测正确: {correct}")
print(f"预测错误: {total - correct}")
if total > 0:
    print(f"准确率: {correct / total:.1%}")

# 按情绪分类统计
from collections import defaultdict

emotion_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
for r in results:
    emotion_stats[r['true_emotion']]['total'] += 1
    if r['is_correct']:
        emotion_stats[r['true_emotion']]['correct'] += 1

print(f"\n各情绪准确率:")
for emotion, stats in emotion_stats.items():
    if stats['total'] > 0:
        acc = stats['correct'] / stats['total']
        print(f"  {emotion}: {stats['correct']}/{stats['total']} ({acc:.1%})")

# 6. 混淆矩阵（简化版）
print(f"\n📊 混淆矩阵:")
# 获取所有出现的情绪标签
all_emotions = sorted(list(set([r['true_emotion'] for r in results] +
                               [r['predicted_emotion'] for r in results])))
print(f"{'真实预测':<12}", end="")
for e in all_emotions:
    print(f"{e:<12}", end="")
print()

for true_e in all_emotions:
    print(f"{true_e:<12}", end="")
    for pred_e in all_emotions:
        count = sum(1 for r in results if r['true_emotion'] == true_e and r['predicted_emotion'] == pred_e)
        print(f"{count:<12}", end="")
    print()

print(f"\n💡 关键发现:")
# 分析哪些情绪容易混淆
if total > 0:
    confusion_pairs = []
    for r in results:
        if not r['is_correct']:
            confusion_pairs.append(f"{r['true_emotion']} → {r['predicted_emotion']}")

    if confusion_pairs:
        from collections import Counter

        pair_counts = Counter(confusion_pairs)
        print(f"   最容易混淆的组合:")
        for pair, count in pair_counts.most_common(3):
            print(f"     {pair}: {count}次")
    else:
        print(f"   所有预测正确！但先别高兴太早...")
        print(f"   请检查:")
        print(f"   1. 数据量是否足够（每种情绪至少10段）")
        print(f"   2. 标签分布是否均衡")
        print(f"   3. 是否存在过拟合（训练集和测试集是否重叠）")