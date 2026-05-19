"""
用 Dreamer HRV 模型预测 H10 采集的情绪数据
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

# 2. 加载你的H10数据
data_file = "h10_session_20260515_173419.json"  # 改成你的文件名
with open(data_file, 'r') as f:
    segments = json.load(f)

# 3. 逐段预测
print("=" * 60)
print("预测结果")
print("=" * 60)

for seg in segments:
    if seg['name'] == 'baseline':
        continue  # 跳过基线

    rr = np.array(seg['rr_intervals'])
    features = calculate_hrv_features_25(rr)

    if features is None:
        print(f"{seg['name']}: RR数据不足，无法预测")
        continue

    # 转为特征向量
    X = np.array([[features[name] for name in FEATURE_NAMES_25]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]

    print(f"\n{seg['name']}:")
    print(f"  你的标注: 效价={seg['valence']}, 唤醒度={seg['arousal']}")
    print(f"  模型预测: {label_map[pred]}")
    for i, prob in enumerate(probs):
        print(f"    {label_map[i]}: {prob:.2%}")