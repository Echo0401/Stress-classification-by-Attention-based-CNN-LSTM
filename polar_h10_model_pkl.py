"""
基于Dreamer模型，用你的H10数据微调
迁移学习：冻结底层 → 只训练分类器
"""
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from dreamer_hrv_extraction import calculate_hrv_features_25, FEATURE_NAMES_25

# 1. 加载原始Dreamer模型（作为特征提取参考）
dreamer_pkg = joblib.load('dreamer_hrv_model.pkl')
dreamer_scaler = dreamer_pkg['scaler']

# 2. 加载你的H10数据
data_file = "h10_session_20260519_180134.json"
with open(data_file, 'r') as f:
    data = json.load(f)

segments = data['segments']

# 3. 提取所有片段的特征和标签
X_list = []
y_list = []
segment_info = []

for seg in segments:
    # 跳过基线
    if seg.get('emotion_label') == '平静':
        continue

    rr = np.array(seg['rr_intervals'])
    if len(rr) < 20:
        continue

    # 提取HRV特征
    features = calculate_hrv_features_25(rr)
    if features is None:
        continue

    # 转为特征向量
    feature_vec = np.array([features[name] for name in FEATURE_NAMES_25])

    # 获取标签（用你的真实标注）
    true_emotion = seg['emotion_label']
    # 统一标签名4
    if true_emotion == '恐惧/压力':
        label = '压力'
    elif true_emotion == '兴奋':
        label = '兴奋'
    else:
        continue

    X_list.append(feature_vec)
    y_list.append(label)
    segment_info.append(seg)

X = np.array(X_list)
y = np.array(y_list)

print(f"总样本数: {len(X)}")
print(f"标签分布: {np.unique(y, return_counts=True)}")

# 4. 数据标准化（使用Dreamer的scaler作为基线，然后调整）
from sklearn.preprocessing import StandardScaler

# 创建新的scaler，基于你的数据
your_scaler = StandardScaler()
X_scaled = your_scaler.fit_transform(X)

print(f"\n特征分布对比:")
print(f"你的数据均值范围: {X_scaled.mean(axis=0).min():.2f} ~ {X_scaled.mean(axis=0).max():.2f}")
print(f"(理想范围应在 -0.5 ~ 0.5 之间)")

# 5. 用你自己的数据训练新模型
print(f"\n{'=' * 60}")
print(f"用你的H10数据训练个性化模型")
print(f"{'=' * 60}")

# 由于样本少（9个），用留一交叉验证评估
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

# 尝试多个模型
models = {
    'SVM (RBF)': SVC(kernel='rbf', probability=True, C=10, gamma='scale'),
    'SVM (Linear)': SVC(kernel='linear', probability=True, C=1),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
}

best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n{name}:")
    predictions = []
    true_labels = []

    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        predictions.append(pred)
        true_labels.append(y_test[0])

    # 计算准确率
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    accuracy = correct / len(predictions)

    print(f"  留一法准确率: {correct}/{len(predictions)} ({accuracy:.1%})")

    # 详细报告
    print(f"  分类报告:")
    print(classification_report(true_labels, predictions, zero_division=0))

    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_model_name = name

# 6. 训练最终模型（用全部数据）
print(f"\n{'=' * 60}")
print(f"训练最终模型: {best_model_name}")
print(f"{'=' * 60}")

final_model = models[best_model_name]
final_model.fit(X_scaled, y)

# 7. 保存你的个性化模型
your_model_pkg = {
    'model': final_model,
    'scaler': your_scaler,
    'label_map': {i: label for i, label in enumerate(np.unique(y))},
    'feature_names': FEATURE_NAMES_25,
    'training_samples': len(X),
    'label_distribution': {label: count for label, count in zip(*np.unique(y, return_counts=True))}
}

model_filename = f"h10_personal_model_{len(X)}samples.pkl"
joblib.dump(your_model_pkg, model_filename)
print(f"\n✅ 个性化模型已保存: {model_filename}")

# 8. 分析特征重要性（如果是随机森林）
if best_model_name == 'Random Forest':
    print(f"\n📊 最重要的10个HRV特征:")
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i + 1}. {FEATURE_NAMES_25[idx]}: {importances[idx]:.4f}")

# 9. 建议
print(f"\n{'=' * 60}")
print(f"📋 下一步建议")
print(f"{'=' * 60}")
print(f"当前样本数: {len(X)}")
print(f"最低要求: 每种情绪至少10段（当前每种只有4-5段）")
print(f"")
print(f"需要做的事:")
print(f"1. 再采集1-2次，累计每种情绪至少15段")
print(f"2. 合并所有采集数据，重新训练模型")
print(f"3. 用留一法验证准确率是否 >80%")
print(f"4. 采集新的测试集（不参与训练）验证泛化能力")
print(f"")
print(f"注意: 9个样本只能做初步验证，不能用于实际预测")
print(f"     留一法准确率高≠模型真的好，样本太少容易过拟合")