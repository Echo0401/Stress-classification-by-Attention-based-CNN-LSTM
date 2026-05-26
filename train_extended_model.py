"""
使用扩展HRV特征训练情绪识别模型（方案B）
"""
import json
import glob
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from dreamer_hrv_extraction import calculate_extended_hrv_features, EXTENDED_FEATURE_NAMES

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("📂 加载数据（扩展特征）...")

json_files = glob.glob("h10_session_*.json")
all_features = []
all_labels = []
all_videos = []

for f in json_files:
    with open(f, 'r') as fp:
        data = json.load(fp)

    for seg in data['segments']:
        if seg['emotion_label'] not in ['兴奋', '恐惧/压力', '平静']:
            continue

        rr = np.array(seg['rr_intervals'])
        if len(rr) < 20:
            continue

        # 用扩展特征提取
        features = calculate_extended_hrv_features(rr)
        if features is None:
            continue

        feature_vector = []
        for name in EXTENDED_FEATURE_NAMES:
            val = features.get(name, np.nan)
            feature_vector.append(val)

        # 检查无效值
        if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
            continue

        all_features.append(feature_vector)
        all_labels.append(seg['emotion_label'])
        video_id = f"{f}_{seg.get('video_name', 'unknown')}"
        all_videos.append(video_id)

X = np.array(all_features)
y = np.array(all_labels)
groups = np.array(all_videos)

print(f"✅ 加载完成")
print(f"   特征矩阵: {X.shape} (样本数 × 特征数)")
print(f"   独立视频数: {len(np.unique(groups))}")
print(f"   标签分布:")
for label in ['兴奋', '恐惧/压力', '平静']:
    print(f"     {label}: {np.sum(y == label)}段")

# 处理NaN（用中位数填充）
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# ==================== 2. 标准化 ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== 3. Leave-One-Video-Out 交叉验证 ====================
print(f"\n{'=' * 60}")
print(f"🔬 按视频分组交叉验证（Leave-One-Video-Out）")
print(f"{'=' * 60}")

logo = LeaveOneGroupOut()

# 尝试多个模型
models = {
    'SVM_rbf': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    'SVM_linear': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
}

best_model = None
best_acc = 0
best_name = ""

for name, model in models.items():
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc = np.mean(y_true_all == y_pred_all)
    print(f"\n📊 {name}:")
    print(f"   准确率: {acc:.2%}")
    print(f"\n分类报告:")
    print(classification_report(y_true_all, y_pred_all,
                                target_names=['兴奋', '恐惧/压力', '平静']))

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# ==================== 4. 特征重要性分析 ====================
print(f"\n{'=' * 60}")
print(f"🔍 特征重要性分析（基于全部数据）")

rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_scaled, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\nTop 15 最重要的特征:")
for i in range(min(15, len(EXTENDED_FEATURE_NAMES))):
    idx = indices[i]
    print(f"  {i + 1:>2}. {EXTENDED_FEATURE_NAMES[idx]:<25s} {importances[idx]:.4f}")

# ==================== 5. 保存最终模型 ====================
print(f"\n{'=' * 60}")
print(f"💾 训练最终模型: {best_name} (LOVO准确率: {best_acc:.2%})")

final_model = models[best_name]
final_model.fit(X_scaled, y)

model_pkg = {
    'model': final_model,
    'scaler': scaler,
    'imputer': imputer,
    'label_map': {0: '兴奋', 1: '恐惧/压力', 2: '平静'},
    'feature_names': EXTENDED_FEATURE_NAMES,
    'training_samples': len(y),
    'lovo_accuracy': best_acc
}

joblib.dump(model_pkg, 'my_emotion_model_extended.pkl')
print(f"✅ 模型已保存: my_emotion_model_extended.pkl")

# ==================== 6. 混淆矩阵 ====================
print(f"\n📊 最终混淆矩阵 ({best_name}):")
y_pred_final = best_model.predict(X_scaled)
cm = confusion_matrix(y, y_pred_final, labels=['兴奋', '恐惧/压力', '平静'])
print(f"{'':>10} 兴奋  恐惧/压力  平静")
for i, label in enumerate(['兴奋', '恐惧/压力', '平静']):
    print(f"{label:>8}  {cm[i, 0]:>3}    {cm[i, 1]:>6}    {cm[i, 2]:>4}")



# # 第1步：计算相邻RR间期的差值
# rr_diff = [720-780, 850-720, 740-850, ...]  # 一阶差分
#
# # 第2步：计算差值的差值（二阶差分）
# rr_diff_diff = [130-(-60), -110-130, ...]  # 变化的加速度
#
# # 第3步：统计"突然剧烈变化"的比例
# sudden_change_index = 超过正常波动范围的突变次数 / 总变化次数