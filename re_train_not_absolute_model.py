"""
重新训练脚本 V3 - 移除数据泄露特征
"""
import json
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("=" * 60)
print("🔄 差分特征重新训练 V3 - 防泄露")
print("=" * 60)

# 加载数据
json_files = glob.glob("h10_session_*.json")
all_segments = []

for f in json_files:
    with open(f, 'r') as fp:
        data = json.load(fp)
    video_name = f.replace('h10_session_', '').replace('.json', '')
    for seg in data.get('segments', []):
        seg['video'] = video_name
        all_segments.append(seg)

print(f"总片段: {len(all_segments)}")


# ==================== 差分特征提取（去除RR数量） ====================

def extract_diff_features(rr_current, rr_baseline=None):
    """提取差分特征 - 只用生理特征"""
    rr = np.array(rr_current)

    if len(rr) < 10:
        return None

    # 当前段生理特征
    hr = 60000 / np.mean(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    sdnn = np.std(rr)
    rr_range = np.max(rr) - np.min(rr)

    # 心率变异性衍生指标
    cv_rr = sdnn / np.mean(rr) * 100  # 变异系数
    pnn20 = np.sum(np.abs(np.diff(rr)) > 20) / len(rr) * 100  # pNN20

    features = [
        hr,  # 心率
        rmssd,  # RMSSD
        sdnn,  # SDNN
        rr_range,  # RR范围
        cv_rr,  # 变异系数
        pnn20,  # pNN20
    ]

    # 差分特征（相对基线）
    if rr_baseline is not None and len(rr_baseline) >= 10:
        baseline_rr = np.array(rr_baseline)
        baseline_hr = 60000 / np.mean(baseline_rr)
        baseline_rmssd = np.sqrt(np.mean(np.diff(baseline_rr) ** 2))
        baseline_sdnn = np.std(baseline_rr)
        baseline_cv = baseline_sdnn / np.mean(baseline_rr) * 100

        diff_features = [
            hr - baseline_hr,  # 心率变化
            (hr - baseline_hr) / baseline_hr * 100,  # 心率变化率%
            rmssd - baseline_rmssd,  # RMSSD变化
            (rmssd - baseline_rmssd) / baseline_rmssd * 100 if baseline_rmssd > 0 else 0,  # RMSSD变化率%
            sdnn - baseline_sdnn,  # SDNN变化
            cv_rr - baseline_cv,  # CV变化
        ]
        features.extend(diff_features)
    else:
        features.extend([0, 0, 0, 0, 0, 0])

    return features


# ==================== 构建训练数据 ====================

# 找到每个视频的平静基线
video_baselines = {}
for seg in all_segments:
    if seg.get('emotion_label') == '平静':
        video = seg.get('video', '')
        rr = seg.get('rr_intervals', [])
        if len(rr) >= 10:
            video_baselines[video] = rr

print(f"找到基线视频: {list(video_baselines.keys())}")

X = []
y_emotion = []
video_labels = []

for i, seg in enumerate(all_segments):
    label = seg.get('emotion_label', '')
    video = seg.get('video', '')
    rr = seg.get('rr_intervals', [])

    baseline_rr = video_baselines.get(video, None)

    features = extract_diff_features(rr, baseline_rr)
    if features is None:
        continue

    X.append(features)

    if label == '平静':
        y_emotion.append(0)
    elif label == '兴奋':
        y_emotion.append(1)
    elif label == '恐惧/压力':
        y_emotion.append(2)
    else:
        continue

    video_labels.append(video)

X = np.array(X)
y_emotion = np.array(y_emotion)
video_labels = np.array(video_labels)

print(f"\n训练样本: {len(X)}")
print(f"特征数量: {X.shape[1]}")
print(f"类别分布: 平静={sum(y_emotion == 0)}, 兴奋={sum(y_emotion == 1)}, 恐惧/压力={sum(y_emotion == 2)}")

# ==================== 按视频分组的交叉验证 ====================

# 处理NaN
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 特征名
feature_names = [
    'HR', 'RMSSD', 'SDNN', 'RR范围', 'CV_RR', 'pNN20',
    'HR变化', 'HR变化率%', 'RMSSD变化', 'RMSSD变化率%', 'SDNN变化', 'CV变化'
]

# 训练模型
print(f"\n📊 训练模型...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

# 按视频分组交叉验证（防止同视频数据泄露）
from sklearn.model_selection import LeaveOneGroupOut

unique_videos = np.unique(video_labels)
print(f"视频数: {len(unique_videos)}")

if len(unique_videos) >= 3:
    logo = LeaveOneGroupOut()
    scores = cross_val_score(model, X_scaled, y_emotion, groups=video_labels, cv=logo)
    print(f"\n📊 留一视频交叉验证:")
    print(f"   准确率: {scores.mean():.1%} (±{scores.std():.1%})")
    print(f"   各次分数: {[f'{s:.1%}' for s in scores]}")

model.fit(X_scaled, y_emotion)

# 特征重要性
print(f"\n📊 特征重要性:")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in indices:
    bar = '█' * int(importances[i] * 40)
    print(f"   {feature_names[i]:18s}: {bar} {importances[i]:.3f}")

# ==================== 详细评估 ====================

y_pred = model.predict(X_scaled)
print(f"\n📊 分类报告:")
print(classification_report(y_emotion, y_pred,
                            target_names=['平静', '兴奋', '恐惧/压力'],
                            zero_division=0))

print(f"混淆矩阵:")
cm = confusion_matrix(y_emotion, y_pred)
print(f"           预测: 平静  兴奋  压力")
for i, label in enumerate(['平静  ', '兴奋  ', '压力  ']):
    print(f"   真实 {label}: {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d}")

# ==================== 保存模型 ====================

model_pkg = {
    'model': model,
    'scaler': scaler,
    'imputer': imputer,
    'feature_names': feature_names,
    'emotion_labels': {0: '平静', 1: '兴奋', 2: '恐惧/压力'},
    'is_direct': True,
    'accuracy': scores.mean() if len(unique_videos) >= 3 else None,
}

joblib.dump(model_pkg, 'my_emotion_model_v7_diff.pkl')
print(f"\n✅ 模型已保存: my_emotion_model_v7_diff.pkl")

# ==================== 测试 ====================
print(f"\n🧪 留一视频预测详情:")
for video in unique_videos:
    train_mask = video_labels != video
    test_mask = video_labels == video

    if sum(test_mask) == 0:
        continue

    X_train, y_train = X_scaled[train_mask], y_emotion[train_mask]
    X_test, y_test = X_scaled[test_mask], y_emotion[test_mask]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)

    print(f"\n  测试视频: {video}")
    print(f"  训练: {sum(train_mask)}样本, 测试: {sum(test_mask)}样本")
    print(f"  准确率: {acc:.1%}")

    for i in range(len(y_test)):
        true_name = ['平静', '兴奋', '恐惧/压力'][y_test[i]]
        pred_name = ['平静', '兴奋', '恐惧/压力'][y_pred[i]]
        match = '✅' if y_test[i] == y_pred[i] else '❌'
        print(f"    {match} 真实={true_name}, 预测={pred_name}")