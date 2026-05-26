"""
V5 最终修正版：层次化分类 - 修复SMOTE样本不一致问题
"""
import json
import glob
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter
from dreamer_hrv_extraction import calculate_hrv_features_25, FEATURE_NAMES_25


# ==================== 特征计算函数 ====================

def _calculate_sample_entropy(rr, m=2, r_factor=0.2):
    N = len(rr)
    if N < m + 2 or np.std(rr) == 0:
        return np.nan
    r = r_factor * np.std(rr)
    if r == 0:
        return np.nan

    def _count_matches(template_len):
        if N - template_len < 2:
            return 0
        templates = np.array([rr[i:i + template_len] for i in range(N - template_len)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dist < r) - 1
        return max(count, 0)

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return -np.log(A / B)


def _calculate_dfa(rr, scales=None):
    if scales is None:
        scales = np.array([4, 8, 16])
    N = len(rr)
    if N < 20:
        return np.nan, np.nan
    rr_integrated = np.cumsum(rr - np.mean(rr))
    fluctuations, valid_scales = [], []

    for scale in scales:
        if scale < 4 or scale > N // 4:
            continue
        n_segments = N // scale
        if n_segments < 2:
            continue
        fluct_sum = 0
        for i in range(n_segments):
            segment = rr_integrated[i * scale:(i + 1) * scale]
            if len(segment) < 4:
                continue
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            fluct_sum += np.mean((segment - trend) ** 2)
        if n_segments > 0:
            fluctuations.append(np.sqrt(fluct_sum / n_segments))
            valid_scales.append(scale)

    if len(fluctuations) < 2:
        return np.nan, np.nan
    try:
        log_scales = np.log(valid_scales)
        log_fluct = np.log(fluctuations)
        alpha1, _ = np.polyfit(log_scales[:2], log_fluct[:2], 1)
        alpha2, _ = np.polyfit(log_scales[1:], log_fluct[1:], 1)
        return alpha1, alpha2
    except:
        return np.nan, np.nan


def _calculate_poincare_features(rr):
    if len(rr) < 3:
        return np.nan, np.nan, np.nan, np.nan
    rr_n, rr_n1 = rr[:-1], rr[1:]
    sd1 = np.std(rr_n1 - rr_n) / np.sqrt(2)
    sd2 = np.std(rr_n1 + rr_n) / np.sqrt(2)
    sd_ratio = sd1 / sd2 if sd2 != 0 else np.nan
    area = np.pi * sd1 * sd2 if not np.isnan(sd1) and not np.isnan(sd2) else np.nan
    return sd1, sd2, sd_ratio, area


def _calculate_arousal_features(rr):
    if len(rr) < 5:
        return np.nan, np.nan, np.nan
    rr_diff = np.diff(rr)
    accelerating = np.sum(rr_diff > 0) / len(rr_diff)
    if len(rr_diff) >= 3:
        sudden_changes = np.sum(np.abs(np.diff(rr_diff)) > np.std(rr_diff)) / (len(rr_diff) - 1)
    else:
        sudden_changes = np.nan
    if len(rr) >= 8:
        window = min(4, len(rr) // 2)
        rolling_std = np.array([np.std(rr[i:i + window]) for i in range(len(rr) - window + 1)])
        symp_index = np.mean(rolling_std) / np.mean(rr) if np.mean(rr) > 0 else np.nan
    else:
        symp_index = np.nan
    return accelerating, sudden_changes, symp_index


def calculate_optimized_hrv_features(rr):
    base_features = calculate_hrv_features_25(rr)
    if base_features is None:
        return None

    feature_list = [base_features[name] for name in FEATURE_NAMES_25]
    feature_list.append(_calculate_sample_entropy(rr))

    dfa_alpha1, dfa_alpha2 = _calculate_dfa(rr)
    feature_list.append(dfa_alpha1)
    feature_list.append(dfa_alpha2)

    sd1, sd2, sd_ratio, area = _calculate_poincare_features(rr)
    feature_list.extend([sd1, sd2, sd_ratio, area])

    acc, sudden, symp = _calculate_arousal_features(rr)
    feature_list.extend([acc, sudden, symp])

    if len(rr) >= 4:
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
        cv_rr = np.std(rr) / np.mean(rr) * 100 if np.mean(rr) > 0 else np.nan
        extreme_points = np.sum(np.abs(rr - np.mean(rr)) > 2 * np.std(rr)) / len(rr)
    else:
        rmssd = cv_rr = extreme_points = np.nan

    feature_list.extend([rmssd, cv_rr, extreme_points])
    return feature_list


OPTIMIZED_FEATURE_NAMES = FEATURE_NAMES_25 + [
    'sample_entropy', 'dfa_alpha1', 'dfa_alpha2',
    'poincare_sd1', 'poincare_sd2', 'poincare_sd_ratio', 'poincare_area',
    'accelerating_index', 'sudden_change_index', 'sympathetic_index',
    'rmssd_raw', 'cv_rr', 'extreme_points_ratio'
]

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("📂 加载并验证数据...")

json_files = glob.glob("h10_session_*.json")
all_features, all_labels, all_videos = [], [], []
label_counter = Counter()

for f in json_files:
    print(f"   处理: {f}")
    try:
        with open(f, 'r') as fp:
            data = json.load(fp)
    except Exception as e:
        print(f"   ⚠️ 错误: {e}")
        continue

    for seg in data.get('segments', []):
        emotion = seg.get('emotion_label', '')
        if emotion not in ['兴奋', '恐惧/压力', '平静']:
            continue

        rr = np.array(seg.get('rr_intervals', []))
        if len(rr) < 20:
            continue

        feature_vector = calculate_optimized_hrv_features(rr)
        if feature_vector is None:
            continue

        label_counter[emotion] += 1
        all_features.append(feature_vector)
        all_labels.append(emotion)
        video_id = f"{f}_{seg.get('video_name', 'unknown')}"
        all_videos.append(video_id)

print(f"\n✅ 加载完成")
print(f"   总样本: {len(all_labels)}")
print(f"   原始标签分布: {dict(label_counter)}")

X = np.array(all_features)
y = np.array(all_labels)
groups = np.array(all_videos)

# ==================== 2. 预处理 ====================
print(f"\n🔧 预处理...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print(f"✅ 预处理完成")

# ==================== 3. 层次化分类 ====================
print(f"\n{'=' * 60}")
print(f"🎯 层次化分类策略")
print(f"{'=' * 60}")

logo = LeaveOneGroupOut()

# 存储所有折的结果
all_arousal_true, all_arousal_pred = [], []
all_valence_true, all_valence_pred = [], []

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups)):
    print(f"\n--- 第{fold_idx + 1}折 ---")

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"   训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"   训练集标签: {dict(Counter(y_train))}")

    if len(np.unique(y_train)) < 2:
        print(f"   ⚠️ 跳过：训练集类别不足")
        continue

    # === 层级1: 唤醒度分类 ===
    y_train_arousal = np.array([1 if label in ['兴奋', '恐惧/压力'] else 0 for label in y_train])
    y_test_arousal = np.array([1 if label in ['兴奋', '恐惧/压力'] else 0 for label in y_test])

    # 【关键修正】如果使用SMOTE，要同时更新X和y
    X_train_arousal_balanced = X_train.copy()
    y_train_arousal_balanced = y_train_arousal.copy()

    if len(np.unique(y_train_arousal)) == 2 and min(Counter(y_train_arousal).values()) >= 2:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(2, min(Counter(y_train_arousal).values()) - 1))
            X_train_arousal_balanced, y_train_arousal_balanced = smote.fit_resample(
                X_train, y_train_arousal
            )
            print(f"   唤醒度SMOTE: {len(X_train)} -> {len(X_train_arousal_balanced)}")
        except Exception as e:
            print(f"   ⚠️ 唤醒度SMOTE失败: {e}")

    # 训练唤醒度分类器
    arousal_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    arousal_clf.fit(X_train_arousal_balanced, y_train_arousal_balanced)
    arousal_pred = arousal_clf.predict(X_test)

    all_arousal_true.extend(y_test_arousal)
    all_arousal_pred.extend(arousal_pred)

    # === 层级2: 效价分类（仅高唤醒样本） ===
    high_arousal_train_mask = np.array([label in ['兴奋', '恐惧/压力'] for label in y_train])
    high_arousal_test_mask = np.array([label in ['兴奋', '恐惧/压力'] for label in y_test])

    X_train_valence = X_train[high_arousal_train_mask]
    y_train_valence = np.array([1 if label == '兴奋' else 0 for label in y_train[high_arousal_train_mask]])

    X_test_valence = X_test[high_arousal_test_mask]
    y_test_valence = np.array([1 if label == '兴奋' else 0 for label in y_test[high_arousal_test_mask]])

    print(
        f"   效价训练集: {len(X_train_valence)} (兴奋: {sum(y_train_valence)}, 恐惧: {len(y_train_valence) - sum(y_train_valence)})")
    print(
        f"   效价测试集: {len(X_test_valence)} (兴奋: {sum(y_test_valence)}, 恐惧: {len(y_test_valence) - sum(y_test_valence)})")

    if len(X_train_valence) > 0 and len(np.unique(y_train_valence)) == 2:
        # 【关键修正】SMOTE处理
        X_train_valence_balanced = X_train_valence.copy()
        y_train_valence_balanced = y_train_valence.copy()

        if min(Counter(y_train_valence).values()) >= 2:
            try:
                smote_valence = SMOTE(random_state=42, k_neighbors=min(2, min(Counter(y_train_valence).values()) - 1))
                X_train_valence_balanced, y_train_valence_balanced = smote_valence.fit_resample(
                    X_train_valence, y_train_valence
                )
                print(f"   效价SMOTE: {len(X_train_valence)} -> {len(X_train_valence_balanced)}")
            except Exception as e:
                print(f"   ⚠️ 效价SMOTE失败: {e}")

        # 训练效价分类器
        valence_clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                          class_weight='balanced', random_state=42)
        valence_clf.fit(X_train_valence_balanced, y_train_valence_balanced)

        if len(X_test_valence) > 0:
            valence_pred = valence_clf.predict(X_test_valence)
            all_valence_true.extend(y_test_valence)
            all_valence_pred.extend(valence_pred)
    else:
        print(f"   ⚠️ 效价分类器训练数据不足，跳过")

# ==================== 4. 评估层级1：唤醒度分类 ====================
print(f"\n{'=' * 60}")
print(f"📊 层级1: 唤醒度分类结果")
print(f"{'=' * 60}")

if len(all_arousal_true) > 0:
    arousal_acc = accuracy_score(all_arousal_true, all_arousal_pred)
    print(f"准确率: {arousal_acc:.2%}")
    print(f"\n分类报告:")
    print(classification_report(all_arousal_true, all_arousal_pred,
                                target_names=['低唤醒(平静)', '高唤醒(兴奋+恐惧)']))

    cm_arousal = confusion_matrix(all_arousal_true, all_arousal_pred)
    print(f"混淆矩阵:")
    print(f"           预测低唤醒  预测高唤醒")
    print(f"实际低唤醒      {cm_arousal[0, 0]:>5}        {cm_arousal[0, 1]:>5}")
    print(f"实际高唤醒      {cm_arousal[1, 0]:>5}        {cm_arousal[1, 1]:>5}")

# ==================== 5. 评估层级2：效价分类 ====================
print(f"\n{'=' * 60}")
print(f"📊 层级2: 效价分类结果（兴奋 vs 恐惧/压力）")
print(f"{'=' * 60}")

if len(all_valence_true) > 0:
    valence_acc = accuracy_score(all_valence_true, all_valence_pred)
    print(f"准确率: {valence_acc:.2%}")
    print(f"\n分类报告:")
    print(classification_report(all_valence_true, all_valence_pred,
                                target_names=['恐惧/压力', '兴奋']))

    cm_valence = confusion_matrix(all_valence_true, all_valence_pred)
    print(f"混淆矩阵:")
    print(f"           预测恐惧  预测兴奋")
    print(f"实际恐惧      {cm_valence[0, 0]:>5}      {cm_valence[0, 1]:>5}")
    print(f"实际兴奋      {cm_valence[1, 0]:>5}      {cm_valence[1, 1]:>5}")

    # 计算恐惧/压力的召回率
    if cm_valence[0, 0] + cm_valence[0, 1] > 0:
        fear_recall = cm_valence[0, 0] / (cm_valence[0, 0] + cm_valence[0, 1])
        print(f"\n🎯 '恐惧/压力'召回率: {fear_recall:.2%}")
else:
    print("⚠️ 没有足够的效价分类测试数据")

# ==================== 6. 特征重要性分析 ====================
print(f"\n{'=' * 60}")
print(f"📈 区分兴奋vs恐惧/压力的关键特征")
print(f"{'=' * 60}")

high_arousal_mask = np.array([label in ['兴奋', '恐惧/压力'] for label in y])
X_high = X_scaled[high_arousal_mask]
y_high = np.array([1 if label == '兴奋' else 0 for label in y[high_arousal_mask]])

if len(X_high) > 0 and len(np.unique(y_high)) == 2:
    rf_valence = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced')
    rf_valence.fit(X_high, y_high)

    importances = rf_valence.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n区分兴奋vs恐惧/压力的Top 10特征:")
    for i in range(10):
        idx = indices[i]
        print(f"   {i + 1:2d}. {OPTIMIZED_FEATURE_NAMES[idx]:30s} 重要性: {importances[idx]:.4f}")

# ==================== 7. 保存模型 ====================
print(f"\n{'=' * 60}")
print(f"💾 保存层次化模型...")

# 重新训练最终模型（使用全部数据）
y_arousal_final = np.array([1 if label in ['兴奋', '恐惧/压力'] else 0 for label in y])
arousal_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
arousal_model.fit(X_scaled, y_arousal_final)

valence_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                    class_weight='balanced', random_state=42)
valence_model.fit(X_high, y_high)

model_pkg = {
    'arousal_model': arousal_model,
    'valence_model': valence_model,
    'scaler': scaler,
    'imputer': imputer,
    'feature_names': OPTIMIZED_FEATURE_NAMES,
    'hierarchical': True,
    'arousal_labels': {0: '平静', 1: '高唤醒'},
    'valence_labels': {0: '恐惧/压力', 1: '兴奋'}
}

joblib.dump(model_pkg, 'my_emotion_model_v5_hierarchical.pkl')
print(f"✅ 模型已保存: my_emotion_model_v5_hierarchical.pkl")


# ==================== 8. 预测函数 ====================
def predict_emotion_hierarchical(rr_intervals, model_pkg):
    """层次化预测情绪"""
    features = calculate_optimized_hrv_features(rr_intervals)
    if features is None:
        return None

    features_array = np.array(features).reshape(1, -1)
    features_imputed = model_pkg['imputer'].transform(features_array)
    features_scaled = model_pkg['scaler'].transform(features_imputed)

    # 层级1：唤醒度
    arousal_proba = model_pkg['arousal_model'].predict_proba(features_scaled)[0]
    arousal_pred = model_pkg['arousal_model'].predict(features_scaled)[0]

    if arousal_pred == 0:
        return {
            'emotion': '平静',
            'confidence': arousal_proba[0],
            'arousal': '低唤醒'
        }
    else:
        # 层级2：效价
        valence_proba = model_pkg['valence_model'].predict_proba(features_scaled)[0]
        valence_pred = model_pkg['valence_model'].predict(features_scaled)[0]

        return {
            'emotion': '兴奋' if valence_pred == 1 else '恐惧/压力',
            'confidence': valence_proba[valence_pred],
            'arousal': '高唤醒',
            'valence_proba': {
                '恐惧/压力': valence_proba[0],
                '兴奋': valence_proba[1]
            }
        }


print(f"\n{'=' * 60}")
print(f"✅ 层次化分类模型训练完成！")
print(f"\n使用示例:")
print(f"   model_pkg = joblib.load('my_emotion_model_v5_hierarchical.pkl')")
print(f"   result = predict_emotion_hierarchical(rr_intervals, model_pkg)")
print(f"\n策略优势:")
print(f"1. 先区分唤醒度（高/低），再区分效价（积极/消极）")
print(f"2. 每个子问题都是二分类，更简单也更准确")
print(f"3. 专门针对'兴奋vs恐惧/压力'训练了效价分类器")
print(f"4. 修正了SMOTE样本不一致问题")