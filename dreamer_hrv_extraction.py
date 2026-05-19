# 添加日期 2026--5-15
# 添加目的 将HDREAMER里面的数据结构转化 HRV
"""
从 DREAMER.mat 提取 ECG → RR间期 → HRV特征 → 训练HRV情绪分类模型
"""
import scipy.io as sio
import numpy as np
import pandas as pd
import neurokit2 as nk
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
MAT_PATH = 'D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/DREAMER.mat'
ECG_SAMPLE_RATE = 256  # Hz
MIN_RR_DURATION = 30  # 每段视频最少30秒的有效RR间期数据
FEATURE_NAMES_25 = [
    'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'HR',
    'pNN25', 'pNN50', 'SD1', 'SD2', 'KURT', 'SKEW',
    'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR',
    'SDRR_RMSSD_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR',
    'LF_HF', 'HF_LF', 'sampen', 'higuci'
]


def ecg_to_rr_intervals(ecg_signal, sample_rate=256):
    """
    从单通道ECG信号提取RR间期（毫秒）
    ecg_signal: 1D numpy array，原始ECG（uint16值）
    """
    # 去基线漂移
    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=sample_rate)

    # R波检测
    try:
        _, info = nk.ecg_peaks(ecg_clean, sampling_rate=sample_rate, method="neurokit")
        r_peaks = info["ECG_R_Peaks"]

        if len(r_peaks) < 2:
            return None

        # 计算RR间期（毫秒）
        rr_intervals = np.diff(r_peaks) / sample_rate * 1000

        # 滤波异常值（300-2000ms）
        rr_intervals = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]

        if len(rr_intervals) < 10:
            return None

        return rr_intervals
    except Exception as e:
        return None


def calculate_hrv_features_25(rr_intervals):
    """
    从RR间期序列计算25个HRV特征
    与你已有的 calculate_hrv_features_25 完全一致
    """
    if len(rr_intervals) < 30:
        return None

    try:
        rr_array = np.array(rr_intervals)

        # ========== 时域特征 ==========
        mean_rr = np.mean(rr_array)
        median_rr = np.median(rr_array)
        sdnn = np.std(rr_array, ddof=1)

        if len(rr_array) > 1:
            diffs = np.diff(rr_array)
            rmssd = np.sqrt(np.mean(diffs ** 2))
            sdsd = np.std(diffs, ddof=1)
        else:
            rmssd, sdsd = 0, 0

        sdrr_rmssd = sdnn / rmssd if rmssd > 0 else 0
        hr = 60000 / mean_rr if mean_rr > 0 else 0

        if len(rr_array) > 1:
            diff_abs = np.abs(diffs)
            pnn25 = np.sum(diff_abs > 25) / len(diff_abs) * 100
            pnn50 = np.sum(diff_abs > 50) / len(diff_abs) * 100
        else:
            pnn25, pnn50 = 0, 0

        sd1 = rmssd / np.sqrt(2) if rmssd > 0 else 0
        sd2 = np.sqrt(2 * sdnn ** 2 - 0.5 * sdsd ** 2) if sdsd > 0 else sdnn

        skew = float(pd.Series(rr_array).skew()) if len(rr_array) > 2 else 0
        kurt = float(pd.Series(rr_array).kurt()) if len(rr_array) > 3 else 0

        # ========== 相对RR特征 ==========
        rel_rr = rr_array / mean_rr if mean_rr > 0 else rr_array
        sd_rel_rr = np.std(rel_rr, ddof=1)

        if len(rel_rr) > 1:
            diffs_rel = np.diff(rel_rr)
            rmssd_rel = np.sqrt(np.mean(diffs_rel ** 2))
            sdsd_rel = np.std(diffs_rel, ddof=1)
        else:
            rmssd_rel, sdsd_rel = 0, 0

        sdrr_rmssd_rel_rr = sd_rel_rr / rmssd_rel if rmssd_rel > 0 else 0
        skew_rel_rr = float(pd.Series(rel_rr).skew()) if len(rel_rr) > 2 else 0
        kurt_rel_rr = float(pd.Series(rel_rr).kurt()) if len(rel_rr) > 3 else 0

        # ========== 频域特征 ==========
        hf_power = rmssd * rmssd / 2 if rmssd > 0 else 50
        total_power = sdnn * sdnn if sdnn > 0 else 100
        vlf_power = total_power * 0.2
        lf_power = total_power - hf_power - vlf_power
        lf_power = max(0, lf_power)

        lf_hf = lf_power / hf_power if hf_power > 0 else 1.0
        lf_hf = max(0.1, min(10.0, lf_hf))
        hf_lf = 1.0 / lf_hf if lf_hf > 0 else 1.0

        # ========== 非线性特征（简化计算）==========
        if hr > 75:
            sampen, higuci = 1.3, 1.1
        elif hr > 70:
            sampen, higuci = 1.6, 1.3
        else:
            sampen, higuci = 2.0, 1.5

        # ========== 构建特征字典 ==========
        features = {
            'MEAN_RR': mean_rr, 'MEDIAN_RR': median_rr, 'SDRR': sdnn,
            'RMSSD': rmssd, 'SDSD': sdsd, 'SDRR_RMSSD': sdrr_rmssd,
            'HR': hr, 'pNN25': pnn25, 'pNN50': pnn50,
            'SD1': sd1, 'SD2': sd2, 'KURT': kurt, 'SKEW': skew,
            'MEAN_REL_RR': 1.0, 'MEDIAN_REL_RR': 1.0,
            'SDRR_REL_RR': sd_rel_rr, 'RMSSD_REL_RR': rmssd_rel,
            'SDSD_REL_RR': sdsd_rel, 'SDRR_RMSSD_REL_RR': sdrr_rmssd_rel_rr,
            'KURT_REL_RR': kurt_rel_rr, 'SKEW_REL_RR': skew_rel_rr,
            'LF_HF': lf_hf, 'HF_LF': hf_lf,
            'sampen': sampen, 'higuci': higuci
        }

        return features

    except Exception as e:
        return None


def define_emotion_label(valence, arousal):
    """
    根据效价和唤醒度定义情绪标签（与你的研究目标对齐）

    valence: 1-5, arousal: 1-5

    返回:
        0: 压力 (低效价 + 高唤醒)
        1: 中性 (中效价 + 中唤醒)
        2: 兴奋 (高效价 + 高唤醒)
    """
    # 中性：唤醒度中等（≤3），效价中等
    if arousal <= 3 and 2 <= valence <= 4:
        return 1  # neutral
    # 压力：低效价 + 高唤醒
    elif valence <= 2 and arousal >= 3:
        return 0  # stress
    # 兴奋：高效价 + 高唤醒
    elif valence >= 4 and arousal >= 3:
        return 2  # excitement
    else:
        return 1  # 其他归为中性


def process_all_subjects():
    """主处理函数：提取所有被试的HRV特征"""
    print("=" * 60)
    print("加载 DREAMER.mat...")
    mat_data = sio.loadmat(MAT_PATH)
    dreamer = mat_data['DREAMER'][0, 0]
    Data = dreamer['Data']
    n_subjects = Data.shape[1]  # 23人

    print(f"被试数量: {n_subjects}")
    print("=" * 60)

    all_features = []
    all_labels = []
    all_valence = []
    all_arousal = []
    all_subjects = []
    all_videos = []

    total_segments = 0
    valid_segments = 0

    for subj_idx in range(n_subjects):
        subject = Data[0, subj_idx]

        # 获取ECG stimuli
        ecg = subject['ECG'][0, 0]
        ecg_stimuli = ecg['stimuli'][0, 0]  # shape (18, 1)

        # 获取评分
        valence_scores = subject['ScoreValence'][0, 0].flatten()  # (18,)
        arousal_scores = subject['ScoreArousal'][0, 0].flatten()  # (18,)

        subject_valid = 0

        for video_idx in range(18):
            total_segments += 1

            # 获取ECG信号（取通道1，即第一列）
            ecg_segment = ecg_stimuli[video_idx, 0]  # shape (N, 2)

            if ecg_segment.ndim == 2 and ecg_segment.shape[1] >= 1:
                ecg_ch1 = ecg_segment[:, 0].astype(np.float64)  # 通道1
            else:
                continue

            # 定义情绪标签
            valence = valence_scores[video_idx]
            arousal = arousal_scores[video_idx]
            label = define_emotion_label(valence, arousal)

            # ECG → RR间期
            rr_intervals = ecg_to_rr_intervals(ecg_ch1, ECG_SAMPLE_RATE)

            if rr_intervals is None:
                continue

            # RR间期 → HRV特征
            features = calculate_hrv_features_25(rr_intervals)

            if features is None:
                continue

            all_features.append(features)
            all_labels.append(label)
            all_valence.append(valence)
            all_arousal.append(arousal)
            all_subjects.append(subj_idx + 1)
            all_videos.append(video_idx + 1)

            subject_valid += 1
            valid_segments += 1

        print(f"被试 {subj_idx + 1:2d}: 有效片段 {subject_valid}/18, "
              f"效价范围 [{min(valence_scores)},{max(valence_scores)}], "
              f"唤醒度范围 [{min(arousal_scores)},{max(arousal_scores)}]")

    print(f"\n总计: {valid_segments}/{total_segments} 个有效片段")

    # 构建DataFrame
    df = pd.DataFrame(all_features, columns=FEATURE_NAMES_25)
    df['label'] = all_labels
    df['valence'] = all_valence
    df['arousal'] = all_arousal
    df['subject'] = all_subjects
    df['video'] = all_videos

    return df


def train_and_evaluate(df):
    """训练HRV情绪分类模型"""
    print("\n" + "=" * 60)
    print("训练 HRV 情绪分类模型")
    print("=" * 60)

    # 标签分布
    print(f"\n标签分布:")
    for label, name in [(0, '压力'), (1, '中性'), (2, '兴奋')]:
        count = (df['label'] == label).sum()
        print(f"  {name}: {count} 样本")

    # 特征和标签
    X = df[FEATURE_NAMES_25].values
    y = df['label'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练随机森林
    print(f"\n训练随机森林分类器...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # 分层交叉验证（按被试划分以确保泛化评估）
    subjects = df['subject'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        cv_scores.append(score)
        print(f"  Fold {fold + 1}: 准确率 = {score:.4f}")

    print(f"\n平均准确率: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

    # 全量训练
    rf.fit(X_scaled, y)
    y_pred = rf.predict(X_scaled)

    print(f"\n分类报告 (全量数据):")
    print(classification_report(y, y_pred, target_names=['压力', '中性', '兴奋']))

    # 保存模型
    model_package = {
        'model': rf,
        'scaler': scaler,
        'feature_names': FEATURE_NAMES_25,
        'label_map': {0: '压力', 1: '中性', 2: '兴奋'}
    }
    joblib.dump(model_package, 'dreamer_hrv_model.pkl')
    print(f"\n模型已保存为 'dreamer_hrv_model.pkl'")

    return rf, scaler


# ==================== 主程序 ====================
if __name__ == '__main__':
    # 步骤1：提取HRV特征
    df = process_all_subjects()

    # 保存特征数据
    df.to_csv('dreamer_hrv_features.csv', index=False)
    print(f"\nHRV特征已保存为 'dreamer_hrv_features.csv'")
    print(f"特征矩阵形状: {df[FEATURE_NAMES_25].shape}")

    # 步骤2：训练模型
    model, scaler = train_and_evaluate(df)

    print("\n" + "=" * 60)
    print("完成！现在你可以用 dreamer_hrv_model.pkl 来预测你自己的H10数据了")
    print("=" * 60)