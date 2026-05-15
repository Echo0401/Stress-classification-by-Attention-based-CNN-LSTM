import pickle
import numpy as np
import scipy.io as sciio
import matplotlib.pyplot as plt
import time
import logging
import os

from dataclasses import dataclass
from typing import List
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense, Multiply, \
    Permute, Add, Activation, GlobalAveragePooling1D, Concatenate
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import neurokit2 as nk

# 设置Matplotlib为非交互式后端
plt.switch_backend('Agg')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rr_training_log.txt', encoding='utf-8')
    ]
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


@dataclass
class PersonData:
    ecg_baseline: List
    ecg_stimuli: List
    valance: List
    arousal: List


@dataclass
class FilmData:
    ecg_baseline: np.ndarray
    ecg_stimuli: np.ndarray
    valance: float
    arousal: float


def extract_rr_from_ecg(ecg_signal, sampling_rate=256):
    """从原始ECG信号中提取均匀采样的RR间期序列"""
    ecg_signal = np.asarray(ecg_signal).flatten()
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    _, rpeaks_dict = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    rpeaks = rpeaks_dict['ECG_R_Peaks']

    if len(rpeaks) < 3:
        logging.warning(f"检测到R峰数量不足: {len(rpeaks)}，返回零序列")
        return np.zeros(len(ecg_signal))

    rr_intervals = np.diff(rpeaks) / sampling_rate * 1000.0
    rr_indices = rpeaks[1:]
    rr_signal_interpolated = np.interp(
        np.arange(len(ecg_signal)),
        rr_indices,
        rr_intervals
    )

    # 处理边界NaN值
    if np.isnan(rr_signal_interpolated[0]):
        first_valid = np.argwhere(~np.isnan(rr_signal_interpolated)).flatten()
        if len(first_valid) > 0:
            rr_signal_interpolated[:first_valid[0]] = rr_signal_interpolated[first_valid[0]]

    if np.isnan(rr_signal_interpolated[-1]):
        last_valid = np.argwhere(~np.isnan(rr_signal_interpolated)).flatten()
        if len(last_valid) > 0:
            rr_signal_interpolated[last_valid[-1] + 1:] = rr_signal_interpolated[last_valid[-1]]

    return rr_signal_interpolated


def preprocess_rr(rr_signal, fs=256):
    """
    改进的RR间期预处理：Z-score标准化，保留分布信息

    重要：不应该使用MinMaxScaler压缩到[0,1]，那样会丢失RR间期的生理意义
    """
    rr_signal = np.asarray(rr_signal).flatten()

    # 1. 去除生理异常值（毫秒单位）
    rr_signal = np.clip(rr_signal, 300, 1500)

    # 2. 去趋势（去除线性漂移）
    from scipy import signal
    rr_signal = signal.detrend(rr_signal)

    # 3. Z-score标准化（保留相对差异）
    rr_normalized = (rr_signal - np.mean(rr_signal)) / (np.std(rr_signal) + 1e-8)

    # 4. 限制范围防止极端值
    rr_normalized = np.clip(rr_normalized, -3, 3)

    return rr_normalized


def create_optimized_rr_model(input_length, num_classes=3):
    """
    为RR间期优化的深度学习模型

    特点：
    1. 多尺度卷积核，自动学习不同时间尺度的HRV模式
    2. 残差连接，帮助训练深层网络
    3. 全局平均池化替代Flatten，减少参数量
    """
    inputs = Input(shape=(input_length, 1), name='rr_input')

    # 第一层：大卷积核捕捉长时程模式
    x = Conv1D(filters=64, kernel_size=15, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # 多尺度特征提取（自动学习LF/HF等频域特征）
    scale1 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x)
    scale2 = Conv1D(filters=64, kernel_size=11, padding='same', activation='relu')(x)
    scale3 = Conv1D(filters=64, kernel_size=21, padding='same', activation='relu')(x)

    x = Concatenate()([scale1, scale2, scale3])
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    # 残差块1
    shortcut = Conv1D(128, 1, padding='same')(x)
    x = Conv1D(128, 7, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv1D(128, 7, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # 残差块2
    shortcut = Conv1D(256, 1, padding='same')(x)
    x = Conv1D(256, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(256, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # 注意力机制
    attention_weights = Conv1D(1, 1, activation='sigmoid')(x)
    x = Multiply()([x, attention_weights])

    # 全局平均池化（替代Flatten + Dense，更高效）
    x = GlobalAveragePooling1D()(x)

    # 全连接层
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def iterate_persons(ppl_array):
    for person_idx in range(len(ppl_array)):
        ppl_struct = ppl_array[person_idx]
        ecg_struct = ppl_struct['ECG'][0][0]
        films_baseline_array = ecg_struct['baseline'][0][0]
        films_stimuli_array = ecg_struct['stimuli'][0][0]
        valance = ppl_struct['ScoreValence'][0][0]
        arousal = ppl_struct['ScoreArousal'][0][0]
        assert len(films_baseline_array) == len(films_stimuli_array), "len of baseline and simuli array not the same."
        data = PersonData(films_baseline_array, films_stimuli_array, valance, arousal)
        yield data


def iterate_films(p_data: PersonData):
    for film_idx in range(len(p_data.ecg_baseline)):
        d_ecg_baseline = p_data.ecg_baseline[film_idx][0]
        d_ecg_stimuli = p_data.ecg_stimuli[film_idx][0]
        valance = p_data.valance[film_idx][0]
        arousal = p_data.arousal[film_idx][0]
        data = FilmData(d_ecg_baseline, d_ecg_stimuli, valance, arousal)
        yield data


# ==================== 数据处理与保存 ====================
logging.info("=" * 60)
logging.info("开始数据处理 (RR间期版本)...")
logging.info("=" * 60)

mat_path = "D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/DREAMER.mat"
data_path = mat_path.replace("\\", "/")
save_path = 'D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/RR_Processed.pkl'

data = sciio.loadmat(data_path)
ppl_array = data['DREAMER'][0][0]['Data'][0]
logging.info(f"原始数据: {ppl_array.shape[0]} 个被试")

Fs = 256
window_seconds = 5
window_samples = Fs * window_seconds
stride_samples = window_samples // 2  # 50%重叠

neutral_segments = []
excitement_segments = []
stress_segments = []

for p_data in iterate_persons(ppl_array):
    for f_data in iterate_films(p_data):

        # 使用第一导联ECG
        base_ecg = f_data.ecg_baseline[:, 0]
        st_ecg = f_data.ecg_stimuli[:, 0]

        # 提取RR间期
        base_rr = extract_rr_from_ecg(base_ecg, Fs)
        st_rr = extract_rr_from_ecg(st_ecg, Fs)

        # 预处理
        base_rr = preprocess_rr(base_rr, Fs)
        st_rr = preprocess_rr(st_rr, Fs)

        # 滑动窗口切分基线数据 -> 中性
        for start in range(0, len(base_rr) - window_samples + 1, stride_samples):
            neutral_segments.append(base_rr[start:start + window_samples])

        # 根据情绪分类切分刺激期数据
        if f_data.arousal > 3 and f_data.valance > 3:
            # 兴奋类
            for start in range(0, len(st_rr) - window_samples + 1, stride_samples):
                excitement_segments.append(st_rr[start:start + window_samples])
        elif f_data.arousal > 3 and f_data.valance < 3:
            # 压力类
            for start in range(0, len(st_rr) - window_samples + 1, stride_samples):
                stress_segments.append(st_rr[start:start + window_samples])

logging.info(
    f"数据统计 - 中性: {len(neutral_segments)}, 兴奋: {len(excitement_segments)}, 压力: {len(stress_segments)}")

# 保存数据
data_dict = {
    'neutral': np.array(neutral_segments),
    'excitement': np.array(excitement_segments),
    'stress': np.array(stress_segments)
}

with open(save_path, 'wb') as f:
    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

logging.info(f"数据保存完成: {save_path}")

# ==================== 模型训练 ====================
logging.info("=" * 60)
logging.info("开始模型训练 (优化版RR间期模型)...")
logging.info("=" * 60)

# 加载数据
with open(save_path, 'rb') as f:
    data_dict = pickle.load(f)

X_neutral = data_dict['neutral']
X_excitement = data_dict['excitement']
X_stress = data_dict['stress']

# ✅ 修复：正确设置标签
# 中性 = 0, 兴奋 = 1, 压力 = 2
y_neutral = np.zeros(len(X_neutral), dtype=np.int32)  # 中性 → 0
y_excitement = np.ones(len(X_excitement), dtype=np.int32)  # 兴奋 → 1
y_stress = 2 * np.ones(len(X_stress), dtype=np.int32)  # 压力 → 2

# 合并数据（顺序：中性、兴奋、压力）
X = np.concatenate([X_neutral, X_excitement, X_stress], axis=0)
y = np.concatenate([y_neutral, y_excitement, y_stress], axis=0)

# 增加通道维度 (samples, length, 1)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 混洗数据
X, y = shuffle(X, y, random_state=42)

logging.info(f"总样本数: {len(X)}, 序列长度: {X.shape[1]}")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    label_name = ['中性', '兴奋', '压力'][u]
    logging.info(f"类别分布 - {label_name}: {c}")

# ==================== 数据诊断 ====================
logging.info("=" * 60)
logging.info("开始数据诊断...")
logging.info("=" * 60)

logging.info(f"总样本数: {len(X)}")
logging.info(f"标签分布: {np.bincount(y.astype(int))}")
logging.info(f"序列长度: {X.shape[1]}")
logging.info(f"RR间期范围: [{X.min():.4f}, {X.max():.4f}]")
logging.info(f"RR间期均值: {X.mean():.4f} ± {X.std():.4f}")

# 检查异常值
invalid = np.sum((X < -5) | (X > 5))
logging.info(f"异常RR间期数量: {invalid} (占比: {invalid / X.size * 100:.4f}%)")

# 检查NaN/Inf
nan_count = np.sum(np.isnan(X))
inf_count = np.sum(np.isinf(X))
logging.info(f"NaN数量: {nan_count}, Inf数量: {inf_count}")

# 保存样本图像
try:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, label in enumerate(['中性', '兴奋', '压力']):
        idx_list = np.where(y == i)[0]
        if len(idx_list) > 0:
            idx = idx_list[0]
            axes[i].plot(X[idx, :, 0][:500])
            axes[i].set_title(f'{label} - RR interval')
            axes[i].set_xlabel('Time (samples)')
            axes[i].set_ylabel('Z-score normalized RR')
    plt.tight_layout()
    plt.savefig('rr_diagnostic.png', dpi=150)
    logging.info("样本图像已保存: rr_diagnostic.png")
    plt.close('all')
except Exception as e:
    logging.warning(f"绘图失败: {e}")

logging.info("数据诊断完成")
logging.info("=" * 60)

# ==================== 10折交叉验证训练 ====================
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
accuracies = []
fold_number = 1

# 计算类别权重（处理不平衡）
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
class_weight_dict = dict(enumerate(class_weights))
logging.info(f"类别权重: {class_weight_dict}")

# 早停和学习率衰减
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

model_save_path = 'D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/RR_Optimized_Model/'
os.makedirs(model_save_path, exist_ok=True)
model_name = 'RR_Optimized_'

total_start_time = time.time()

for train_index, test_index in skf.split(X, y):
    fold_start_time = time.time()

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]

    y_train = to_categorical(Y_train, num_classes=3)
    y_test = to_categorical(Y_test, num_classes=3)

    logging.info("=" * 50)
    logging.info(f"Fold {fold_number}/{n_folds}")
    logging.info(f"训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")

    # 创建优化后的模型
    model = create_optimized_rr_model(X.shape[1], num_classes=3)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 打印模型结构
    model.summary()

    logging.info(f"开始训练 Fold {fold_number}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=80,
        batch_size=64,  # 增大batch_size提高训练稳定性
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )

    # 评估
    scores = model.evaluate(X_test, y_test, verbose=0)
    fold_accuracy = scores[1]
    accuracies.append(fold_accuracy)

    # 预测
    pred_prob = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(pred_prob, axis=1)

    fold_time = time.time() - fold_start_time
    logging.info(f"Fold {fold_number} 完成 - 准确率: {fold_accuracy:.4f}, 耗时: {fold_time:.2f}秒")

    # 保存模型
    model.save(f"{model_save_path}{model_name}{fold_number}_{fold_accuracy:.4f}.h5")
    logging.info(f"模型已保存: {model_save_path}{model_name}{fold_number}_{fold_accuracy:.4f}.h5")

    # 绘制并保存混淆矩阵
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        cm = confusion_matrix(Y_test, pred_labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['中性', '兴奋', '压力'])
        disp.plot(ax=ax)
        plt.title(f'Fold {fold_number} Confusion Matrix (Acc: {fold_accuracy:.4f})')
        plt.savefig(f"{model_save_path}{model_name}{fold_number}_confusion_matrix.png", dpi=150)
        plt.close('all')
        logging.info(f"混淆矩阵已保存: {model_save_path}{model_name}{fold_number}_confusion_matrix.png")
    except Exception as e:
        logging.warning(f"绘制混淆矩阵失败: {e}")

    fold_number += 1

total_time = time.time() - total_start_time
average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

logging.info("=" * 60)
logging.info("训练完成!")
logging.info(f"10折交叉验证结果:")
logging.info(f"  各折准确率: {[f'{acc:.4f}' for acc in accuracies]}")
logging.info(f"  平均准确率: {average_accuracy:.4f} (±{std_accuracy:.4f})")
logging.info(f"  最高准确率: {max(accuracies):.4f}")
logging.info(f"  最低准确率: {min(accuracies):.4f}")
logging.info(f"总耗时: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")
logging.info("=" * 60)

# 保存最终结果
results = {
    'accuracies': accuracies,
    'mean_accuracy': average_accuracy,
    'std_accuracy': std_accuracy,
    'max_accuracy': max(accuracies),
    'min_accuracy': min(accuracies)
}

with open(f"{model_save_path}training_results.pkl", 'wb') as f:
    pickle.dump(results, f)

logging.info(f"训练结果已保存: {model_save_path}training_results.pkl")