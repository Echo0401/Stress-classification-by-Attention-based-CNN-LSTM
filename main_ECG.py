# main_ECG_complete.py - 完整训练代码
import numpy as np
import pandas as pd
import time
import os

# 导入你的模块
import data_utils as du
import Model_selection as Ms
import Plot_utils as pu

from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold

# ==================== 路径设置 ====================
# 先生成 Denoised.pkl（运行上面的 generate_pkl.py 一次）
pkl_path = "D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/Denoised.pkl"
model_save_path = 'D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/ECG_Model/'
os.makedirs(model_save_path, exist_ok=True)
model_name = 'Fusion_model_Attention_7Convlayer_'

# ==================== 参数设置 ====================
np.random.seed(12)
sampling_rate = 256
left = int(np.round(0.04 * 6 * sampling_rate))  # 61
right = int(np.round(0.04 * 10 * sampling_rate))  # 102
window = sampling_rate * 5  # 1280
accuracies = []
fold_number = 1

# ==================== 加载数据 ====================
print("=" * 60)
print("加载数据...")
print("=" * 60)

# 使用你原来的 load_DREAMER_class 函数
excite_b, neutral_b, stress_b = du.load_DREAMER_class(pkl_path, 'beat', window)
excite_r, neutral_r, stress_r = du.load_DREAMER_class(pkl_path, 'rhythm', window)

print(f"Beat数据 - 兴奋: {excite_b.shape}, 中性: {neutral_b.shape}, 压力: {stress_b.shape}")
print(f"Rhythm数据 - 兴奋: {excite_r.shape}, 中性: {neutral_r.shape}, 压力: {stress_r.shape}")

# 对齐数据量（与你原版一致）
excite_b = excite_b[:excite_r.shape[0]]
neutral_b = neutral_b[:neutral_r.shape[0]]
stress_b = stress_b[:stress_r.shape[0]]

print(f"对齐后Beat - 兴奋: {excite_b.shape}, 中性: {neutral_b.shape}, 压力: {stress_b.shape}")

# 创建标签（与你原版完全一致 - 使用 beat 的标签）
excite_label = np.zeros(excite_b.shape[0], dtype=np.int32)  # 兴奋 → 0
neutral_label = np.ones(neutral_b.shape[0], dtype=np.int32)  # 中性 → 1
stress_label = 2 * np.ones(stress_b.shape[0], dtype=np.int32)  # 压力 → 2

# 合并数据
Train_beat = np.concatenate([excite_b, neutral_b, stress_b])
Train_rhythm = np.concatenate([excite_r, neutral_r, stress_r])
Train_label = np.concatenate([excite_label, neutral_label, stress_label])

print(f"合并后 - Beat: {Train_beat.shape}, Rhythm: {Train_rhythm.shape}, Label: {Train_label.shape}")

# 增加通道维度
Train_beat = Train_beat.reshape(Train_beat.shape[0], Train_beat.shape[1], 1)
Train_rhythm = Train_rhythm.reshape(Train_rhythm.shape[0], Train_rhythm.shape[1], 1)

# 打乱数据
train_beat, train_rhythm, train_y = du.shuffle_data2(Train_beat, Train_rhythm, Train_label)

print(f"最终数据 - Beat: {train_beat.shape}, Rhythm: {train_rhythm.shape}, Label: {train_y.shape}")
print(f"标签分布: {np.bincount(train_y)}")
print(f"总样本数: {len(train_y)}")
print(f"预计每轮steps: {len(train_y) // 32}")
# 添加在数据加载后
print("=" * 60)
print("数据诊断信息:")
print("=" * 60)

# 1. 检查数据范围（确认是否做了MinMax归一化）
print(f"Beat数据范围: [{Train_beat.min():.4f}, {Train_beat.max():.4f}]")
print(f"Rhythm数据范围: [{Train_rhythm.min():.4f}, {Train_rhythm.max():.4f}]")

# 2. 检查样本数
print(f"Beat样本总数: {Train_beat.shape[0]}")
print(f"标签分布: {np.bincount(Train_label)}")

# 3. 检查样本长度
print(f"Beat样本长度: {Train_beat.shape[1]}")
print(f"Rhythm样本长度: {Train_rhythm.shape[1]}")

# 4. 检查是否有NaN或Inf
print(f"Beat有NaN: {np.any(np.isnan(Train_beat))}")
print(f"Beat有Inf: {np.any(np.isinf(Train_beat))}")
print(f"Rhythm有NaN: {np.any(np.isnan(Train_rhythm))}")
print(f"Rhythm有Inf: {np.any(np.isinf(Train_rhythm))}")

# ==================== 10折交叉验证 ====================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in skf.split(train_beat, train_y):
    print("=" * 50)
    print(f"Fold {fold_number}/10")

    X_train_b = train_beat[train_index]
    X_test_b = train_beat[test_index]
    X_train_r = train_rhythm[train_index]
    X_test_r = train_rhythm[test_index]
    Y_train = train_y[train_index]
    Y_test = train_y[test_index]

    y_train = to_categorical(Y_train, num_classes=3)
    y_test = to_categorical(Y_test, num_classes=3)

    print(f"训练样本: {X_train_b.shape[0]}, 测试样本: {X_test_b.shape[0]}")
    print(f"训练steps/epoch: {X_train_b.shape[0] // 32}")

    # 使用你原来的 fusion_model
    model = Ms.fusion_model(X_train_b.shape[1], X_train_r.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"开始训练 Fold {fold_number} (60轮)...")
    history = model.fit(
        [X_train_b, X_train_r], y_train,
        validation_data=([X_test_b, X_test_r], y_test),
        epochs=60,
        batch_size=32,
        verbose=1
    )

    # 评估
    scores = model.evaluate([X_test_b, X_test_r], y_test, verbose=0)
    fold_accuracy = scores[1]
    accuracies.append(fold_accuracy)

    print(f"Fold {fold_number} 准确率: {fold_accuracy:.4f}")

    # 保存模型
    model.save(f"{model_save_path}{model_name}{fold_number}_{fold_accuracy:.4f}.h5")

    fold_number += 1


# 最终结果
average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("=" * 60)
print("训练完成!")
print(f"各折准确率: {[f'{acc:.4f}' for acc in accuracies]}")
print(f"平均准确率: {average_accuracy:.4f} (±{std_accuracy:.4f})")