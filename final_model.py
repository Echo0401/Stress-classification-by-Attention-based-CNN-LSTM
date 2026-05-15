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

# ==================== 10折交叉验证 ====================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ==================== 在10折交叉验证结束后添加 ====================

print("\n" + "=" * 60)
print("用全部数据训练最终模型...")
print("=" * 60)

# 1. 创建新模型（和之前一样的结构）
final_model = Ms.fusion_model(train_beat.shape[1], train_rhythm.shape[1])
final_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 2. 用全部数据训练
# 注意：这里没有验证集，因为我们要用所有数据
y_all = to_categorical(train_y, num_classes=3)

history_final = final_model.fit(
    [train_beat, train_rhythm],  # 使用全部数据
    y_all,                        # 全部标签
    epochs=60,                    # 和交叉验证相同的轮数
    batch_size=32,
    verbose=1
)

# 3. 保存最终模型
final_model_path = model_save_path + model_name + 'FINAL_MODEL.h5'
final_model.save(final_model_path)

print(f"\n最终模型已保存到: {final_model_path}")
print("这个模型使用了所有数据进行训练，可以用于实际预测")