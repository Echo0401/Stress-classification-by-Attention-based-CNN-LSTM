# rebuild_and_save.py - 用TF 2.15重建模型
import tensorflow as tf
from tensorflow.keras.layers import (Layer, Input, LSTM, Dense, Conv1D, TimeDistributed,
                                     Reshape, MaxPooling1D, BatchNormalization, Dropout, Activation,
                                     add, Concatenate, GlobalAveragePooling1D, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

print(f"TF版本: {tf.__version__}")


# ==================== 自定义层 ====================
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, query, key, value):
        scores = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = scores / tf.math.sqrt(depth)
        weights = tf.nn.softmax(scores, axis=-1)
        attention_output = tf.matmul(weights, value)
        return attention_output


# ==================== 构建模型（与训练时完全一样）====================
def conv_residual_block(input_tensor, filters, kernel_size, pool_size=2):
    conv1 = Conv1D(filters, kernel_size, activation=None, padding='same')(input_tensor)
    conv2 = Conv1D(filters, kernel_size, activation=None, padding='same')(conv1)
    bn = BatchNormalization()(conv2)
    act = Activation('relu')(bn)
    residual = add([conv1, act])
    max_pool = MaxPooling1D(pool_size, pool_size)(residual)
    return max_pool


def build_model(beat_shape=163, rhythm_shape=1280):
    input_beat = Input(shape=(beat_shape, 1), name='input_beat')

    x_beat = conv_residual_block(input_beat, 16, 11)
    x_beat = conv_residual_block(x_beat, 32, 9)
    x_beat = conv_residual_block(x_beat, 64, 7)
    x_beat = conv_residual_block(x_beat, 128, 5)
    x_beat = conv_residual_block(x_beat, 256, 3)

    conv1_B_5 = Conv1D(128, 1, activation=None, padding='same')(x_beat)
    bn1_B_5 = BatchNormalization()(conv1_B_5)
    act_B_5 = Activation('relu')(bn1_B_5)
    residual_B_5 = add([conv1_B_5, act_B_5])

    re_B = Reshape((1, residual_B_5.shape[1], residual_B_5.shape[2]))(residual_B_5)
    t1_B = TimeDistributed(Flatten())(re_B)
    x_B_LSTM = LSTM(256, return_sequences=True)(t1_B)
    dot_product_1 = DotProductAttention()(x_B_LSTM, x_B_LSTM, x_B_LSTM)
    x_B_LSTM2 = LSTM(50, return_sequences=True)(dot_product_1)
    dot_product_2 = DotProductAttention()(x_B_LSTM2, x_B_LSTM2, x_B_LSTM2)
    GAP_B = GlobalAveragePooling1D()(dot_product_2)

    input_rhythm = Input(shape=(rhythm_shape, 1), name='input_rhythm')

    x_rhythm = conv_residual_block(input_rhythm, 16, 15)
    x_rhythm = conv_residual_block(x_rhythm, 32, 13)
    x_rhythm = conv_residual_block(x_rhythm, 64, 11)
    x_rhythm = conv_residual_block(x_rhythm, 128, 9)
    x_rhythm = conv_residual_block(x_rhythm, 256, 7)
    x_rhythm = conv_residual_block(x_rhythm, 512, 5)
    x_rhythm = conv_residual_block(x_rhythm, 256, 3)

    conv1_R = Conv1D(128, 1, activation=None, padding='same')(x_rhythm)
    bn1_R = BatchNormalization()(conv1_R)
    act_R = Activation('relu')(bn1_R)
    residual_R = add([conv1_R, act_R])

    re_R = Reshape((1, residual_R.shape[1], residual_R.shape[2]))(residual_R)
    t1_R = TimeDistributed(Flatten())(re_R)
    x_R_LSTM = LSTM(256, return_sequences=True)(t1_R)
    dot_product_R = DotProductAttention()(x_R_LSTM, x_R_LSTM, x_R_LSTM)
    x_R_LSTM2 = LSTM(50, return_sequences=True)(dot_product_R)
    dot_product_R2 = DotProductAttention()(x_R_LSTM2, x_R_LSTM2, x_R_LSTM2)
    GAP_R = GlobalAveragePooling1D()(dot_product_R2)

    concatenated = Concatenate()([GAP_B, GAP_R])
    den1 = Dense(128, activation='relu')(concatenated)
    drop = Dropout(0.05)(den1)
    den2 = Dense(64, activation='relu')(drop)
    fusion_output = Dense(3, activation='softmax')(den2)

    model = Model(inputs=[input_beat, input_rhythm], outputs=fusion_output)
    return model


# ==================== 主程序 ====================
print("\n构建新模型...")
model = build_model(163, 1280)

# 加载之前保存的权重
print("加载权重...")
weights_dir = "temp_weights"
weight_files = sorted([f for f in os.listdir(weights_dir) if f.endswith('.npy')])

extracted_weights = []
for wf in weight_files:
    w = np.load(os.path.join(weights_dir, wf))
    extracted_weights.append(w)

print(f"加载了 {len(extracted_weights)} 个权重矩阵")

# 设置权重
model.set_weights(extracted_weights)
print("✅ 权重加载成功！")

# 编译
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 保存为兼容格式
output_dir = "ECG_Model_Final"
os.makedirs(output_dir, exist_ok=True)

# SavedModel格式（最兼容）
model.save(os.path.join(output_dir, "emotion_model_tf"), save_format='tf')
print("✅ SavedModel格式已保存")

# 测试
beat_test = np.random.randn(1, 163, 1).astype(np.float32)
rhythm_test = np.random.randn(1, 1280, 1).astype(np.float32)
pred = model.predict([beat_test, rhythm_test], verbose=0)
print(f"预测输出: 类别{np.argmax(pred)}")
print("✅ 模型重建完成！")