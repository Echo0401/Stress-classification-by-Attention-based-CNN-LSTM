# test_model_load.py
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) == 3:
                query, key, value = inputs
            elif len(inputs) == 2:
                query, value = inputs
                key = value
            else:
                raise ValueError(f"期望2或3个输入，得到{len(inputs)}个")
        else:
            query = inputs
            key = inputs
            value = inputs

        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output

    def get_config(self):
        config = super(DotProductAttention, self).get_config()
        return config


# 加载模型
model_path = "Model/Fusion_model_Attention_7Convlayer_1(92.556).h5"
print(f"正在加载模型: {model_path}")

model = load_model(model_path, custom_objects={'DotProductAttention': DotProductAttention})
print("✅ 模型加载成功！")
print(f"模型输入: {model.input_names}")
print(f"模型输出: {model.output_names}")

# 测试预测
print("\n测试预测...")
# 创建测试数据（根据你的模型输入形状）
test_beat = np.random.randn(1, 163, 1).astype(np.float32)  # beat输入形状: (batch, 163, 1)
test_rhythm = np.random.randn(1, 1280, 1).astype(np.float32)  # rhythm输入形状: (batch, 1280, 1)

prediction = model.predict([test_beat, test_rhythm], verbose=0)
print(f"预测结果形状: {prediction.shape}")
print(f"预测概率: {prediction[0]}")
print(f"预测类别: {np.argmax(prediction[0])} (0=兴奋, 1=中性, 2=压力)")
print("\n✅ 一切正常！")