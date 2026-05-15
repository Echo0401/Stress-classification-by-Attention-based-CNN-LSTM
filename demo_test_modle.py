# test_model_only.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
import sys
sys.path.append('D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM')
import Model_selection as Ms

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            del kwargs['batch_shape']
        if 'optional' in kwargs:
            del kwargs['optional']
        super(DotProductAttention, self).__init__(**kwargs)
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) == 3:
                query, key, value = inputs
            elif len(inputs) == 2:
                query, value = inputs
                key = value
            else:
                query = inputs
                key = inputs
                value = inputs
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
        return super(DotProductAttention, self).get_config()

# 重建模型并加载权重
beat_length = 163
rhythm_length = 1280
model = Ms.fusion_model(beat_length, rhythm_length)
model.load_weights("ECG_Model/Fusion_model_Attention_7Convlayer_FINAL_MODEL.h5")

print("=" * 60)
print("测试1: 随机噪声")
for i in range(5):
    beat = np.random.randn(1, 163, 1)
    rhythm = np.random.randn(1, 1280, 1)
    pred = model.predict([beat, rhythm], verbose=0)
    print(f"  兴奋={pred[0][0]:.3f}, 中性={pred[0][1]:.3f}, 压力={pred[0][2]:.3f}")

print("\n" + "=" * 60)
print("测试2: 全零输入")
beat_zero = np.zeros((1, 163, 1))
rhythm_zero = np.zeros((1, 1280, 1))
pred = model.predict([beat_zero, rhythm_zero], verbose=0)
print(f"兴奋={pred[0][0]:.3f}, 中性={pred[0][1]:.3f}, 压力={pred[0][2]:.3f}")

print("\n" + "=" * 60)
print("测试3: 全1输入")
beat_one = np.ones((1, 163, 1))
rhythm_one = np.ones((1, 1280, 1))
pred = model.predict([beat_one, rhythm_one], verbose=0)
print(f"兴奋={pred[0][0]:.3f}, 中性={pred[0][1]:.3f}, 压力={pred[0][2]:.3f}")

print("\n" + "=" * 60)
print("测试4: 极端值输入 (×100)")
beat_extreme = np.random.randn(1, 163, 1) * 100
rhythm_extreme = np.random.randn(1, 1280, 1) * 100
pred = model.predict([beat_extreme, rhythm_extreme], verbose=0)
print(f"兴奋={pred[0][0]:.3f}, 中性={pred[0][1]:.3f}, 压力={pred[0][2]:.3f}")

print("\n" + "=" * 60)
print("测试5: 正弦波输入")
t = np.linspace(0, 4*np.pi, 1280)
sine_wave = np.sin(t).reshape(1, 1280, 1)
beat_sine = np.sin(np.linspace(0, 2*np.pi, 163)).reshape(1, 163, 1)
pred = model.predict([beat_sine, sine_wave], verbose=0)
print(f"兴奋={pred[0][0]:.3f}, 中性={pred[0][1]:.3f}, 压力={pred[0][2]:.3f}")