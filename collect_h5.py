import numpy as np
from tensorflow.keras.models import load_model

# 1. 加载全部 10 个模型（可以依次加载，不用同时放显存）
# TODO  增加10个模型
model_paths = [
    'Fusion_model_Attention_7Convlayer_1_0.92556.h5',
    'Fusion_model_Attention_7Convlayer_2_0.91834.h5',
    # ... 共10个
]


def predict_ensemble(beat_data, rhythm_data, model_paths):
    """
    集成预测：10个模型投票
    beat_data, rhythm_data: 预处理好的单样本 (1, seq_len)
    """
    predictions = []

    for path in model_paths:
        model = load_model(path)
        pred = model.predict([beat_data, rhythm_data], verbose=0)
        predicted_class = np.argmax(pred, axis=1)[0]  # 0=兴奋, 1=中性, 2=压力
        predictions.append(predicted_class)

        # 可选：不保留所有模型在内存，用完后删除
        del model

    # 投票
    final_class = np.bincount(predictions).argmax()
    return final_class, predictions