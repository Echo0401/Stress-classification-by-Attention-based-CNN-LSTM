#!/usr/bin/env python
# coding: utf-8
# 注意力融合模型 - 包含模型选择功能

import tensorflow as tf
from tensorflow.keras.layers import (Layer, Input, LSTM, Dense, Flatten, Conv1D, TimeDistributed,
                                     Reshape, MaxPooling1D, BatchNormalization, Dropout, Activation,
                                     add, Concatenate, GlobalAveragePooling1D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
import numpy as np
import os
import glob


# ===================== 1. 修复早停函数 =====================
def set_early_stopping(patience=15, min_delta=0.005):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        min_delta=min_delta,
        restore_best_weights=True
    )
    return early_stopping


# ===================== 2. 数据划分函数 =====================
def train_val_test_split(data, label, stratify=False):
    if stratify == False:
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            data, label, test_size=0.2, random_state=128
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=0.25, random_state=128
        )
    else:
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            data, label, test_size=0.2, random_state=128, stratify=label
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=0.25, random_state=128, stratify=y_trainval
        )
    return x_train, y_train, x_val, y_val, x_test, y_test


# ===================== 3. DotProductAttention类 =====================
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


# ===================== 4. 卷积残差块 =====================
def conv_residual_block(input_tensor, filters, kernel_size, pool_size=2):
    conv1 = Conv1D(filters, kernel_size, activation=None, padding='same')(input_tensor)
    conv2 = Conv1D(filters, kernel_size, activation=None, padding='same')(conv1)
    bn = BatchNormalization()(conv2)
    act = Activation('relu')(bn)
    residual = add([conv1, act])
    max_pool = MaxPooling1D(pool_size, pool_size)(residual)
    return max_pool


# ===================== 5. 融合模型 =====================
def fusion_model(beat_shape, rhythm_shape):
    # ---------------- Beat 分支 ----------------
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

    # ---------------- Rhythm 分支 ----------------
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

    # ---------------- 融合分支 ----------------
    concatenated = Concatenate()([GAP_B, GAP_R])
    den1 = Dense(128, activation='relu')(concatenated)
    drop = Dropout(0.05)(den1)
    den2 = Dense(64, activation='relu')(drop)
    fusion_output = Dense(3, activation='softmax')(den2)

    fusion_model = Model(inputs=[input_beat, input_rhythm], outputs=fusion_output)
    fusion_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    fusion_model.summary()

    return fusion_model


# ===================== 6. 单输入模型 =====================
def single_input_model(input_shape):
    """
    单输入模型（用于RR间期数据）
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense, Flatten, \
        Attention, Permute, Multiply, Lambda
    from tensorflow.keras import backend as K
    import tensorflow as tf

    inputs = Input(shape=(input_shape, 1), name='rr_input')

    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    attention_weights = Dense(1, activation='sigmoid')(x)
    x = Multiply()([x, attention_weights])

    x = Permute((2, 1))(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.5)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax', name='emotion_output')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# ===================== 7. 训练函数 =====================
def train_fusion_model(beat_data, rhythm_data, labels, stratify_split=True):
    """
    训练融合模型
    """
    x_train_beat, y_train, x_val_beat, y_val, x_test_beat, y_test = train_val_test_split(
        beat_data, labels, stratify=stratify_split
    )
    x_train_rhythm, _, x_val_rhythm, _, x_test_rhythm, _ = train_val_test_split(
        rhythm_data, labels, stratify=stratify_split
    )

    beat_shape = beat_data.shape[1]
    rhythm_shape = rhythm_data.shape[1]
    model = fusion_model(beat_shape, rhythm_shape)

    early_stopping = set_early_stopping()

    history = model.fit(
        x=[x_train_beat, x_train_rhythm],
        y=y_train,
        validation_data=([x_val_beat, x_val_rhythm], y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    y_pred = model.predict([x_test_beat, x_test_rhythm])
    y_pred_argmax = tf.argmax(y_pred, axis=1).numpy()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_argmax),
        'precision': precision_score(y_test, y_pred_argmax, average='weighted'),
        'recall': recall_score(y_test, y_pred_argmax, average='weighted'),
        'f1': f1_score(y_test, y_pred_argmax, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred_argmax)
    }

    print("\n测试集分类报告：")
    print(classification_report(y_test, y_pred_argmax))

    return model, history, metrics


# ===================== 8. 模型选择器类（新增核心功能）=====================
class ModelSelector:
    """
    模型选择器：从已训练的模型中选择最好的模型
    """

    def __init__(self, models_dir='./ECG_Model/'):
        self.models_dir = models_dir
        self.models_info = []

    def load_models_info(self):
        """
        加载模型文件夹中的所有模型信息
        """
        if not os.path.exists(self.models_dir):
            raise ValueError(f"模型目录 {self.models_dir} 不存在！")

        # 获取所有.h5模型文件
        model_files = glob.glob(os.path.join(self.models_dir, '*.h5'))

        if not model_files:
            raise ValueError(f"在 {self.models_dir} 中没有找到.h5模型文件！")

        print(f"找到 {len(model_files)} 个模型文件")

        self.models_info = []
        for model_file in model_files:
            filename = os.path.basename(model_file)
            # 尝试从文件名中提取准确率（假设格式为：Fusion_model_Attention_7Convlayer_X_0.XXXX.h5）
            try:
                parts = filename.split('_')
                fold_idx = next(i for i, part in enumerate(parts) if part.isdigit())
                fold_num = int(parts[fold_idx])
                # 提取准确率（最后一个数字部分去掉.h5）
                accuracy_str = parts[-1].replace('.h5', '')
                accuracy = float(accuracy_str)

                self.models_info.append({
                    'path': model_file,
                    'filename': filename,
                    'fold': fold_num,
                    'accuracy': accuracy
                })
            except:
                # 如果无法从文件名解析，则添加基本信息
                self.models_info.append({
                    'path': model_file,
                    'filename': filename,
                    'fold': len(self.models_info) + 1,
                    'accuracy': None
                })

        # 按准确率降序排序
        self.models_info.sort(key=lambda x: x.get('accuracy', 0) if x.get('accuracy') is not None else 0, reverse=True)

        return self.models_info

    def get_best_model_by_accuracy(self):
        """
        根据准确率选择最佳模型（需要文件名中包含准确率）
        """
        if not self.models_info:
            self.load_models_info()

        best_model = self.models_info[0]
        if best_model['accuracy'] is None:
            print("警告：无法从文件名解析准确率，请使用 evaluate_models 方法进行评估")

        print(f"\n=== 基于文件名的准确率选择 ===")
        print(f"最佳模型: {best_model['filename']}")
        print(f"准确率: {best_model['accuracy']:.4f}")
        print(f"Fold: {best_model['fold']}")

        return best_model

    def load_model(self, model_path):
        """
        加载保存的模型
        """
        try:
            model = load_model(model_path, custom_objects={'DotProductAttention': DotProductAttention})
            print(f"成功加载模型: {model_path}")
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None

    def evaluate_models_on_data(self, X_test_beat, X_test_rhythm, y_test):
        """
        在测试数据上评估所有模型，选择最佳模型

        参数:
            X_test_beat: Beat分支测试数据
            X_test_rhythm: Rhythm分支测试数据
            y_test: 测试标签
        """
        if not self.models_info:
            self.load_models_info()

        results = []
        print(f"\n评估 {len(self.models_info)} 个模型...")

        for i, model_info in enumerate(self.models_info):
            print(f"\n[{i + 1}/{len(self.models_info)}] 评估模型: {model_info['filename']}")

            try:
                model = self.load_model(model_info['path'])
                if model is None:
                    continue

                # 评估模型
                scores = model.evaluate([X_test_beat, X_test_rhythm], y_test, verbose=0)

                # 预测并计算其他指标
                y_pred = model.predict([X_test_beat, X_test_rhythm], verbose=0)

                if y_test.ndim > 1 and y_test.shape[1] > 1:
                    # one-hot编码的标签
                    y_test_labels = np.argmax(y_test, axis=1)
                else:
                    y_test_labels = y_test

                y_pred_labels = np.argmax(y_pred, axis=1)

                metrics = {
                    'accuracy': accuracy_score(y_test_labels, y_pred_labels),
                    'precision': precision_score(y_test_labels, y_pred_labels, average='weighted'),
                    'recall': recall_score(y_test_labels, y_pred_labels, average='weighted'),
                    'f1': f1_score(y_test_labels, y_pred_labels, average='weighted'),
                    'loss': scores[0],
                    'model_path': model_info['path'],
                    'filename': model_info['filename']
                }

                results.append(metrics)
                print(f"  准确率: {metrics['accuracy']:.4f}, F1分数: {metrics['f1']:.4f}")

            except Exception as e:
                print(f"  评估失败: {e}")
                continue

        # 按准确率排序结果
        if results:
            results.sort(key=lambda x: x['accuracy'], reverse=True)

            print("\n" + "=" * 60)
            print("模型评估结果排名:")
            print("=" * 60)
            for i, result in enumerate(results):
                print(f"{i + 1}. {result['filename']}")
                print(f"   准确率: {result['accuracy']:.4f}")
                print(f"   F1分数: {result['f1']:.4f}")
                print(f"   精确率: {result['precision']:.4f}")
                print(f"   召回率: {result['recall']:.4f}")
                print()

            return results

        return None

    def get_best_model_by_evaluation(self, X_test, X_test_r, y_test):
        """
        通过实际评估选择最佳模型
        """
        results = self.evaluate_models_on_data(X_test, X_test_r, y_test)
        if results:
            best_result = results[0]

            print("=" * 60)
            print("最终选择的最佳模型:")
            print("=" * 60)
            print(f"文件名: {best_result['filename']}")
            print(f"准确率: {best_result['accuracy']:.4f}")
            print(f"F1分数: {best_result['f1']:.4f}")

            # 加载并返回最佳模型
            best_model = self.load_model(best_result['model_path'])

            return best_model, best_result

        return None, None


# ===================== 9. 便捷函数：快速模型选择 =====================
def select_best_model(model_dir='./ECG_Model/', X_test_b=None, X_test_r=None, y_test=None):
    """
    快速选择最佳模型的便捷函数

    如果提供了测试数据，将在数据上评估模型性能选择最佳模型；
    否则，将从文件名中提取准确率选择最佳模型。
    """
    selector = ModelSelector(model_dir)

    if X_test_b is not None and X_test_r is not None and y_test is not None:
        print("使用测试数据评估所有模型...")
        best_model, best_result = selector.get_best_model_by_evaluation(X_test_b, X_test_r, y_test)
    else:
        print("从文件名解析准确率...")
        best_model_info = selector.get_best_model_by_accuracy()
        best_model = selector.load_model(best_model_info['path'])
        best_result = best_model_info

    return best_model, best_result


# ===================== 10. 示例调用 =====================
if __name__ == "__main__":
    # print("=" * 60)
    # print("模型选择器示例")
    # print("=" * 60)
    #
    # # 方法1：仅从文件名选择最佳模型
    # print("\n方法1：基于文件名的准确率选择")
    # best_model, best_info = select_best_model(model_dir='./ECG_Model/')
    # if best_model:
    #     print(f"\n最佳模型已加载: {best_info}")
    #     best_model.summary()


    model_dir = 'D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/ECG_Model/'
    selector = ModelSelector(model_dir)

    # 查看所有模型按准确率排名
    models_info = selector.load_models_info()
    print("\n模型排名（按准确率）：")
    for i, info in enumerate(models_info):
        print(f"{i + 1}. {info['filename']} - 准确率: {info['accuracy']}")

    # 自动选择最佳模型
    best = models_info[0]
    print(f"\n最佳模型是: {best['filename']}")
    print(f"准确率: {best['accuracy']:.4f}")
    best_model = selector.load_model(best['path'])

    # 方法2：如果要在实际数据上评估（需要数据）
    # 如果您想在实际数据上测试，请取消下面的注释并提供数据
    """
    # 加载您的测试数据
    import data_utils as du

    # 这里需要根据您的实际数据路径来加载
    # 示例代码（需要根据实际情况调整）:

    # X_test_b, X_test_r, y_test = 加载您的测试数据...

    # best_model, best_result = select_best_model(
    #     model_dir='./ECG_Model/',
    #     X_test_b=X_test_b,
    #     X_test_r=X_test_r,
    #     y_test=y_test
    # )

    # if best_model:
    #     print(f"最佳模型准确率: {best_result['accuracy']:.4f}")
    #     print(f"最佳模型F1分数: {best_result['f1']:.4f}")
    """