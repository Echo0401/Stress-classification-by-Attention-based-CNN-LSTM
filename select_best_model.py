# select_best_model.py - 模型选择脚本
"""
从ECG_Model文件夹中选择最佳模型的脚本
"""

import numpy as np
import os
import sys

# 导入必要的模块
import Model_selection as Ms
import data_utils as du


def main():
    # 设置路径
    model_dir = 'D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/ECG_Model/'

    print("=" * 60)
    print("ECG情绪识别模型选择器")
    print("=" * 60)

    # 创建模型选择器
    selector = Ms.ModelSelector(model_dir)

    # 加载所有模型信息
    models_info = selector.load_models_info()

    if not models_info:
        print("错误：没有找到模型文件！")
        return

    print(f"\n找到 {len(models_info)} 个模型:")
    for i, info in enumerate(models_info):
        print(f"{i + 1}. {info['filename']} - 准确率: {info['accuracy']}")

    # 选择方法
    print("\n选择最佳模型的方法:")
    print("1. 基于文件名的准确率（快速）")
    print("2. 在测试数据上实际评估所有模型（准确但需要测试数据）")

    choice = input("\n请选择方法 (1/2，默认1): ").strip() or "1"

    if choice == "1":
        # 方法1：基于文件名
        best_model_info = selector.get_best_model_by_accuracy()
        best_model = selector.load_model(best_model_info['path'])

        print(f"\n已选择最佳模型: {best_model_info['filename']}")
        print(f"预测准确率: {best_model_info['accuracy']:.4f}")

    elif choice == "2":
        # 方法2：在实际数据上评估
        # 这里需要加载您的测试数据
        # 请根据您的实际数据路径修改

        # 示例：加载DREAMER数据（需要根据实际情况调整）
        pkl_path = "D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/Denoised.pkl"

        print(f"加载测试数据: {pkl_path}")

        # 这里需要根据您的数据加载方式调整
        # 示例代码：
        """
        sampling_rate = 256
        window = sampling_rate * 5

        excite_b, neutral_b, stress_b = du.load_DREAMER_class(pkl_path, 'beat', window)
        excite_r, neutral_r, stress_r = du.load_DREAMER_class(pkl_path, 'rhythm', window)

        # 对齐数据
        excite_b = excite_b[:excite_r.shape[0]]
        neutral_b = neutral_b[:neutral_r.shape[0]]
        stress_b = stress_b[:stress_r.shape[0]]

        # 创建标签
        excite_label = np.zeros(excite_b.shape[0])
        neutral_label = np.ones(neutral_b.shape[0])
        stress_label = 2 * np.ones(stress_b.shape[0])

        # 合并数据
        X_b = np.concatenate([excite_b, neutral_b, stress_b])
        X_r = np.concatenate([excite_r, neutral_r, stress_r])
        y = np.concatenate([excite_label, neutral_label, stress_label])

        # 增加通道维度
        X_b = X_b.reshape(X_b.shape[0], X_b.shape[1], 1)
        X_r = X_r.reshape(X_r.shape[0], X_r.shape[1], 1)

        # 评估所有模型并选择最佳
        results = selector.evaluate_models_on_data(X_b, X_r, y)
        best_model, best_result = selector.get_best_model_by_evaluation(X_b, X_r, y)
        """

        print("此功能需要您提供测试数据路径。请修改脚本中的测试数据加载部分。")

    else:
        print("无效选择，使用默认方法1")
        best_model_info = selector.get_best_model_by_accuracy()
        best_model = selector.load_model(best_model_info['path'])

    # 显示最佳模型信息
    if best_model:
        print("\n" + "=" * 60)
        print("最佳模型详情:")
        print("=" * 60)
        best_model.summary()

        # 保存最佳模型路径到文件
        with open('best_model_info.txt', 'w') as f:
            f.write(f"Best model: {best_model_info['filename']}\n")
            f.write(f"Accuracy: {best_model_info['accuracy']}\n")
            f.write(f"Path: {best_model_info['path']}\n")

        print(f"\n最佳模型信息已保存到 'best_model_info.txt'")


if __name__ == "__main__":
    main()