# extract_weights_recursive.py - 递归提取所有权重
import h5py
import numpy as np
import os

model_path = "D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/ECG_Model/Fusion_model_Attention_7Convlayer_7_0.9317.h5"

print(f"提取模型: {model_path}\n")

weights_list = []
layer_names = []


def extract_weights_recursive(group, path=""):
    """递归提取所有权重"""
    if isinstance(group, h5py.Dataset):
        # 如果是Dataset，直接读取
        data = group[:]
        weights_list.append(data)
        layer_names.append(path)
        print(f"  [{len(weights_list)}] {path}: {data.shape}")
        return

    # 如果是Group，遍历子项
    for key in group.keys():
        new_path = f"{path}/{key}" if path else key
        try:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data = item[:]
                weights_list.append(data)
                layer_names.append(new_path)
                print(f"  [{len(weights_list)}] {new_path}: {data.shape}")
            elif isinstance(item, h5py.Group):
                # 检查是否直接包含数据
                if 'weight' in item or 'bias' in item or 'gamma' in item or 'beta' in item:
                    extract_weights_recursive(item, new_path)
                else:
                    # 继续递归
                    extract_weights_recursive(item, new_path)
        except Exception as e:
            print(f"  跳过 {new_path}: {e}")


with h5py.File(model_path, 'r') as f:
    print("模型权重结构:")
    if 'model_weights' in f:
        extract_weights_recursive(f['model_weights'])

print(f"\n✅ 成功提取 {len(weights_list)} 个权重矩阵")

# 保存
os.makedirs("temp_weights", exist_ok=True)
for i, (name, weight) in enumerate(zip(layer_names, weights_list)):
    np.save(f"temp_weights/weight_{i:03d}.npy", weight)

print(f"权重已保存到 temp_weights/ (共{len(weights_list)}个文件)")
print("\n现在运行 rebuild_and_save.py 重建模型")