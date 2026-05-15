# test_pkl_deep.py
import pickle
import numpy as np

with open("Denoised.pkl", 'rb') as f:
    data = pickle.load(f)

print("数据类型:", type(data))
print("键:", data.keys())

# 查看每个类别的结构
for key in ['baseline', 'excitement', 'stress']:
    if key in data:
        item = data[key]
        print(f"\n{key}:")
        print(f"  类型: {type(item)}")
        if isinstance(item, dict):
            print(f"  键: {item.keys()}")
            # 查看beat和rhythm
            if 'beat' in item:
                beat_data = item['beat']
                print(f"  beat形状: {beat_data.shape if hasattr(beat_data, 'shape') else len(beat_data)}")
                print(f"  beat范围: [{beat_data.min():.4f}, {beat_data.max():.4f}]")
                print(f"  beat均值: {beat_data.mean():.4f}")
            if 'rhythm' in item:
                rhythm_data = item['rhythm']
                print(f"  rhythm形状: {rhythm_data.shape if hasattr(rhythm_data, 'shape') else len(rhythm_data)}")
                print(f"  rhythm范围: [{rhythm_data.min():.4f}, {rhythm_data.max():.4f}]")
                print(f"  rhythm均值: {rhythm_data.mean():.4f}")
        elif hasattr(item, 'shape'):
            print(f"  形状: {item.shape}")
            print(f"  范围: [{item.min():.4f}, {item.max():.4f}]")