import scipy.io as sio
import numpy as np

mat_path = 'D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/DREAMER.mat'
mat_data = sio.loadmat(mat_path)

dreamer = mat_data['DREAMER'][0, 0]
Data = dreamer['Data']

subject_0 = Data[0, 0]
ecg = subject_0['ECG'][0, 0]
ecg_stimuli = ecg['stimuli'][0, 0]

print(f"ecg_stimuli shape: {ecg_stimuli.shape}")
print(f"ecg_stimuli[0,0] 类型: {type(ecg_stimuli[0, 0])}")

# 取出第一段
seg0 = ecg_stimuli[0, 0]
print(f"\n第一段视频ECG:")
print(f"  类型: {type(seg0)}")
if isinstance(seg0, np.ndarray):
    print(f"  shape: {seg0.shape}")
    print(f"  dtype: {seg0.dtype}")
    print(f"  总采样点数: {seg0.size}")
    print(f"  时长: {seg0.size/256:.1f}秒")
    print(f"  前20个值: {seg0.flatten()[:20]}")
else:
    print(f"  值: {seg0}")

# 如果是多维数组，看是不是双通道
if isinstance(seg0, np.ndarray) and seg0.ndim == 2:
    print(f"\n  可能是双通道ECG")
    print(f"  通道1前10个值: {seg0[0, :10]}")
    print(f"  通道2前10个值: {seg0[1, :10]}")

# 同样检查 baseline
ecg_baseline = ecg['baseline'][0, 0]
print(f"\nECG baseline:")
print(f"  类型: {type(ecg_baseline)}")
print(f"  shape: {ecg_baseline.shape}")
if isinstance(ecg_baseline, np.ndarray):
    # baseline可能是一个长信号，不是按18段分的
    if ecg_baseline.size > 100:
        print(f"  总采样点数: {ecg_baseline.size}")
        print(f"  时长: {ecg_baseline.size/256:.1f}秒")
        print(f"  前10个值: {ecg_baseline.flatten()[:10]}")
    else:
        print(f"  前几个值: {ecg_baseline.flatten()}")