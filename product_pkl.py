# generate_pkl.py - 只运行一次，生成 Denoised.pkl
import scipy.io as sciio
import numpy as np
import pickle
from data_utils import butter_highpass, butter_bandstop, butter_lowpass, MinMax
from dataclasses import dataclass
from typing import List


@dataclass
class PersonData:
    ecg_baseline: List
    ecg_stimuli: List
    valance: List
    arousal: List


@dataclass
class FilmData:
    ecg_baseline: np.ndarray
    ecg_stimuli: np.ndarray
    valance: float
    arousal: float


def iterate_persons(ppl_array):
    for person_idx in range(len(ppl_array)):
        ppl_struct = ppl_array[person_idx]
        ecg_struct = ppl_struct['ECG'][0][0]
        films_baseline_array = ecg_struct['baseline'][0][0]
        films_stimuli_array = ecg_struct['stimuli'][0][0]
        valance = ppl_struct['ScoreValence'][0][0]
        arousal = ppl_struct['ScoreArousal'][0][0]
        data = PersonData(films_baseline_array, films_stimuli_array, valance, arousal)
        yield data


def iterate_films(p_data: PersonData):
    for film_idx in range(len(p_data.ecg_baseline)):
        d_ecg_baseline = p_data.ecg_baseline[film_idx][0]
        d_ecg_stimuli = p_data.ecg_stimuli[film_idx][0]
        valance = p_data.valance[film_idx][0]
        arousal = p_data.arousal[film_idx][0]
        data = FilmData(d_ecg_baseline, d_ecg_stimuli, valance, arousal)
        yield data


# 处理数据
mat_path = "D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/DREAMER.mat"
data = sciio.loadmat(mat_path)
ppl_array = data['DREAMER'][0][0]['Data'][0]

Fs = 256
high_cut = 0.5
low_cut = 100
low, high = 57, 63
order = 2

neutral_data, excitement_data, stress_data = [], [], []

for p_data in iterate_persons(ppl_array):
    for f_data in iterate_films(p_data):
        base = f_data.ecg_baseline[:, 0]
        st_data = f_data.ecg_stimuli[:, 0]

        base = butter_highpass(base.flatten(), high_cut, Fs, order)
        st_data = butter_highpass(st_data.flatten(), high_cut, Fs, order)
        base = butter_bandstop(base, low, high, Fs, order)
        st_data = butter_bandstop(st_data, low, high, Fs, order)
        base = butter_lowpass(base, low_cut, Fs, order)
        st_data = butter_lowpass(st_data, low_cut, Fs, order)

        base = MinMax(base)
        st_data = MinMax(st_data)

        neutral_data.append(base)

        if f_data.arousal > 3 and f_data.valance > 3:
            excitement_data.append(st_data)
        elif f_data.arousal > 3 and f_data.valance < 3:
            stress_data.append(st_data)

# 保存
data_dict = {
    'baseline': np.concatenate(neutral_data).flatten(),
    'excitement': np.concatenate(excitement_data).flatten(),
    'stress': np.concatenate(stress_data).flatten()
}

save_path = 'D:/PycharmProjects/Stress-classification-by-Attention-based-CNN-LSTM/Denoised.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"数据已保存到: {save_path}")