import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# ⚠️ 这里换成你刚刚复制的真实wav文件路径！
# 比如：audio_path = 'C:/Users/AS_60/Desktop/DroneAudioDataset/Multiclass_Drone_Audio/bebop_1/xxx.wav'
audio_path = 'D:/XZ/B_S2_D1_067-bebop_000_.wav'

# 读取音频
y, sr = librosa.load(audio_path, sr=None)
print(f"成功读取音频！时长：{len(y)/sr:.2f} 秒")

# 生成梅尔频谱图
plt.figure(figsize=(10, 4))
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (Drone Audio)')
plt.show()