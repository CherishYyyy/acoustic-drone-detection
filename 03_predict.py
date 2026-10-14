import torch
import torch.nn as nn
import librosa
import numpy as np

# 1. 定义和训练时一模一样的CNN模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 3 * 10, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 3 * 10)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 加载刚才保存的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cnn_drone_model.pth', map_location=device))
model.eval() # 切换到评估模式

# 3. 处理你要测试的音频（换成你自己录的一段真实噪音，比如键盘声或说话声）
audio_path = 'E:/DroneAudioDataset-master/DroneAudioDataset-master/Multiclass_Drone_Audio/bebop_1/mixed_65-bebop_000_.wav' # 👈 改这里！

y, sr = librosa.load(audio_path, sr=16000)
if len(y) < 1600:
    print("音频太短，请换一个！")
else:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # 统一长度
    if mfcc.shape[1] < 40:
        mfcc = np.pad(mfcc, ((0, 0), (0, 40 - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :40]
    
    # 转换成模型需要的数据格式
    input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 4. 让模型预测
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        prob = torch.nn.functional.softmax(output, dim=1)
        
    print("-" * 30)
    if predicted.item() == 1:
        print(f"🚨 警告：检测到无人机！(置信度: {prob[0][1]*100:.2f}%)")
    else:
        print(f"✅ 安全：判定为噪音。(置信度: {prob[0][0]*100:.2f}%)")
    print("-" * 30)