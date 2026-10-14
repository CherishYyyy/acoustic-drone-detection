import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ================= 1. 基础配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
base_dir = 'E:/Audio_Data'

# ================= 2. 特征提取：改变就在这里！ =================
def extract_mfcc_matrix(file_path, max_len=40):
    y, sr = librosa.load(file_path, sr=16000)
    if len(y) < 1600:
        return None
    # 提取 MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 统一长度：因为CNN需要固定尺寸的输入
    if mfcc.shape[1] < max_len:
        # 不够长就补零
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # 太长就截断
        mfcc = mfcc[:, :max_len]
        
    return mfcc # 返回形状为 (13, 40) 的矩阵

X = [] # 特征
y = [] # 标签

print("开始读取音频数据（保留时间维度）...")
for label, folder in enumerate(['noise', 'drone']):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path): continue
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            feat = extract_mfcc_matrix(os.path.join(folder_path, file))
            if feat is not None:
                X.append(feat)
                y.append(label)

# 转换为 PyTorch 需要的格式：(样本数, 通道数, 高度, 宽度)
# 这里我们用 1 个通道（代表单色图像），尺寸是 (13, 40)
X = np.array(X)
y = np.array(y)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # 增加一个通道维度
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"数据读取完成！共收集了 {len(X)} 个样本。")

# ================= 3. 划分数据集 =================
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.3, random_state=42
)

# 创建 PyTorch 的数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ================= 4. 搭建 CNN 模型 =================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层 1：提取局部特征
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层 2：提取更高层特征
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 全连接层：分类
        self.fc1 = nn.Linear(32 * 3 * 10, 64) # 经过两次池化，尺寸变为 (32, 3, 10)
        self.fc2 = nn.Linear(64, 2) # 二分类（噪音/无人机）
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 3 * 10) # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model = SimpleCNN().to(device)

# ================= 5. 训练模型 =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("开始训练 CNN...")
epochs = 5 # 跑5轮
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ================= 6. 评估与混淆矩阵 =================
model.eval()
y_pred_list = []
y_true_list = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(labels.numpy())

acc = accuracy_score(y_true_list, y_pred_list)
print(f"🎉 CNN 训练完成！AI 的准确率为: {acc * 100:.2f}%")

cm = confusion_matrix(y_true_list, y_pred_list)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['噪音(0)', '无人机(1)'])
disp.plot(cmap='Blues')
plt.title("CNN 分类混淆矩阵")
plt.show()
# 保存模型参数
torch.save(model.state_dict(), 'cnn_drone_model.pth')
print("模型已保存为 cnn_drone_model.pth")