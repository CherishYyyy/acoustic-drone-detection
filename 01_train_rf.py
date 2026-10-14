import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ================= 1. 基础配置 =================
# 配置 Matplotlib 正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ⚠️ 极其重要：确保你的路径和电脑里实际的一致
base_dir = 'E:/Audio_Data'

# ================= 2. 特征提取函数 =================
def extract_features(file_path):
    # 读取音频，统一采样率为 16000
    y, sr = librosa.load(file_path, sr=16000)
    
    # 过滤掉太短的音频（少于 0.1 秒）
    if len(y) < 1600:
        return None
        
    # 提取 MFCC 特征（13维）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # 取时间轴的平均值，压缩成一个固定长度的数组
    return np.mean(mfcc.T, axis=0)

# ================= 3. 读取数据 =================
X = [] # 特征
y = [] # 标签
file_paths = [] # 记录文件路径，方便后续排查

print("开始读取音频数据...")

# 3.1 读取无人机音频（标签为 1）
drone_dir = os.path.join(base_dir, 'drone')
for file in os.listdir(drone_dir):
    if file.endswith('.wav'):
        full_path = os.path.join(drone_dir, file)
        feat = extract_features(full_path)
        if feat is not None:
            X.append(feat)
            y.append(1)
            file_paths.append(full_path)

# 3.2 读取噪音音频（标签为 0）
noise_dir = os.path.join(base_dir, 'noise')
for file in os.listdir(noise_dir):
    if file.endswith('.wav'):
        full_path = os.path.join(noise_dir, file)
        feat = extract_features(full_path)
        if feat is not None:
            X.append(feat)
            y.append(0)
            file_paths.append(full_path)

X = np.array(X)
y = np.array(y)
file_paths = np.array(file_paths)

print(f"数据读取完成！共收集了 {len(X)} 个样本。")

# ================= 4. 划分数据集 =================
# 注意：这里把 file_paths 也一起传入切分，保证特征、标签和文件路径一一对应
X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y, file_paths, test_size=0.3, random_state=42
)

# ================= 5. 训练随机森林模型 =================
print("开始训练AI模型...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train) # 喂给AI的过程

# ================= 6. 评估与混淆矩阵 =================
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎉 训练完成！AI 的准确率为: {acc * 100:.2f}%")

print("\n正在生成混淆矩阵...")
cm = confusion_matrix(y_test, y_pred)
print("原始混淆矩阵数据：")
print(cm)

# 画图
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['噪音(0)', '无人机(1)'])
disp.plot(cmap='Blues')
plt.title("AI 分类混淆矩阵")
plt.show()

# ================= 7. 排查漏报的无人机 =================
print("\n=== 开始排查漏报的无人机（真实是1，预测是0） ===")
# 找出真实标签是 1（无人机），但预测成 0（噪音）的索引
error_indices = np.where((y_test == 1) & (y_pred == 0))[0]

if len(error_indices) > 0:
    print(f"一共发现了 {len(error_indices)} 个漏报的无人机，文件如下：")
    for idx in error_indices:
        print(f"漏报文件路径: {paths_test[idx]}")
else:
    print("太棒了！没有任何漏报。")
        # 把这 80 个路径写入一个 txt 文件
    with open('漏报的80个文件.txt', 'w', encoding='utf-8') as f:
        for idx in error_indices:
            f.write(paths_test[idx] + '\n')
    print("已生成 漏报的80个文件.txt，请去代码所在文件夹查看！")