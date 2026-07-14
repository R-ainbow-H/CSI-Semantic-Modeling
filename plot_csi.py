import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 修改成你自己的mat文件路径
# ===============================
file = r'./Data/NTU-Fi_HAR/train_amp/box/box0.mat'

# ===============================
# 读取CSI数据
# ===============================
data = sio.loadmat(file)
csi = data["CSIamp"]

print("CSI shape:", csi.shape)

# ===============================
# Figure 1 原始CSI热力图
# ===============================
plt.figure(figsize=(12,6))
plt.imshow(csi, aspect='auto', cmap='jet')
plt.colorbar(label="Amplitude")
plt.title("Original CSI Amplitude")
plt.xlabel("Time")
plt.ylabel("Subcarrier")
plt.tight_layout()

# ===============================
# Figure 2 第1个子载波随时间变化
# ===============================
plt.figure(figsize=(12,4))
plt.plot(csi[0])
plt.title("Subcarrier 1")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# ===============================
# Figure 3 第100个子载波
# ===============================
plt.figure(figsize=(12,4))
plt.plot(csi[99])
plt.title("Subcarrier 100")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# ===============================
# Figure 4 所有子载波平均值
# ===============================
mean_signal = np.mean(csi, axis=0)

plt.figure(figsize=(12,4))
plt.plot(mean_signal)
plt.title("Mean CSI Amplitude")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()