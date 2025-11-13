import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from NIOModules import NIOFP
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Dataset for the parameter prediction task
# -----------------------------
class ParameterDataset(Dataset):
    def __init__(self, file_path):
        """
        加载数据并对 y, V, g, kappa 做缩放（除以整体最大值，不减均值）
        """
        data = np.load(file_path, allow_pickle=True).item()
        self.y = data["y"]         # shape: (num_samples, num_time_samples, 128)
        self.g = data["g"]         # shape: (num_samples,)
        self.kappa = data["kappa"] # shape: (num_samples,)
        self.V = data["V"]         # shape: (num_samples, 128)
     
        
        # 计算整体最大值（全局标量）
        self.y_max = self.y.max()
        self.V_max = self.V.max()
        self.g_max = self.g.max()
        self.kappa_max = self.kappa.max()
        print("Scaling factors:")
        print("y_max:", self.y_max, "V_max:", self.V_max, "g_max:", self.g_max, "kappa_max:", self.kappa_max)
        
        # 缩放（除以最大值）
        self.y = self.y / self.y_max
        self.V = self.V / self.V_max
        self.g = self.g / self.g_max
        self.kappa = self.kappa / self.kappa_max

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # 输入为轨道 y，形状 (num_time_samples, 128)
        trajectory = self.y[idx]  # (T, 128)
        # 目标参数：构造一个 (128,3) 的张量，
        # 第一列为 V(x)（已有形状 (128,)），
        # 第二列为 g (扩展为 (128,))，
        # 第三列为 kappa (扩展为 (128,))
        V_target = self.V[idx]  # (128,)
        g_target = np.full((V_target.shape[0],), self.g[idx])
        kappa_target = np.full((V_target.shape[0],), self.kappa[idx])
        target = np.stack([V_target, g_target, kappa_target], axis=-1)  # (128, 3)
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return trajectory, target

# -----------------------------
# Training code
# -----------------------------
# Initialize dataset and dataloaders
file_path = "training_data_Vk.npy"
batch_size = 32
dataset = ParameterDataset(file_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model参数设置（注意输出维度改为3）
input_dimensions_trunk = 1  # 根据你模型的设计
n_hidden_layers = 3
neurons = 100
n_basis = 25
fno_layers = 4
width = 25
modes = 32
output_dim = 3  # 输出三个通道：V, g, kappa
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NIOFP(
    input_dimensions_trunk,
    n_hidden_layers,
    neurons,
    n_basis,
    fno_layers,
    width,
    modes,
    output_dim,
    device
).to(device)

# 定义loss和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练设置
num_epochs = 1000
save_interval = 10
os.makedirs("results_vk", exist_ok=True)
# 定义空间网格（与原始空间网格[-10,10]对应，这里归一化到[0,1]供网络使用）
grid = torch.linspace(0, 1, 128).unsqueeze(-1).to(device)
test_loss_curve = []
loss_V_curve = []
loss_g_curve = []
loss_kappa_curve = []

# Training loop
for epoch in tqdm(range(1, num_epochs + 1)):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        # inputs: shape (batch, num_time_samples, 128)
        # targets: shape (batch, 128, 3)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs, grid)  # 输出 shape: (batch, 128, 3)
        # 对于V通道：直接比较每个网格点
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    if epoch % save_interval == 0:
        model.eval()
        test_loss = 0.0
        total_loss_V = 0.0
        total_loss_g = 0.0
        total_loss_kappa = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs, grid)
                loss_V = criterion(predictions[:, :, 0], targets[:, :, 0])
                loss_g = criterion(predictions[:, :, 1].mean(dim=1), targets[:, 0, 1])
                loss_kappa = criterion(predictions[:, :, 2].mean(dim=1), targets[:, 0, 2])
                loss = loss_V + loss_g + loss_kappa
                test_loss += loss.item() * inputs.size(0)
                total_loss_V += loss_V.item() * batch_size
                total_loss_g += loss_g.item() * batch_size
                total_loss_kappa += loss_kappa.item() * batch_size
                total_samples += batch_size
        test_loss /= len(test_loader.dataset)
        avg_loss_V = total_loss_V / total_samples
        avg_loss_g = total_loss_g / total_samples
        avg_loss_kappa = total_loss_kappa / total_samples
        test_loss_curve.append(test_loss)
        loss_V_curve.append(avg_loss_V)
        loss_g_curve.append(avg_loss_g)
        loss_kappa_curve.append(avg_loss_kappa)

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, ")
        print(f"Test Loss V: {avg_loss_V:.6f},Test Loss g: {avg_loss_g:.6f},Test Loss kappa: {avg_loss_kappa:.6f}")
        # 绘制loss曲线图并保存
        plt.figure(figsize=(8,6))
        epochs = np.arange(save_interval, epoch+1, save_interval)
        plt.plot(epochs, test_loss_curve, label="Overall Test Loss", marker='o')
        plt.plot(epochs, loss_V_curve, label="Test Loss V", marker='o')
        plt.plot(epochs, loss_g_curve, label="Test Loss g", marker='o')
        plt.plot(epochs, loss_kappa_curve, label="Test Loss κ", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Test Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig("loss_curve.png")
        plt.close()
        # 可视化一个测试样本
        model.eval()
        sample_idx = np.random.randint(0, len(test_dataset))
        sample_input, sample_target = test_dataset[sample_idx]
        sample_input = sample_input.unsqueeze(0).to(device)  # (1, num_time_samples, 128)
        sample_target = sample_target.unsqueeze(0).to(device)  # (1, 128, 3)
        with torch.no_grad():
            sample_pred = model(sample_input, grid)  # (1, 128, 3)
        
        # 由于保存时我们除以最大值，现在进行反缩放
        # 假设最大值信息存储在 dataset 中：
        V_max = dataset.V_max
        g_max = dataset.g_max
        kappa_max = dataset.kappa_max
        
        pred_V = sample_pred[:, :, 0] * V_max
        pred_g = sample_pred[:, :, 1].mean(axis=1, keepdim=True) * g_max
        pred_kappa = sample_pred[:, :, 2].mean(axis=1, keepdim=True) * kappa_max
        pred_g = pred_g.repeat(1,128)
        pred_kappa = pred_kappa.repeat(1,128)
        true_V = sample_target[:, :, 0] * V_max
        true_g = sample_target[:, :, 1] * g_max
        true_kappa = sample_target[:, :, 2] * kappa_max
       
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(true_V[0].cpu().numpy(), label="True V", linewidth=2)
        plt.plot(pred_V[0].cpu().numpy(), label="Predicted V", linewidth=2)
        plt.title("V(x)")
        plt.legend()
        plt.grid()
        
        plt.subplot(1, 3, 2)
        plt.plot(np.linspace(-10,10,128),true_g[0].cpu().numpy(), label="True g", linewidth=2)
        plt.plot(np.linspace(-10,10,128),pred_g[0].cpu().numpy(), label="Predicted g", linewidth=2)
        plt.title("g (mean prediction)")
        plt.legend()
        plt.grid()
        
        plt.subplot(1, 3, 3)
        plt.plot(np.linspace(-10,10,128),true_kappa[0].cpu().numpy(), label="True κ", linewidth=2)
        plt.plot(np.linspace(-10,10,128),pred_kappa[0].cpu().numpy(), label="Predicted κ", linewidth=2)
        plt.title("κ (mean prediction)")
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(f"results_vk/{epoch}_pred_vs_true.png")
        plt.close()
