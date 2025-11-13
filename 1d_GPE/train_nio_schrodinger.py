import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from NIOModules import NIOFP_schrodinger
import random
import numpy as np
import torch
def set_seed(seed=42):
    """固定所有随机种子以实现可复现性"""
    # Python内置随机种子
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 关闭自动优化卷积算法
    
    # 如果使用GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # 适用于TensorFlow 2.x
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# 使用示例
set_seed(2022311356)
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
        self.y_max = self.y.max()/3
        self.V_max = self.V.max()/3
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
       
        target = np.stack([V_target], axis=-1)  # (128, 3)
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return trajectory, target




file_path = "/home/ubuntu/unlabel_PDE_official/1dGPE/training_data_Schrodinger.npy"
batch_size = 32
# Create dataset
dataset = ParameterDataset(file_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
input_dimensions_trunk = 1
n_hidden_layers = 3
neurons = 100
n_basis = 25
fno_layers = 3
fno_layers = 3
width = 10
modes = 30
output_dim = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NIOFP_schrodinger(
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

# Loss function and optimizer
# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
# Training loop
num_epochs = 400
save_interval = 10
train_losses = []
test_loss_curve = []
best_test_loss = float('inf')
best_ckpt_path = None  # 用于保存当前最佳 ckpt 文件路径
os.makedirs("results_schrodinger_nio", exist_ok=True)
grid = torch.linspace(0, 1, 128).unsqueeze(-1).to(device)

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
    scheduler.step()
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    if epoch % save_interval == 0:
        model.eval()
        test_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs, grid)
                errors = torch.norm((predictions - targets).view(predictions.size(0), -1), dim=1) / \
                               torch.norm(targets.view(targets.size(0), -1), dim=1)
                test_loss += errors.sum().item()
                
        test_loss = test_loss / len(test_loader.dataset)
       
        test_loss_curve.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            best_ckpt_path = f"results_schrodinger_nio/model_checkpoint_best_{best_test_loss:.6f}.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved with Test Loss {best_test_loss:.6f} ")

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, ")
        # 绘制loss曲线图并保存
        plt.figure(figsize=(8,6))
        epochs = np.arange(save_interval, epoch+1, save_interval)
        plt.plot(epochs, test_loss_curve, label="Overall Test Loss", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Test Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig("results_schrodinger_nio/loss_curve.png")
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
    
        
        pred_V = sample_pred[:, :, 0] * V_max
    
        true_V = sample_target[:, :, 0] * V_max
      
       
        plt.figure(figsize=(12, 6))

        plt.plot(true_V[0].cpu().numpy(), label="True V", linewidth=2)
        plt.plot(pred_V[0].cpu().numpy(), label="Predicted V", linewidth=2)
        plt.title("V(x)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"results_schrodinger_nio/{epoch}_pred_vs_true.png")
        plt.close()


np.save("results_schrodinger_nio/train_losses.npy", np.array(train_losses))
np.save("results_schrodinger_nio/test_losses.npy", np.array(test_loss_curve))

