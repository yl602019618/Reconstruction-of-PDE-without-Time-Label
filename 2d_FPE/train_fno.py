import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from NIOModules import NIOFP2D, NIOFP2D_FNO, NIOFP2D_FNO_attn

# Dataset class for 2D Fokker-Planck data
class TrajectoryDataset2D(Dataset):
    def __init__(self, file_path):
        """
        Initialize the dataset, loading data and performing normalization.

        Args:
        - file_path (str): Path to the .npz dataset file.
        """
        # Load dataset
        data = np.load(file_path, allow_pickle=True)
        self.trajectories = np.array(data["trajectories"], dtype=np.float32)*1e10  # Shape: (M, 100, Nx, Ny)
        self.potential = np.array(data["potential"], dtype=np.float32)*1e21        # Shape: (M, Nx, Ny)
        self.drag = np.array(data["drag"], dtype=np.float32)*1e6      

        # Compute normalization parameters for trajectories
        self.trajectories_mean = self.trajectories.mean(axis=(0, 1), keepdims=True)
        self.trajectories_std = self.trajectories.std(axis=(0, 1), keepdims=True) + 1e-8

        # Compute normalization parameters for potential
        self.potential_mean = self.potential.mean(axis=(0), keepdims=True)
        self.potential_std = self.potential.std(axis=(0), keepdims=True) + 1e-8
        self.drag_mean = self.drag.mean(axis=(0), keepdims=True)
        self.drag_std = self.drag.std(axis=(0), keepdims=True)+1e-8 
        
        # Standardize data
        self.trajectories = (self.trajectories - self.trajectories_mean) / self.trajectories_std
        self.potential = (self.potential - self.potential_mean) / self.potential_std
        self.drag =  (self.drag - self.drag_mean) / self.drag_std
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
        - idx (int): Index of the sample.

        Returns:
        - input_trajectory (torch.Tensor): Shape (100, Nx, Ny)
        - output_potential (torch.Tensor): Shape (Nx, Ny)
        """
        input_trajectory = torch.tensor(self.trajectories[idx], dtype=torch.float32)  # Shape: (100, Nx, Ny)
        output_potential = torch.tensor(self.potential[idx], dtype=torch.float32).unsqueeze(-1)     # Shape: (Nx, Ny)
        output_drag = torch.tensor(self.drag[idx], dtype=torch.float32).unsqueeze(-1)

        output = torch.cat((output_potential , output_drag), axis=2)
       
        return input_trajectory, output

# ------------------------ 数据加载 ------------------------
file_path = "/home/ubuntu/unlabelPDE_official/2d_diffusion/dataset_2D_drift_diffusion.npz"
batch_size = 4

dataset = TrajectoryDataset2D(file_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------ 创建 Accelerator ------------------------
accelerator = Accelerator(
    kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
)
seed = 1
np.random.seed(seed + accelerator.process_index)
torch.manual_seed(seed + accelerator.process_index)
torch.cuda.manual_seed_all(seed + accelerator.process_index)
# ------------------------ 模型和训练参数 ------------------------

nx = 61
ny = 61
input_dimensions_trunk = 2
n_hidden_layers = 3
neurons = 100
n_basis = 25
fno_layers = 3
width = 12
modes = 32
output_dim = 2

model = NIOFP2D_FNO(input_dimensions_trunk,
                    n_hidden_layers,
                    neurons,
                    n_basis,
                    fno_layers,
                    width,
                    modes,
                    output_dim)

# 定义设备（accelerator 会自动选择合适的设备）
device = accelerator.device
model = model.to(device)

# 定义网格信息
Nx, Ny = 61, 61
grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, Nx, dtype=np.float32),
                             np.linspace(-1, 1, Ny, dtype=np.float32),
                             indexing="ij")
grid = np.stack([grid_x, grid_y], axis=2)
grid = torch.tensor(grid, dtype=torch.float32).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# 使用 accelerator.prepare 包装模型、优化器、数据加载器和 scheduler
model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, test_loader, scheduler
)

# ------------------------ 训练与测试 ------------------------
num_epochs = 400
save_interval = 5
os.makedirs("result_2d_fno", exist_ok=True)
train_losses = []
test_losses = []
test_losses_drift = []
test_losses_diffusion = []
best_test_loss = float('inf')
best_ckpt_path = None  # 用于保存当前最佳 ckpt 文件路径
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    # 训练循环
    for inputs, outputs in tqdm(train_loader, desc=f"Epoch {epoch} Training", disable=not accelerator.is_local_main_process):
        optimizer.zero_grad()
        predictions = model(inputs, grid)
        loss = criterion(predictions, outputs)
        accelerator.backward(loss)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    scheduler.step()
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    if epoch % save_interval == 0:
        model.eval()
        test_loss = 0.0
        test_loss_drift = 0.0
        test_loss_diffusion = 0.0
        # 测试循环
        for inputs, outputs in tqdm(test_loader, desc=f"Epoch {epoch} Testing", disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                predictions = model(inputs, grid)
                # 计算归一化后的相对误差
                errors_drift = torch.norm((predictions[...,0] - outputs[...,0]).view(predictions[...,0].size(0), -1), dim=1) / \
                               torch.norm(outputs.view(outputs[...,0].size(0), -1), dim=1)
                errors_diffusion = torch.norm((predictions[...,1] - outputs[...,1]).view(predictions[...,1].size(0), -1), dim=1) / \
                                   torch.norm(outputs.view(outputs[...,1].size(0), -1), dim=1)
                test_loss_drift += errors_drift.sum().item()
                test_loss_diffusion += errors_diffusion.sum().item()
        test_loss = (test_loss_drift + test_loss_diffusion) / len(test_loader.dataset)
        test_loss_drift = test_loss_drift / len(test_loader.dataset)
        test_loss_diffusion = test_loss_diffusion / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_losses_drift.append(test_loss_drift)
        test_losses_diffusion.append(test_loss_diffusion)
        
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
                  f"Test Loss drift: {test_loss_drift:.6f}, Test Loss diffusion: {test_loss_diffusion:.6f}")
            
            # 绘制 loss 曲线图
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
            plt.plot(range(save_interval, save_interval * len(test_losses) + 1, save_interval), test_losses, label="Test Loss", marker='s')
            plt.plot(range(save_interval, save_interval * len(test_losses) + 1, save_interval), test_losses_drift, label="Test Loss drift", marker='.')
            plt.plot(range(save_interval, save_interval * len(test_losses) + 1, save_interval), test_losses_diffusion, label="Test Loss diffusion", marker='.')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Test Loss Curve")
            plt.legend()
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(f"result_2d_fno/loss_curve.png")
            plt.close()

            # 如果当前 test_loss 低于历史最低值，则保存模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                best_ckpt_path = f"result_2d_fno/model_checkpoint_best_{best_test_loss:.6f}.pt"
                accelerator.save(model.state_dict(), best_ckpt_path)
                print(f"New best model saved with Test Loss {best_test_loss:.6f} ")

            # 随机选取 3 个测试样本进行预测和绘图
            test_indices = torch.randint(0, len(test_dataset), (3,))
            test_inputs = torch.stack([test_dataset[idx][0] for idx in test_indices]).to(device)
            test_outputs = torch.stack([test_dataset[idx][1] for idx in test_indices]).to(device)
            predictions = model(test_inputs, grid)
            
            # 反归一化
            potential_std = torch.tensor(dataset.potential_std, dtype=torch.float32, device=device)
            drag_std = torch.tensor(dataset.drag_std, dtype=torch.float32, device=device)
            true_potential = test_outputs[..., 0] * potential_std
            pred_potential = predictions[..., 0] * potential_std
            true_drag = test_outputs[..., 1] * drag_std
            pred_drag = predictions[..., 1] * drag_std
            
            # 绘制预测结果图（仅展示潜力图）
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            ax = axes[0, 0]
            im = ax.imshow(true_potential[0].cpu().numpy(), cmap="viridis", origin="lower",vmax = 0)
            ax.set_title("True Potential")
            plt.colorbar(im, ax=ax)
            
            ax = axes[0, 1]
            im = ax.imshow(pred_potential[0].detach().cpu().numpy(), cmap="viridis", origin="lower",vmax = 0)
            ax.set_title("Predicted Potential")
            plt.colorbar(im, ax=ax)
            
            ax = axes[1, 0]
            im = ax.imshow(true_potential[1].cpu().numpy(), cmap="viridis", origin="lower",vmax = 0)
            ax.set_title("True Potential")
            plt.colorbar(im, ax=ax)
            
            ax = axes[1, 1]
            im = ax.imshow(pred_potential[1].detach().cpu().numpy(), cmap="viridis", origin="lower",vmax = 0)
            ax.set_title("Predicted Potential")
            plt.colorbar(im, ax=ax)
            
            plt.suptitle(f"Test Predictions at Epoch {epoch}, Test Loss: {test_loss:.6f}")
            plt.savefig(f"result_2d_fno/potential_epoch_{epoch}.png")
            plt.close()
            
            # 绘制预测结果图（仅展示潜力图）
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            ax = axes[0, 0]
            im = ax.imshow(true_drag[0].cpu().numpy(), cmap="viridis", origin="lower",vmax = 0)
            ax.set_title("True Drag")
            plt.colorbar(im, ax=ax)
            
            ax = axes[0, 1]
            im = ax.imshow(pred_drag[0].detach().cpu().numpy(), cmap="viridis", origin="lower",vmax = 0)
            ax.set_title("Predicted Drag")
            plt.colorbar(im, ax=ax)
            
            ax = axes[1, 0]
            im = ax.imshow(true_drag[1].cpu().numpy(), cmap="viridis", origin="lower",vmax = 0)
            ax.set_title("True Drag")
            plt.colorbar(im, ax=ax)
            
            ax = axes[1, 1]
            im = ax.imshow(pred_drag[1].detach().cpu().numpy(), cmap="viridis", origin="lower",vmax = 0)
            ax.set_title("Predicted Drag")
            plt.colorbar(im, ax=ax)
            
            plt.suptitle(f"Test Predictions at Epoch {epoch}, Test Loss: {test_loss:.6f}")
            plt.savefig(f"result_2d_fno/drag_epoch_{epoch}.png")
            plt.close()

np.save("result_2d_fno/train_losses.npy", np.array(train_losses))
np.save("result_2d_fno/test_losses.npy", np.array(test_losses))
np.save("result_2d_fno/test_losses_drift.npy", np.array(test_losses_drift))
np.save("result_2d_fno/test_losses_diffusion.npy", np.array(test_losses_diffusion))
