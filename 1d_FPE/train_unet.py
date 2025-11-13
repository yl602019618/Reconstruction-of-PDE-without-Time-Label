import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from NIOModules import NIOFP, NIOFP_FNO, PermInvUNet_attn1D
class TrajectoryDataset1D(Dataset):
    def __init__(self, file_path):
        """
        Initialize the dataset, loading data and performing normalization.

        Args:
        - file_path (str): Path to the .npz dataset file.
        """
        # Load dataset
        data = np.load(file_path, allow_pickle=True)
        # 1e2
        self.trajectories = np.array(data["trajectories"], dtype=np.float32)*1e5  # Shape: (M, 100, Nx)
        self.potential = np.array(data["potential"], dtype=np.float32)*1e20       # Shape: (M, Nx)
        self.drag = np.array(data["drag"], dtype=np.float32)*1e5      
        self.drag = self.drag[:,np.newaxis]  # Shape: (M,1)
        # Compute normalization parameters for trajectories
        self.trajectories_mean = self.trajectories.mean(axis=(0, 1), keepdims=True)
        self.trajectories_std = self.trajectories.std(axis=(0, 1), keepdims=True)+1e-8

        # Compute normalization parameters for potential
        self.potential_mean = self.potential.mean(axis=(0), keepdims=True)
        self.potential_std = self.potential.std(axis=(0), keepdims=True) +1e-8

        self.drag_mean = self.drag.mean(axis=(0), keepdims=True)
        self.drag_std = self.drag.std(axis=(0), keepdims=True)+1e-8 
        # Standardize data

        self.trajectories = (self.trajectories - self.trajectories_mean) / self.trajectories_std
        self.potential = (self.potential - self.potential_mean) / self.potential_std
        self.drag = (self.drag - self.drag_mean) / self.drag_std
        
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
        input_trajectory = torch.tensor(self.trajectories[idx], dtype=torch.float32)  # Shape: (100, Nx)
        output_potential = torch.tensor(self.potential[idx], dtype=torch.float32)     # Shape: (Nx)
        output_drag = torch.tensor(self.drag[idx], dtype=torch.float32).repeat(output_potential.shape[0])     # Shape: (1)
        output = np.stack((output_potential , output_drag), axis=1)
        return input_trajectory, output

file_path = "/home/ubuntu/unlabel_PDE_official/1d/dataset_1D_drift_diffusion.npz"
batch_size = 32
# Create dataset
dataset = TrajectoryDataset1D(file_path)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
fno_layers = 3
width = 10
modes = 15
output_dim = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PermInvUNet_attn1D(in_ch=1, out_ch=2, base_ch=1, depth=6, input_size=80,device = device).to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
# Training loop
num_epochs = 400
save_interval = 10
train_losses = []
test_losses = []
test_losses_drift = []
test_losses_diffusion = []
best_test_loss = float('inf')
best_ckpt_path = None  # 用于保存当前最佳 ckpt 文件路径
os.makedirs("results_unet", exist_ok=True)
grid = torch.linspace(0, 1, 80).unsqueeze(-1).to(device)
for epoch in tqdm(range(1, num_epochs + 1)):
    model.train()
    train_loss = 0.0
    for inputs, outputs in train_loader:
        inputs, outputs = inputs.to(device), outputs.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
      
        loss = criterion(predictions, outputs) 
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    scheduler.step()
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    if epoch % save_interval == 0:
        # Test the model
        model.eval()
        test_loss = 0.0
        test_loss_drift = 0.0
        test_loss_diffusion = 0.0
        with torch.no_grad():
            for inputs, outputs in test_loader:
                inputs, outputs = inputs.to(device), outputs.to(device)
                predictions = model(inputs)
                
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
        # Plot losses
        plt.figure()
        #plt.plot(np.arange(len(train_losses)) * save_interval + save_interval, train_losses, label='Train Loss')
        plt.plot(np.arange(len(test_losses)) * save_interval + save_interval, test_losses, label='Test Loss')
        plt.plot(np.arange(len(test_losses_drift)) * save_interval + save_interval, test_losses_drift, label='Drift Loss')
        plt.plot(np.arange(len(test_losses_diffusion)) * save_interval + save_interval, test_losses_diffusion, label='Diffusion Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Testing Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results_unet/loss_curve.png")
        plt.close()
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            best_ckpt_path = f"results_unet/model_checkpoint_best_{best_test_loss:.6f}.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved with Test Loss {best_test_loss:.6f} ")
        # Print training and testing loss
        ids = np.random.randint(32)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
                f"Test Loss drift: {test_loss_drift:.6f}, Test Loss diffusion: {test_loss_diffusion:.6f}")
        pred_potential = predictions[ids][:, 0].detach().cpu().numpy()
        true_potential =outputs[ids][:, 0].detach().cpu().numpy()
        pred_drag = predictions[ids][:, 1].detach().cpu().numpy()
        true_drag = outputs[ids][:, 1].detach().cpu().numpy()
        grid_np = grid.cpu().numpy().squeeze()  # shape: (80,)
        plt.figure()
        plt.plot(grid_np, true_potential, label="True Potential")
        plt.plot(grid_np, pred_potential, label="Predicted Potential", linestyle="--")
        plt.xlabel("Grid")
        plt.ylabel("Potential")
        plt.title("Potential Prediction vs Ground Truth")
        plt.legend()
        plt.savefig(f"results_unet/potential_prediction_{epoch}.png")  # 保存 potential 图
        plt.close()
        # 绘制 drag 的对比图
        plt.figure()
        plt.plot(grid_np, true_drag, label="True Drag")
        plt.plot(grid_np, np.ones_like(grid_np)*pred_drag.mean(), label="Predicted Drag", linestyle="--")
        plt.xlabel("Grid")
        plt.ylabel("Drag")
        plt.title("Drag Prediction vs Ground Truth")
        plt.ylim(-2,2)
        plt.legend()
        plt.savefig(f"results_unet/drag_prediction_{epoch}.png")  # 保存 drag 图
        plt.close()
        
# Save numpy arrays
np.save("results_unet/train_losses.npy", np.array(train_losses))
np.save("results_unet/test_losses.npy", np.array(test_losses))
np.save("results_unet/test_losses_drift.npy", np.array(test_losses_drift))
np.save("results_unet/test_losses_diffusion.npy", np.array(test_losses_diffusion))
