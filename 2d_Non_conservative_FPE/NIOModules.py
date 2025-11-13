import numpy as np
import torch
import torch.nn as nn

from Baselines import Encoder,EncoderHelm2,Encoder2D, Encoder_ode, Encoder3D,Encoder3D_down
from DeepONetModules import FeedForwardNN, DeepOnetNoBiasOrg, FFN
from FNOModules import FNO1d,FNO2d,FNO3d
from model import Transolver_Irregular_Mesh, Transolver_Structured_Mesh_2D, Transolver_Structured_Mesh_3D
import math
import torch.nn.functional as F
################################################################

class NIOFP2D(nn.Module):
    def __init__(self,
                 input_dimensions_trunk,
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim):
        super(NIOFP2D, self).__init__()
        output_dimensions = n_basis
        # self.fno_inputs = fno_input_dimension
        self.trunk = FFN(input_dimensions_trunk, 
                                    output_dimensions, 
                                    n_hidden_layers, 
                                    neurons, 
                                    "leaky_relu", 
                                    0.0)
        self.fno_layers = fno_layers
        self.branch = Encoder2D(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)
        self.fc0 = nn.Linear(3, width)
        self.fno_Fx = FNO2d(modes=modes,
                               width=width,
                               n_layers=self.fno_layers,
                               input_dim=width,
                               output_dim=1)
        self.fno_Fy = FNO2d(modes=modes,
                               width=width,
                               n_layers=self.fno_layers,
                               input_dim=width,
                               output_dim=1)           
    def forward(self, x, grid):
        '''
        x: batch,L = 100, Nx = 100, Ny = 100
        grid: Nx,Ny,2
        '''
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]

        nx = (grid.shape[0])
        ny = (grid.shape[1])

        grid_r = grid.view(-1,2)
        x = self.deeponet(x.unsqueeze(2), grid_r)
       
        x = x.view(x.shape[0], x.shape[1], nx, ny) # batch, L ,Nx, Ny
        
        grid = grid.unsqueeze(0).repeat(x.shape[0],1,1,1).permute(0,3,1,2)# 

        x = torch.cat((grid, x), 1)
        
        weight_trans_mat = self.fc0.weight.data
        bias_trans_mat = self.fc0.bias.data
        
        weight_trans_mat = torch.cat([weight_trans_mat[:, :2], weight_trans_mat[:, 2].view(-1, 1).repeat(1, L) / L], dim=1)   # width, L+2
        x = x.permute(0, 2, 3, 1)
        
        x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat # B,3,x,y
  
        fx = self.fno_Fx(x)
        fy = self.fno_Fy(x)
        x_out = torch.cat((fx,fy),dim = -1)  
        
        return x_out

class NIOFP2D_Trans(nn.Module):
    def __init__(self,
                 input_dimensions_trunk,
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim):
        super(NIOFP2D_Trans, self).__init__()
        output_dimensions = n_basis
        # self.fno_inputs = fno_input_dimension
        
        self.fno_layers = fno_layers
        self.branch = Encoder2D(output_dimensions)
       
        self.fc0 = nn.Linear(3, width)


        self.trans_input = Transolver_Structured_Mesh_2D.Model(space_dim=2,
                                                                    n_layers=3,
                                                                    n_hidden=32,
                                                                    dropout=0.0,
                                                                    n_head=4,
                                                                    Time_Input=False,
                                                                    mlp_ratio=1,
                                                                    fun_dim=1,
                                                                    out_dim=1,
                                                                    slice_num=16,
                                                                    ref=8,
                                                                    unified_pos=0,
                                                                    H=61, W=61)

        self.fno_drift = FNO2d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = 1)
        self.fno_diffusion = FNO2d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = 1)            
    def forward(self, x, grid):
        '''
        x: batch,L = 100, Nx = 100, Ny = 100
        grid: Nx,Ny,2
        '''
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]
        batchsize = x.shape[0]
        x_input = x.reshape(x.shape[0]*x.shape[1],x.shape[2]*x.shape[3],1) # B*L, nx*ny,1
        nx = (grid.shape[0])# nx,ny,2
        ny = (grid.shape[1])# nx,ny,2
        grid_r = grid.unsqueeze(0).repeat(x_input.shape[0],1,1,1).reshape(x_input.shape[0],-1,2) ## B*L,nx*ny,2 
        
        
        x = self.trans_input(x_input,grid_r) # input 3 dim output 1 dim  # B*L,61,61,1
       
        x = x.view(batchsize, L, nx, ny) # batch, L ,Nx, Ny
        grid = grid.unsqueeze(0).repeat(x.shape[0],1,1,1).permute(0,3,1,2)# 

        x = torch.cat((grid, x), 1) # b,L,61,61
  
        weight_trans_mat = self.fc0.weight.data
        bias_trans_mat = self.fc0.bias.data
        
        weight_trans_mat = torch.cat([weight_trans_mat[:, :2], weight_trans_mat[:, 2].view(-1, 1).repeat(1, L) / L], dim=1)   # width, L+2
        x = x.permute(0, 2, 3, 1) 

        x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat # B,3,x,y
  
        potential = self.fno_drift(x)
        drag = self.fno_diffusion(x)
        x = torch.cat((potential,drag),dim = -1)
        
        return x


class NIOFP2D_Trans_attn(nn.Module):
    def __init__(self,
                 input_dimensions_trunk,
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim,
                 nx, 
                 ny):
        super(NIOFP2D_Trans_attn, self).__init__()
        output_dimensions = n_basis
        # self.fno_inputs = fno_input_dimension
        
        self.fno_layers = fno_layers
        self.branch = Encoder2D(output_dimensions)
       
        self.fc0 = nn.Linear(3, width)


        self.trans_input = Transolver_Structured_Mesh_2D.Model(space_dim=2,
                                                                    n_layers=3,
                                                                    n_hidden=32,
                                                                    dropout=0.0,
                                                                    n_head=4,
                                                                    Time_Input=False,
                                                                    mlp_ratio=1,
                                                                    fun_dim=1,
                                                                    out_dim=1,
                                                                    slice_num=16,
                                                                    ref=8,
                                                                    unified_pos=0,
                                                                    H=61, W=61)

        self.fno_drift = FNO2d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = 1)
        self.fno_diffusion = FNO2d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = 1)   
        # fc0 用于后续 token 加权融合（原始定义）
        self.fc0 = nn.Linear(3, width)
        # 从输入尺寸得到网格大小
        self.nx = nx
        self.ny = ny         
    def forward(self, x, grid):
        '''
        x: batch,L = 100, Nx = 100, Ny = 100
        grid: Nx,Ny,2
        '''
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]
        batchsize = x.shape[0]
        x_input = x.reshape(x.shape[0]*x.shape[1],x.shape[2]*x.shape[3],1) # B*L, nx*ny,1
        nx = (grid.shape[0])# nx,ny,2
        ny = (grid.shape[1])# nx,ny,2
        grid_r = grid.unsqueeze(0).repeat(x_input.shape[0],1,1,1).reshape(x_input.shape[0],-1,2) ## B*L,nx*ny,2 
        
        
        x_proc = self.trans_input(x_input,grid_r) # input 3 dim output 1 dim  # B*L,61,61,1
        x_proc = x_proc.view(batchsize, L, nx, ny)
        
        # --- 2. 拼接 grid 与 x_proc 得到 token 序列 ---
        # 将 grid 扩展到 batch 维度，shape 变为 (B, 2, nx, ny)
        grid_rep = grid.unsqueeze(0).repeat(batchsize, 1, 1, 1).permute(0, 3, 1, 2)
        # 拼接后 x_tokens 的形状为 (B, T, nx, ny)，其中 T = L + 2
        x_tokens = torch.cat((grid_rep, x_proc), dim=1)
        B, T, nx, ny = x_tokens.shape  # T = L + 2
        
        # --- 3. 对 x_tokens 做自注意力（Q=K=V） ---
        # 将每个 token 展平为向量，维度 d = nx*ny
        d = nx * ny
        x_flat = x_tokens.view(B, T, d)   # (B, T, d)
        scale = math.sqrt(d)
        # 计算注意力得分： (B, T, T)
        scores = torch.matmul(x_flat, x_flat.transpose(1, 2)) / scale
        attn_weights = F.softmax(scores, dim=-1)
        # 计算自注意力输出 Z，形状 (B, T, d)
        Z = torch.matmul(attn_weights, x_flat)
        # Z 的形状为 (B, T, nx*ny)
        
        # --- 4. 利用原来的方法融合 token ---
        # 先恢复为 (B, T, nx, ny)
        Z = Z.view(B, T, nx, ny)
        # 从 fc0 获取权重和偏置（fc0 定义为 nn.Linear(3, width)）
        weight_trans_mat = self.fc0.weight.data  # shape: (width, 3)
        bias_trans_mat = self.fc0.bias.data        # shape: (width,)
        # T = L + 2，故 L = T - 2
        L_val = T - 2
        # 构造新的变换矩阵，形状为 (width, T)：
        # 前两列直接取 fc0.weight 的前两列，
        # 第三列取 fc0.weight 的第3列重复 L 次，并除以 L
        weight_trans_mat_new = torch.cat([
            weight_trans_mat[:, :2],
            weight_trans_mat[:, 2].view(-1, 1).repeat(1, L_val) / L_val
        ], dim=1)  # (width, T)
        # 将 Z 转置为 (B, nx, ny, T)
        Z_permuted = Z.permute(0, 2, 3, 1)
        # 进行加权融合，得到 (B, nx, ny, width)
        fused = torch.matmul(Z_permuted, weight_trans_mat_new.T) + bias_trans_mat
        # x = x.view(batchsize, L, nx, ny) # batch, L ,Nx, Ny
        # grid = grid.unsqueeze(0).repeat(x.shape[0],1,1,1).permute(0,3,1,2)# 

        # x = torch.cat((grid, x), 1) # b,L,61,61
  
        # weight_trans_mat = self.fc0.weight.data
        # bias_trans_mat = self.fc0.bias.data
        
        # weight_trans_mat = torch.cat([weight_trans_mat[:, :2], weight_trans_mat[:, 2].view(-1, 1).repeat(1, L) / L], dim=1)   # width, L+2
        # x = x.permute(0, 2, 3, 1) 

        # x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat # B,3,x,y
  
        potential = self.fno_drift(fused)
        drag = self.fno_diffusion(fused)
        x = torch.cat((potential,drag),dim = -1)
        
        return x



class NIOFP2D_FNO_attn(nn.Module):
    def __init__(self,
                 input_dimensions_trunk,  # (nx, ny)
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim,
                 nx,ny):
        super(NIOFP2D_FNO_attn, self).__init__()
        output_dimensions = n_basis
        self.fno_layers = fno_layers
  


        self.FNO_input = FNO2d(modes=12,
                               width=4,
                               n_layers=2,
                               input_dim=3,
                               output_dim=1)
        self.fno_Fx = FNO2d(modes=modes,
                               width=width,
                               n_layers=self.fno_layers,
                               input_dim=width,
                               output_dim=1)
        self.fno_Fy = FNO2d(modes=modes,
                               width=width,
                               n_layers=self.fno_layers,
                               input_dim=width,
                               output_dim=1)   
        # fc0 用于后续 token 加权融合（原始定义）
        self.fc0 = nn.Linear(1, width)
        # 从输入尺寸得到网格大小
        self.nx = nx
        self.ny = ny
        
    def forward(self, x, grid):
        """
        x: tensor, shape (B, L, nx, ny)    L 为可变 token 数量
        grid: tensor, shape (nx, ny, 2)
        """
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L, replace=False)
            x = x[:, idx]
        else:
            L = x.shape[1]
        batchsize = x.shape[0]
        
        # --- 1. 通过 FNO_input 处理 x ---
        # 将 x reshape 为 (B*L, 1, nx, ny)
        x_input = x.reshape(batchsize * L, 1, x.shape[2], x.shape[3])
        nx = grid.shape[0]
        ny = grid.shape[1]
        # 将 grid 从 (nx, ny, 2) 转换为 (B*L, 2, nx, ny)
        grid_r = grid.permute(2, 0, 1).unsqueeze(0).repeat(x_input.shape[0], 1, 1, 1)
        # 拼接 x_input 与 grid 得到 (B*L, 3, nx, ny)，再转为 (B*L, nx, ny, 3)
        inp = torch.cat((x_input, grid_r), dim=1).permute(0, 2, 3, 1)
        # FNO_input 处理，输出 (B*L, nx, ny, 1)
        x_proc = self.FNO_input(inp)
        # reshape 回 (B, L, nx, ny)
        x_proc = x_proc.view(batchsize, L, nx, ny)
        
        # --- 2. 拼接 grid 与 x_proc 得到 token 序列 ---
        # 将 grid 扩展到 batch 维度，shape 变为 (B, 2, nx, ny)
        grid_rep = grid.unsqueeze(0).repeat(batchsize, 1, 1, 1).permute(0, 3, 1, 2)
        # 拼接后 x_tokens 的形状为 (B, T, nx, ny)，其中 T = L + 2
        x_tokens = torch.cat((grid_rep, x_proc), dim=1)
        B, T, nx, ny = x_tokens.shape  # T = L + 2
        
        # --- 3. 对 x_tokens 做自注意力（Q=K=V） ---
        # 将每个 token 展平为向量，维度 d = nx*ny
        d = nx * ny
        x_flat = x_tokens.view(B, T, d)   # (B, T, d)
        scale = math.sqrt(d)
        # 计算注意力得分： (B, T, T)
        scores = torch.matmul(x_flat, x_flat.transpose(1, 2)) / scale
        attn_weights = F.softmax(scores, dim=-1)
        # 计算自注意力输出 Z，形状 (B, T, d)
        Z = torch.matmul(attn_weights, x_flat)
        # Z 的形状为 (B, T, nx*ny)
        
        # --- 4. 利用原来的方法融合 token ---
        # 先恢复为 (B, T, nx, ny)
        Z = Z.view(B, T, nx, ny)
        # 从 fc0 获取权重和偏置（fc0 定义为 nn.Linear(3, width)）
        weight_trans_mat = self.fc0.weight.data  # shape: (width, 3)
        bias_trans_mat = self.fc0.bias.data        # shape: (width,)
        # T = L + 2，故 L = T - 2
        L_val = T 
        # 构造新的变换矩阵，形状为 (width, T)：
        # 前两列直接取 fc0.weight 的前两列，
        # 第三列取 fc0.weight 的第3列重复 L 次，并除以 L
        weight_trans_mat_new = weight_trans_mat[:, 0:1].view(-1, 1).repeat(1, L_val) / L_val
        # 将 Z 转置为 (B, nx, ny, T)
        Z_permuted = Z.permute(0, 2, 3, 1)
        # 进行加权融合，得到 (B, nx, ny, width)
        fused = torch.matmul(Z_permuted, weight_trans_mat_new.T) + bias_trans_mat
        # 转换为 (B, width, nx, ny)
        fx = self.fno_Fx(fused)
        fy = self.fno_Fy(fused)
        x_out = torch.cat((fx,fy),dim = -1)  
        return x_out

        
class NIOFP2D_attn(nn.Module):
    def __init__(self,
                 input_dimensions_trunk,
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim):
        super(NIOFP2D_attn, self).__init__()
        output_dimensions = n_basis
        # self.fno_inputs = fno_input_dimension
        self.trunk = FFN(input_dimensions_trunk, 
                                    output_dimensions, 
                                    n_hidden_layers, 
                                    neurons, 
                                    "leaky_relu", 
                                    0.0)
        self.fno_layers = fno_layers
        self.branch = Encoder2D(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)
        self.fc0 = nn.Linear(1, width)
        self.fno_Fx = FNO2d(modes=modes,
                               width=width,
                               n_layers=self.fno_layers,
                               input_dim=width,
                               output_dim=1)
        self.fno_Fy = FNO2d(modes=modes,
                               width=width,
                               n_layers=self.fno_layers,
                               input_dim=width,
                               output_dim=1)           
    def forward(self, x, grid):
        '''
        x: batch,L = 100, Nx = 100, Ny = 100
        grid: Nx,Ny,2
        '''
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]

        nx = (grid.shape[0])
        ny = (grid.shape[1])
        B = x.shape[0]
        grid_r = grid.view(-1,2)
        x = self.deeponet(x.unsqueeze(2), grid_r)
       
        x_proc = x.view(x.shape[0], x.shape[1], nx, ny) # batch, L ,Nx, Ny
        
        grid_rep = grid.unsqueeze(0).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)
        # 拼接后 x_tokens 的形状为 (B, T, nx, ny)，其中 T = L + 2
        x_tokens = torch.cat((grid_rep, x_proc), dim=1)
        B, T, nx, ny = x_tokens.shape  # T = L + 2
        
        # --- 3. 对 x_tokens 做自注意力（Q=K=V） ---
        # 将每个 token 展平为向量，维度 d = nx*ny
        d = nx * ny
        x_flat = x_tokens.view(B, T, d)   # (B, T, d)
        scale = math.sqrt(d)
        # 计算注意力得分： (B, T, T)
        scores = torch.matmul(x_flat, x_flat.transpose(1, 2)) / scale
        attn_weights = F.softmax(scores, dim=-1)
        # 计算自注意力输出 Z，形状 (B, T, d)
        Z = torch.matmul(attn_weights, x_flat)
        # Z 的形状为 (B, T, nx*ny)
        
        # --- 4. 利用原来的方法融合 token ---
        # 先恢复为 (B, T, nx, ny)
        Z = Z.view(B, T, nx, ny)
        # 从 fc0 获取权重和偏置（fc0 定义为 nn.Linear(3, width)）
        weight_trans_mat = self.fc0.weight.data  # shape: (width, 3)
        bias_trans_mat = self.fc0.bias.data        # shape: (width,)
        # T = L + 2，故 L = T - 2
        L_val = T 
        # 构造新的变换矩阵，形状为 (width, T)：
        # 前两列直接取 fc0.weight 的前两列，
        # 第三列取 fc0.weight 的第3列重复 L 次，并除以 L
        weight_trans_mat_new = weight_trans_mat[:, 0:1].view(-1, 1).repeat(1, L_val) / L_val
        # 将 Z 转置为 (B, nx, ny, T)
        Z_permuted = Z.permute(0, 2, 3, 1)
        # 进行加权融合，得到 (B, nx, ny, width)
        fused = torch.matmul(Z_permuted, weight_trans_mat_new.T) + bias_trans_mat
        # 转换为 (B, width, nx, ny)
        

        fx = self.fno_Fx(fused)
        fy = self.fno_Fy(fused)
        x_out = torch.cat((fx,fy),dim = -1)  
        
        return x_out



class NIOFP2D_FNO(nn.Module):
    def __init__(self,
                 input_dimensions_trunk,
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim):
        super(NIOFP2D_FNO, self).__init__()
        output_dimensions = n_basis
        # self.fno_inputs = fno_input_dimension
        
        self.fno_layers = fno_layers
        self.branch = Encoder2D(output_dimensions)
       
        self.fc0 = nn.Linear(3, width)


        self.FNO_input = FNO2d(modes = 12,
                    width = 4,
                    n_layers = 2,
                    input_dim  = 3,
                    output_dim = 1)
        self.fno_Fx = FNO2d(modes=modes,
                               width=width,
                               n_layers=self.fno_layers,
                               input_dim=width,
                               output_dim=1)
        self.fno_Fy = FNO2d(modes=modes,
                               width=width,
                               n_layers=self.fno_layers,
                               input_dim=width,
                               output_dim=1)
               
    def forward(self, x, grid):
        '''
        x: batch,L = 100, Nx = 100, Ny = 100
        grid: Nx,Ny,2
        '''
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]
        batchsize = x.shape[0]
        x_input = x.reshape(x.shape[0]*x.shape[1],1,x.shape[2],x.shape[3])
        nx = (grid.shape[0])
        ny = (grid.shape[1])
        grid_r = grid.permute(2,0,1).unsqueeze(0).repeat(x_input.shape[0],1,1,1)
        
        input = torch.cat((x_input,grid_r),dim = 1).permute(0,2,3,1)
    
        x = self.FNO_input(input) # input 3 dim output 1 dim  # B*L,61,61,1
       
        x = x.view(batchsize, L, nx, ny) # batch, L ,Nx, Ny
        grid = grid.unsqueeze(0).repeat(x.shape[0],1,1,1).permute(0,3,1,2)# 

        x = torch.cat((grid, x), 1) # b,L,61,61
  
        weight_trans_mat = self.fc0.weight.data
        bias_trans_mat = self.fc0.bias.data
        
        weight_trans_mat = torch.cat([weight_trans_mat[:, :2], weight_trans_mat[:, 2].view(-1, 1).repeat(1, L) / L], dim=1)   # width, L+2
        x = x.permute(0, 2, 3, 1) 

        x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat # B,3,x,y
  
        fx = self.fno_Fx(x)
        fy = self.fno_Fy(x)
        x_out = torch.cat((fx,fy),dim = -1)  
        
        return x_out







class NIOFP(nn.Module):
    def __init__(self,
                 input_dimensions_trunk,
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim,
                 device):
        super(NIOFP, self).__init__()
        output_dimensions = n_basis
        # self.fno_inputs = fno_input_dimension
        self.trunk = FFN(input_dimensions_trunk, 
                                    output_dimensions, 
                                    n_hidden_layers, 
                                    neurons, 
                                    "leaky_relu", 
                                    0.0)
        self.fno_layers = fno_layers
        self.branch = Encoder(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)
        self.fc0 = nn.Linear(2, width)
        self.device = device
        self.fno = FNO1d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = output_dim,
                    device = self.device)
    def forward(self, x, grid):
        '''
        x: batch,L = 100, Nx = 100
        grid: Nx,1
        '''
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]

        nx = (grid.shape[0])

        grid_r = grid
        x = self.deeponet(x, grid_r)

        x = x.view(x.shape[0], x.shape[1], nx) # batch, L ,Nx
        grid = grid.unsqueeze(0).repeat(x.shape[0],1,1).permute(0,2,1)

        x = torch.cat((grid, x), 1)
        weight_trans_mat = self.fc0.weight.data
        bias_trans_mat = self.fc0.bias.data
        
        weight_trans_mat = torch.cat([weight_trans_mat[:, :1], weight_trans_mat[:, 1].view(-1, 1).repeat(1, L) / L], dim=1)   # width, L+2
        x = x.permute(0, 2, 1)

        x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat
        x = self.fno(x)

        return x




class NIOFP_ode(nn.Module):
    def __init__(self,
                 input_dimensions_trunk,
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim,
                 device):
        super(NIOFP_ode, self).__init__()
        output_dimensions = n_basis
        # self.fno_inputs = fno_input_dimension
        self.trunk = FFN(input_dimensions_trunk, 
                                    output_dimensions, 
                                    n_hidden_layers, 
                                    neurons, 
                                    "leaky_relu", 
                                    0.0)
        self.fno_layers = fno_layers
        self.branch = Encoder_ode(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)
        self.fc0 = nn.Linear(2, width)
        self.device = device
        self.fno = FNO1d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = output_dim,
                    device = self.device)
    def forward(self, x, grid):
        '''
        x: batch,L = 200, Nx = 11
        grid: 100,1
        '''
        
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]

        nx = (grid.shape[0])

        grid_r = grid
        x = self.deeponet(x, grid_r)

        x = x.view(x.shape[0], x.shape[1], nx) # batch, L ,Nx
        grid = grid.unsqueeze(0).repeat(x.shape[0],1,1).permute(0,2,1)

        x = torch.cat((grid, x), 1)
        weight_trans_mat = self.fc0.weight.data
        bias_trans_mat = self.fc0.bias.data
        
        weight_trans_mat = torch.cat([weight_trans_mat[:, :1], weight_trans_mat[:, 1].view(-1, 1).repeat(1, L) / L], dim=1)   # width, L+2
        x = x.permute(0, 2, 1)

        x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat
        x = self.fno(x)

        return x
    
    
class NIOFP3D(nn.Module):
    def __init__(self,
                 input_dimensions_trunk,
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim,
                 device,
                 down = False):
        super(NIOFP3D, self).__init__()
        output_dimensions = n_basis
        # self.fno_inputs = fno_input_dimension
        self.trunk = FFN(input_dimensions_trunk, 
                                    output_dimensions, 
                                    n_hidden_layers, 
                                    neurons, 
                                    "leaky_relu", 
                                    0.0)
        self.fno_layers = fno_layers
        if down:
            self.branch = Encoder3D_down(output_dimensions)
        else:
            self.branch = Encoder3D(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)
        self.fc0 = nn.Linear(4, width) 
        self.device = device
        self.fno = FNO3d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = output_dim,
                    device = self.device)
    def forward(self, x, grid):
        '''
        x: batch,L = 100, Nx = 40, Ny = 40, Nz = 40
        grid: Nx,Ny,Nz,3
        '''
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]

        nx = (grid.shape[0])
        ny = (grid.shape[1])
        nz = (grid.shape[2])

        grid_r = grid.view(-1,3)
        x = self.deeponet(x.unsqueeze(2), grid_r)

        x = x.view(x.shape[0], x.shape[1], nx, ny, nz) # batch, L ,Nx, Ny
        grid = grid.unsqueeze(0).repeat(x.shape[0],1,1,1,1).permute(0,4,1,2,3)# 

        x = torch.cat((grid, x), 1)
        weight_trans_mat = self.fc0.weight.data
        bias_trans_mat = self.fc0.bias.data
        
        weight_trans_mat = torch.cat([weight_trans_mat[:, :3], weight_trans_mat[:, 3].view(-1, 1).repeat(1, L) / L], dim=1)   # width, L+2
        x = x.permute(0, 2, 3, 4, 1)
        
        x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat # B,3,x,y
  
        x = self.fno(x)

        return x


class PermInvUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base_ch=1, depth=4, input_size=(61, 61)):
        super(PermInvUNet, self).__init__()
        self.depth = depth
        # Channel sizes at each level
        self.chs = [base_ch * (2 ** i) for i in range(depth + 1)]
        self.width = 12
        # Compute skip connection sizes and required output_padding for ConvTranspose2d
        H0, W0 = input_size
        skip_H = [H0]
        skip_W = [W0]
        for _ in range(depth):
            skip_H.append(skip_H[-1] // 2)
            skip_W.append(skip_W[-1] // 2)
        pads = []
        cur_H, cur_W = skip_H[-1], skip_W[-1]
        for size_H, size_W in zip(reversed(skip_H[:-1]), reversed(skip_W[:-1])):
            pad_H = size_H - ((cur_H - 1) * 2 + 2)
            pad_W = size_W - ((cur_W - 1) * 2 + 2)
            # pad_H and pad_W should be 0 or 1
            pads.append((pad_H, pad_W))
            cur_H, cur_W = size_H, size_W

        # Downsampling path
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(nn.Sequential(
            nn.Conv2d(in_ch, self.chs[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.chs[0]),
            nn.ReLU(inplace=True)
        ))
        for i in range(depth):
            self.pools.append(nn.MaxPool2d(2))
            self.down_convs.append(nn.Sequential(
                nn.Conv2d(self.chs[i], self.chs[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(self.chs[i + 1]),
                nn.ReLU(inplace=True)
            ))

        # Normalization for skip connections
        self.skip_norms = nn.ModuleList([
            nn.BatchNorm2d(ch) for ch in self.chs
        ])

        # Upsampling path with fixed output_padding to match skip sizes
        self.up_transposes = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for (pad_H, pad_W), i in zip(pads, reversed(range(depth))):
            self.up_transposes.append(nn.ConvTranspose2d(
                self.chs[i + 1], self.chs[i],
                kernel_size=2, stride=2,
                output_padding=(pad_H, pad_W)
            ))
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(self.chs[i] * 2, self.chs[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(self.chs[i]),
                nn.ReLU(inplace=True)
            ))

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(self.chs[0], self.width, kernel_size=1)

        self.fno_drift = FNO2d(modes = 32,
                    width = self.width,
                    n_layers = 3,
                    input_dim  = self.width,
                    output_dim = 1)
        self.fno_diffusion = FNO2d(modes = 32,
                    width = self.width,
                    n_layers = 3,
                    input_dim  = self.width,
                    output_dim = 1)  

    def forward(self, x):
        # x: (B, L, H, W)
        B, L, H, W = x.shape
        h = x.view(B * L, 1, H, W)

        # Downsampling
        feats = []
        for i in range(self.depth + 1):
            h = self.down_convs[i](h)
            feats.append(h)
            if i < self.depth:
                h = self.pools[i](h)

        # Bottom aggregation over time dimension
        _, c_bot, H_bot, W_bot = h.shape
        h = h.view(B, L, c_bot, H_bot, W_bot).mean(dim=1)

        # Upsampling with exact shape matching
        for i in range(self.depth):
            h = self.up_transposes[i](h)
            skip = feats[self.depth - 1 - i]
            B_L, c_s, H_s, W_s = skip.shape
            skip = skip.view(B, L, c_s, H_s, W_s).mean(dim=1)
            skip = self.skip_norms[self.depth - 1 - i](skip)
            # Concatenate skip features (shapes now match by design)
            h = torch.cat([h, skip], dim=1)
            h = self.up_convs[i](h)

        # Final output
        fused = self.final_conv(h).permute(0,2,3,1)
        potential = self.fno_drift(fused)
        drag = self.fno_diffusion(fused)
        x = torch.cat((potential,drag),dim = -1)
        return x



class TemporalSelfAttention(nn.Module):
    """
    对时间维度 L 上的特征做自注意力，保持对置换的等变性。
    将 C, H, W 展平为特征维度，直接在序列长度 L 上计算全局注意力，并加入残差与 LayerNorm 以提升稳定性。
    输入 x 形状: (B, L, C, H, W)
    输出形状相同: (B, L, C, H, W)
    """
    def __init__(self, C, H, W):
        super(TemporalSelfAttention, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.D = C * H * W
        # 在特征维度上做 LayerNorm
        self.norm = nn.LayerNorm(self.D)

    def forward(self, x):
        # x: (B, L, C, H, W)
        B, L, C, H, W = x.size()
        # 将通道和空间维度展平到特征维度 D
        x_flat = x.reshape(B, L, self.D)           # [B, L, D]
        # 计算注意力分数: QK^T / sqrt(D)
        scores = torch.matmul(x_flat, x_flat.transpose(1, 2)) / math.sqrt(self.D)  # [B, L, L]
        attn = torch.softmax(scores, dim=-1)      # [B, L, L]
        # 加权求和
        out_flat = torch.matmul(attn, x_flat)     # [B, L, D]
        # 残差连接
        out_flat = out_flat + x_flat
        # LayerNorm 提升数值稳定性
        out_flat = self.norm(out_flat)
        # 恢复形状
        out = out_flat.reshape(B, L, C, H, W)      # [B, L, C, H, W]
        return out


class PermInvUNet_attn(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base_ch=1, depth=4, input_size=(61, 61)):
        super(PermInvUNet_attn, self).__init__()
        self.depth = depth
        self.width = 12
        # 各层通道数: [base_ch*2**i]
        self.chs = [base_ch * (2 ** i) for i in range(depth + 1)]

        # 计算每层跳跃连接的空间尺寸，以及 ConvTranspose2d 的 output_padding
        H0, W0 = input_size
        skip_H, skip_W = [H0], [W0]
        for _ in range(depth):
            skip_H.append(skip_H[-1] // 2)
            skip_W.append(skip_W[-1] // 2)
        pads = []
        cur_H, cur_W = skip_H[-1], skip_W[-1]
        for size_H, size_W in zip(reversed(skip_H[:-1]), reversed(skip_W[:-1])):
            pad_H = size_H - ((cur_H - 1) * 2 + 2)
            pad_W = size_W - ((cur_W - 1) * 2 + 2)
            pads.append((pad_H, pad_W))
            cur_H, cur_W = size_H, size_W

        # 下采样
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(nn.Sequential(
            nn.Conv2d(in_ch, self.chs[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.chs[0]),
            nn.ReLU(inplace=True)
        ))
        for i in range(depth):
            self.pools.append(nn.MaxPool2d(2))
            self.down_convs.append(nn.Sequential(
                nn.Conv2d(self.chs[i], self.chs[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(self.chs[i + 1]),
                nn.ReLU(inplace=True)
            ))

        # 跳跃连接的归一化
        self.skip_norms = nn.ModuleList([
            nn.BatchNorm2d(ch) for ch in self.chs
        ])
        # 时间维度自注意力模块，为每层的特征图建立一个, 并注入对应的 C, H, W
        self.temp_atts = nn.ModuleList([
            TemporalSelfAttention(self.chs[i], skip_H[i], skip_W[i]) for i in range(depth + 1)
        ])

        # 上采样
        self.up_transposes = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for (pad_H, pad_W), i in zip(pads, reversed(range(depth))):
            self.up_transposes.append(nn.ConvTranspose2d(
                self.chs[i + 1], self.chs[i], kernel_size=2, stride=2,
                output_padding=(pad_H, pad_W)
            ))
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(self.chs[i] * 2, self.chs[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(self.chs[i]),
                nn.ReLU(inplace=True)
            ))

        # 最终 1x1 卷积，输出融合特征
        self.final_conv = nn.Conv2d(self.chs[0], self.width, kernel_size=1)
        # FNO 分支
        self.fno_drift = FNO2d(modes=32, width=self.width, n_layers=3, input_dim=self.width, output_dim=1)
        self.fno_diffusion = FNO2d(modes=32, width=self.width, n_layers=3, input_dim=self.width, output_dim=1)
        self.fno_Fx = FNO2d(modes=32,
                               width=self.width,
                               n_layers=3,
                               input_dim=self.width,
                               output_dim=1)
        self.fno_Fy = FNO2d(modes=32,
                               width=self.width,
                               n_layers=3,
                               input_dim=self.width,
                               output_dim=1)
               
    def forward(self, x):
        if self.training:
            L = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], L)
            x = x[:, idx]
        else:
            L = x.shape[1]
        # x: (B, L, H, W)
        B, L, H, W = x.shape
        # 将时序展开到 Batch×L 上并作为单通道输入
        h = x.view(B * L, 1, H, W)

        # 下采样，保存各层特征
        feats = []
        for i in range(self.depth + 1):
            h = self.down_convs[i](h)
            feats.append(h)
            if i < self.depth:
                h = self.pools[i](h)

        # 底层特征自注意力 + 聚合
        _, c_bot, H_bot, W_bot = h.shape
        h_seq = h.view(B, L, c_bot, H_bot, W_bot)
        h_att = self.temp_atts[self.depth](h_seq)
        h = h_att.mean(dim=1)  # (B, c_bot, H_bot, W_bot)

        # 上采样，融合跳跃连接
        for i in range(self.depth):
            h = self.up_transposes[i](h)
            skip = feats[self.depth - 1 - i]
            B_L, c_s, H_s, W_s = skip.shape
            skip_seq = skip.view(B, L, c_s, H_s, W_s)
            skip_att = self.temp_atts[self.depth - 1 - i](skip_seq)
            skip_agg = skip_att.mean(dim=1)
            skip_norm = self.skip_norms[self.depth - 1 - i](skip_agg)
            h = torch.cat([h, skip_norm], dim=1)
            h = self.up_convs[i](h)

        # 输出融合，进入 FNO 分支
        fused = self.final_conv(h).permute(0, 2, 3, 1)  # (B, H, W, width)
        fx = self.fno_Fx(fused)
        fy = self.fno_Fy(fused)
        x_out = torch.cat((fx,fy),dim = -1)  
        return x_out


# class TemporalTransformerEncoder(nn.Module):
#     """
#     基于Transformer Encoder的时序特征处理模块。
#     输入 x 形状: (B, L, C, H, W)
#     输出形状相同: (B, L, C, H, W)
#     """
#     def __init__(self, embed_dim, num_heads=1, ff_dim=None, num_layers=2, dropout=0.1):
#         super(TemporalTransformerEncoder, self).__init__()
#         if ff_dim is None:
#             ff_dim = embed_dim * 4
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=ff_dim,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         # x: (B, L, C, H, W)
#         B, L, C, H, W = x.size()
#         # 合并空间位置到batch维度: (B*H*W, L, C)
#         x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, L, C)
#         # Transformer Encoder
#         out = self.transformer(x_flat)
#         # 恢复形状: (B, L, C, H, W)
#         out = out.reshape(B, H, W, L, C).permute(0, 3, 4, 1, 2)
#         return out

# class PermInvUNet_attn(nn.Module):
#     def __init__(self, in_ch=1, out_ch=2, base_ch=1, depth=4, input_size=(61, 61),
#                  attn_heads=1, attn_ff_mult=4, attn_layers=2):
#         super(PermInvUNet_attn, self).__init__()
#         self.depth = depth
#         self.width = 12
#         # 各层通道数
#         self.chs = [base_ch * (2 ** i) for i in range(depth + 1)]

#         # 计算跳跃连接尺寸和ConvTranspose2d output_padding
#         H0, W0 = input_size
#         skip_H, skip_W = [H0], [W0]
#         for _ in range(depth):
#             skip_H.append(skip_H[-1] // 2)
#             skip_W.append(skip_W[-1] // 2)
#         pads = []
#         cur_H, cur_W = skip_H[-1], skip_W[-1]
#         for size_H, size_W in zip(reversed(skip_H[:-1]), reversed(skip_W[:-1])):
#             pad_H = size_H - ((cur_H - 1) * 2 + 2)
#             pad_W = size_W - ((cur_W - 1) * 2 + 2)
#             pads.append((pad_H, pad_W))
#             cur_H, cur_W = size_H, size_W

#         # 下采样层
#         self.down_convs = nn.ModuleList()
#         self.pools = nn.ModuleList()
#         self.down_convs.append(nn.Sequential(
#             nn.Conv2d(in_ch, self.chs[0], kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.chs[0]),
#             nn.ReLU(inplace=True)
#         ))
#         for i in range(depth):
#             self.pools.append(nn.MaxPool2d(2))
#             self.down_convs.append(nn.Sequential(
#                 nn.Conv2d(self.chs[i], self.chs[i+1], kernel_size=3, padding=1),
#                 nn.BatchNorm2d(self.chs[i+1]),
#                 nn.ReLU(inplace=True)
#             ))

#         # 跳跃连接归一化
#         self.skip_norms = nn.ModuleList([nn.BatchNorm2d(ch) for ch in self.chs])
#         # 时序Transformer模块
       
#         self.temp_atts = nn.ModuleList([
#             TemporalTransformerEncoder(
#                 embed_dim=ch,
#                 num_heads=attn_heads,
#                 ff_dim=ch * attn_ff_mult,
#                 num_layers=attn_layers
#             ) for ch in self.chs
#         ])

#         # 上采样层
#         self.up_transposes = nn.ModuleList()
#         self.up_convs = nn.ModuleList()
#         for (pad_H, pad_W), i in zip(pads, reversed(range(depth))):
#             self.up_transposes.append(nn.ConvTranspose2d(
#                 self.chs[i+1], self.chs[i],
#                 kernel_size=2, stride=2,
#                 output_padding=(pad_H, pad_W)
#             ))
#             self.up_convs.append(nn.Sequential(
#                 nn.Conv2d(self.chs[i]*2, self.chs[i], kernel_size=3, padding=1),
#                 nn.BatchNorm2d(self.chs[i]),
#                 nn.ReLU(inplace=True)
#             ))

#         # 最终1x1卷积
#         self.final_conv = nn.Conv2d(self.chs[0], self.width, kernel_size=1)
#         # FNO分支
#         self.fno_drift = FNO2d(modes=32, width=self.width, n_layers=3, input_dim=self.width, output_dim=1)
#         self.fno_diffusion = FNO2d(modes=32, width=self.width, n_layers=3, input_dim=self.width, output_dim=1)

#     def forward(self, x):
#         # x: (B, L, H, W)
#         B, L, H, W = x.shape
#         h = x.view(B * L, 1, H, W)

#         # 下采样
#         feats = []
#         for i in range(self.depth + 1):
#             h = self.down_convs[i](h)
#             feats.append(h)
#             if i < self.depth:
#                 h = self.pools[i](h)

#         # 底层时序Transformer处理 + 聚合
#         _, c_bot, H_bot, W_bot = h.shape
#         h_seq = h.view(B, L, c_bot, H_bot, W_bot)
#         h_att = self.temp_atts[self.depth](h_seq)
#         h = h_att.mean(dim=1)

#         # 上采样融合跳跃
#         for i in range(self.depth):
#             h = self.up_transposes[i](h)
#             skip = feats[self.depth - 1 - i]
#             B_L, c_s, H_s, W_s = skip.shape
#             skip_seq = skip.view(B, L, c_s, H_s, W_s)
#             skip_att = self.temp_atts[self.depth - 1 - i](skip_seq)
#             skip_agg = skip_att.mean(dim=1)
#             skip_norm = self.skip_norms[self.depth - 1 - i](skip_agg)
#             h = torch.cat([h, skip_norm], dim=1)
#             h = self.up_convs[i](h)

#         # 输出并进入FNO
#         fused = self.final_conv(h).permute(0, 2, 3, 1)
#         potential = self.fno_drift(fused)
#         diffusion = self.fno_diffusion(fused)
#         out = torch.cat([potential, diffusion], dim=-1)
#         return out


if __name__ == "__main__":
    # For 1D
    # input_dimensions_trunk = 1
    # n_hidden_layers = 3
    # neurons = 100
    # n_basis = 25
    # fno_layers = 3
    # width = 12
    # modes = 32
    # device = "cpu"
    # model = NIOFP(input_dimensions_trunk,
    #              n_hidden_layers,
    #              neurons,
    #              n_basis,
    #              fno_layers,
    #              width,
    #              modes,
    #              device)

    #     # 测试网络
    # batch, L, N = 4, 100, 200
    # output_dim = 25

    # # 模拟输入
    # x = torch.randn(batch, L, N)
    # grid = torch.randn(N,1)
    # output = model(x,grid)

    # print("输入形状:", x.shape)  # (batch, 1, L, N)
    # print("输出形状:", output.shape)  # (batch, L, output_dim)
    
    
    
    #For 2D
    # input_dimensions_trunk = 2
    # n_hidden_layers = 3
    # neurons = 100
    # n_basis = 25
    # fno_layers = 3
    # width = 12
    # modes = 32
    # output_dim = 1
    # device = "cpu"
    # model = NIOFP2D(input_dimensions_trunk,
    #              n_hidden_layers,
    #              neurons,
    #              n_basis,
    #              fno_layers,
    #              width,
    #              modes,
    #              output_dim,
    #              device)

    #     # 测试网络
    # batch, L, Nx,Ny = 4, 100, 61, 61
    # output_dim = 25

    # # 模拟输入
    # x = torch.randn(batch, L, Nx,Ny)
    # grid = torch.randn(Nx,Ny,2)
    # output = model(x,grid)

    # print("输入形状:", x.shape)  # (batch, 1, L, N)
    # print("输出形状:", output.shape)  # (batch, L, output_dim)
    
    # For ode
   
    # input_dimensions_trunk = 1
    # n_hidden_layers = 3
    # neurons = 100
    # n_basis = 25
    # fno_layers = 3
    # width = 12
    # modes = 32
    # device = "cpu"
    # output_dim  = 1
    # model = NIOFP_ode(input_dimensions_trunk,
    #              n_hidden_layers,
    #              neurons,
    #              n_basis,
    #              fno_layers,
    #              width,
    #              modes,
    #              output_dim,
    #              device)
    
    #     # 测试网络
    # batch, L, N = 4, 200, 11
    # output_dim = 25

    # # 模拟输入
    # x = torch.randn(batch, L, N)
    # grid = torch.randn(100,1)
    # output = model(x,grid)

    # print("输入形状:", x.shape)  # (batch, 1, L, N)
    # print("输出形状:", output.shape)  # (batch, L, output_dim)
    
    #For 3D
    # input_dimensions_trunk = 3
    # n_hidden_layers = 3
    # neurons = 100
    # n_basis = 25
    # fno_layers = 3
    # width = 12
    # modes = 15
    # output_dim = 1
    # device = "cpu"
    # model = NIOFP3D(input_dimensions_trunk,
    #              n_hidden_layers,
    #              neurons,
    #              n_basis,
    #              fno_layers,
    #              width,
    #              modes,
    #              output_dim,
    #              device)

    #     # 测试网络
    # batch, L, Nx, Ny, Nz = 4, 100, 40,40,40
    # output_dim = 25

    # # 模拟输入
    # x = torch.randn(batch, L, Nx,Ny,Nz)
    # grid = torch.randn(Nx,Ny,Nz,3)
    # output = model(x,grid)

    # print("输入形状:", x.shape)  # (batch, 1, L, N)
    # print("输出形状:", output.shape)  # (batch, L, output_dim)



    # 测试模型输入输出形状

    B, L, Nx, Ny = 32, 100, 61, 61
    model = PermInvUNet_attn(in_ch=1, out_ch=2, base_ch=1, depth=4, input_size=(Nx, Ny))
    x = torch.randn(B, L, Nx, Ny)
    y = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", y.shape)  # torch.Size([32, 2, 61, 61])

