import numpy as np
import torch
import torch.nn as nn

from Baselines import Encoder,EncoderHelm2,Encoder2D, Encoder_ode, Encoder3D,Encoder3D_down
from DeepONetModules import FeedForwardNN, DeepOnetNoBiasOrg, FFN
from FNOModules import FNO1d,FNO2d,FNO3d


################################################################




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
        self.fno_drift = FNO1d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = 1,
                    device = self.device)
        self.fno_diffusion = FNO1d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = 1,
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
        potential = self.fno_drift(x)
        drag = self.fno_diffusion(x)
        x = torch.cat((potential,drag),dim = -1)
     
        return x


class NIOFP_FNO(nn.Module):
    def __init__(self,
                 fno_layers,
                 width,
                 modes,
                 output_dim,
                 device):
        super(NIOFP_FNO, self).__init__()
       
        # self.fno_inputs = fno_input_dimension
        self.device = device
        self.fno_layers = fno_layers
        self.FNO_input = FNO1d(modes = 12,
                    width = 4,
                    n_layers = 2,
                    input_dim  = 2,
                    output_dim = 1,device = self.device)
        
        self.fc0 = nn.Linear(2, width)
        self.device = device
        self.fno_drift = FNO1d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = 1,
                    device = self.device)
        self.fno_diffusion = FNO1d(modes = modes,
                    width = width,
                    n_layers = self.fno_layers,
                    input_dim  = width,
                    output_dim = 1,
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
        batchsize = x.shape[0]
        x_input = x.reshape(x.shape[0]*x.shape[1],x.shape[2],1)
        nx = (grid.shape[0])

        grid_r = grid.unsqueeze(0).repeat(x.shape[0]*x.shape[1],1,1)
        
        input = torch.cat((x_input,grid_r),dim = 2)
        x = self.FNO_input(input)

        x = x.view(batchsize, L, nx) # batch, L ,Nx
        grid = grid.unsqueeze(0).repeat(x.shape[0],1,1).permute(0,2,1)

        x = torch.cat((grid, x), 1)
        weight_trans_mat = self.fc0.weight.data
        bias_trans_mat = self.fc0.bias.data
        
        weight_trans_mat = torch.cat([weight_trans_mat[:, :1], weight_trans_mat[:, 1].view(-1, 1).repeat(1, L) / L], dim=1)   # width, L+2
        x = x.permute(0, 2, 1)

        x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat
        
        potential = self.fno_drift(x)
        drag = self.fno_diffusion(x)
        x = torch.cat((potential,drag),dim = -1)
     
        return x





import torch
import torch.nn as nn
import math

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1D深度可分离卷积
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)  # (B, C, L)
        x = x.permute(0, 2, 1)  # (B, L, C) 用于LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # 恢复形状 (B, C, L)
        return x + shortcut

class TemporalSelfAttention1D(nn.Module):
    def __init__(self, C, L):
        super().__init__()
        self.C = C
        self.L = L
        self.D = C * L  # 1D特征维度
        self.norm = nn.LayerNorm(self.D)

    def forward(self, x):
        B, T, C, L = x.size()  # (Batch, Time, Channel, Length)
   
        x_flat = x.reshape(B, T, -1)  # 展平为(B, T, C*L)
        
        # 计算注意力
        scores = torch.matmul(x_flat, x_flat.transpose(1, 2)) / math.sqrt(self.D)
        attn = torch.softmax(scores, dim=-1)
        
        # 应用注意力并添加残差连接
        out_flat = torch.matmul(attn, x_flat)
        out_flat += x_flat
        out_flat = self.norm(out_flat)
        
        # 恢复形状
        out = out_flat.view(B, T, C, L)
        return out

class PermInvUNet_attn1D(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base_ch=1, depth=4, input_size=61,device = None ):
        super().__init__()
        self.device = device 
        self.depth = depth
        self.width = 30
        self.chs = [base_ch * (2 ** i) for i in range(depth + 1)]

        # 下采样尺寸计算
        L0 = input_size
        skip_L = [L0]
        for _ in range(depth):
            skip_L.append(skip_L[-1] // 2)
        
        # 计算上采样需要的padding
        pads = []
        cur_L = skip_L[-1]
        for size_L in reversed(skip_L[:-1]):
            pad_L = size_L - ((cur_L - 1) * 2 + 2)
            pads.append(pad_L)
            cur_L = size_L

        # 下采样路径
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(nn.Sequential(
            nn.Conv1d(in_ch, self.chs[0], 3, padding=1),
            ConvNeXtBlock1D(self.chs[0])
        ))
        for i in range(depth):
            self.pools.append(nn.MaxPool1d(2))
            self.down_convs.append(nn.Sequential(
                nn.Conv1d(self.chs[i], self.chs[i+1], 3, padding=1),
                ConvNeXtBlock1D(self.chs[i+1])
            ))

        # 注意力机制和归一化
        self.skip_norms = nn.ModuleList([nn.BatchNorm1d(ch) for ch in self.chs])
        self.temp_atts = nn.ModuleList([
            TemporalSelfAttention1D(self.chs[i], skip_L[i]) for i in range(depth+1)
        ])

        # 上采样路径
        self.up_transposes = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for pad_L, i in zip(pads, reversed(range(depth))):
            self.up_transposes.append(nn.ConvTranspose1d(
                self.chs[i+1], self.chs[i], 2, stride=2,
                output_padding=pad_L
            ))
            self.up_convs.append(nn.Sequential(
                nn.Conv1d(self.chs[i]*2, self.chs[i], 3, padding=1),
                ConvNeXtBlock1D(self.chs[i])
            ))

        # 最终输出层
        self.final_conv = nn.Conv1d(self.chs[0], self.width, 1)

        self.fno_drift = FNO1d(modes =15,
                    width = self.width,
                    n_layers = 3,
                    input_dim  = self.width,
                    output_dim = 1,
                    device = self.device)
        self.fno_diffusion = FNO1d(modes =15,
                    width = self.width,
                    n_layers = 3,
                    input_dim  = self.width,
                    output_dim = 1,
                    device = self.device)

    def forward(self, x):
        B, T, L = x.shape  # 输入形状 (Batch, Time, Length)
        h = x.view(B*T, 1, L)  # 展开时间维度

        # 下采样过程
        feats = []
        for i in range(self.depth + 1):
            h = self.down_convs[i](h)
            feats.append(h)
            if i < self.depth:
                h = self.pools[i](h)

        # 底层特征处理
        _, c_bot, L_bot = h.shape
        h_seq = h.view(B, T, c_bot, L_bot)
        h_att = self.temp_atts[self.depth](h_seq)
        h = h_att.mean(dim=1)  # 时间维度平均

        # 上采样过程
        for i in range(self.depth):
            h = self.up_transposes[i](h)
            skip = feats[self.depth-1-i]
            B_T, c_skip, L_skip = skip.shape
            
            # 处理跳跃连接
            skip_seq = skip.view(B, T, c_skip, L_skip)
            skip_att = self.temp_atts[self.depth-1-i](skip_seq)
            skip_agg = skip_att.mean(dim=1)
            skip_norm = self.skip_norms[self.depth-1-i](skip_agg)
            
            # 特征拼接
            h = torch.cat([h, skip_norm], dim=1)
            h = self.up_convs[i](h)

        # 最终输出
        fused = self.final_conv(h).permute(0,2,1) 
       
        potential = self.fno_drift(fused)
        drag = self.fno_diffusion(fused)
        x = torch.cat((potential,drag),dim = -1)
        return x


class PermInvUNet_attn1D_bag(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base_ch=1, depth=4, input_size=61,device = None ):
        super().__init__()
        self.device = device 
        self.depth = depth
        self.width = 30
        self.chs = [base_ch * (2 ** i) for i in range(depth + 1)]

        # 下采样尺寸计算
        L0 = input_size
        skip_L = [L0]
        for _ in range(depth):
            skip_L.append(skip_L[-1] // 2)
        
        # 计算上采样需要的padding
        pads = []
        cur_L = skip_L[-1]
        for size_L in reversed(skip_L[:-1]):
            pad_L = size_L - ((cur_L - 1) * 2 + 2)
            pads.append(pad_L)
            cur_L = size_L

        # 下采样路径
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(nn.Sequential(
            nn.Conv1d(in_ch, self.chs[0], 3, padding=1),
            ConvNeXtBlock1D(self.chs[0])
        ))
        for i in range(depth):
            self.pools.append(nn.MaxPool1d(2))
            self.down_convs.append(nn.Sequential(
                nn.Conv1d(self.chs[i], self.chs[i+1], 3, padding=1),
                ConvNeXtBlock1D(self.chs[i+1])
            ))

        # 注意力机制和归一化
        self.skip_norms = nn.ModuleList([nn.BatchNorm1d(ch) for ch in self.chs])
        self.temp_atts = nn.ModuleList([
            TemporalSelfAttention1D(self.chs[i], skip_L[i]) for i in range(depth+1)
        ])

        # 上采样路径
        self.up_transposes = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for pad_L, i in zip(pads, reversed(range(depth))):
            self.up_transposes.append(nn.ConvTranspose1d(
                self.chs[i+1], self.chs[i], 2, stride=2,
                output_padding=pad_L
            ))
            self.up_convs.append(nn.Sequential(
                nn.Conv1d(self.chs[i]*2, self.chs[i], 3, padding=1),
                ConvNeXtBlock1D(self.chs[i])
            ))

        # 最终输出层
        self.final_conv = nn.Conv1d(self.chs[0], self.width, 1)

        self.fno_drift = FNO1d(modes =15,
                    width = self.width,
                    n_layers = 3,
                    input_dim  = self.width,
                    output_dim = 1,
                    device = self.device)
        self.fno_diffusion = FNO1d(modes =15,
                    width = self.width,
                    n_layers = 3,
                    input_dim  = self.width,
                    output_dim = 1,
                    device = self.device)

    def forward(self, x):
        if self.training:
            T = np.random.randint(50, x.shape[1])
            idx = np.random.choice(x.shape[1], T)
            x = x[:, idx]
        else:
            T = x.shape[1]
        B, T, L = x.shape  # 输入形状 (Batch, Time, Length)
        h = x.view(B*T, 1, L)  # 展开时间维度

        # 下采样过程
        feats = []
        for i in range(self.depth + 1):
            h = self.down_convs[i](h)
            feats.append(h)
            if i < self.depth:
                h = self.pools[i](h)

        # 底层特征处理
        _, c_bot, L_bot = h.shape
        h_seq = h.view(B, T, c_bot, L_bot)
        h_att = self.temp_atts[self.depth](h_seq)
        h = h_att.mean(dim=1)  # 时间维度平均

        # 上采样过程
        for i in range(self.depth):
            h = self.up_transposes[i](h)
            skip = feats[self.depth-1-i]
            B_T, c_skip, L_skip = skip.shape
            
            # 处理跳跃连接
            skip_seq = skip.view(B, T, c_skip, L_skip)
            skip_att = self.temp_atts[self.depth-1-i](skip_seq)
            skip_agg = skip_att.mean(dim=1)
            skip_norm = self.skip_norms[self.depth-1-i](skip_agg)
            
            # 特征拼接
            h = torch.cat([h, skip_norm], dim=1)
            h = self.up_convs[i](h)

        # 最终输出
        fused = self.final_conv(h).permute(0,2,1) 
       
        potential = self.fno_drift(fused)
        drag = self.fno_diffusion(fused)
        x = torch.cat((potential,drag),dim = -1)
        return x
    



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
    input_dimensions_trunk = 3
    n_hidden_layers = 3
    neurons = 100
    n_basis = 25
    fno_layers = 3
    width = 12
    modes = 15
    output_dim = 1
    device = "cpu"
    model = NIOFP3D(input_dimensions_trunk,
                 n_hidden_layers,
                 neurons,
                 n_basis,
                 fno_layers,
                 width,
                 modes,
                 output_dim,
                 device)

        # 测试网络
    batch, L, Nx, Ny, Nz = 4, 100, 40,40,40
    output_dim = 25

    # 模拟输入
    x = torch.randn(batch, L, Nx,Ny,Nz)
    grid = torch.randn(Nx,Ny,Nz,3)
    output = model(x,grid)

    print("输入形状:", x.shape)  # (batch, 1, L, N)
    print("输出形状:", output.shape)  # (batch, L, output_dim)


