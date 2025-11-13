# Credit to Deng et al. https://arxiv.org/pdf/2111.02926.pdf
# Code from https://openfwi-lanl.github.io/index.html

from collections import OrderedDict
from math import ceil

import torch.nn as nn
import torch.nn.functional as F

from debug_tools import *

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}
NORM_LAYERS_3d = {'bn': nn.BatchNorm3d, 'in': nn.InstanceNorm3d}

# Replace the key names in the checkpoint in which legacy network building blocks are used
def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
             .replace('Conv2DwithBN_Tanh', 'layers')
             .replace('Deconv2DwithBN', 'layers')
             .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)

class ConvBlock3D(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slope=0.2, dropout=None):
        super(ConvBlock3D, self).__init__()
        layers = [nn.Conv3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS_3d:
            layers.append(NORM_LAYERS_3d[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slope, inplace=True))
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slope=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slope, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EncoderHelm2(nn.Module):
    def __init__(self, n_out, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, print_bool=False, **kwargs):
        super(EncoderHelm2, self).__init__()
        # self.n_out = n_out
        self.convblock1 = ConvBlock(1, dim1, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(1, 3), padding=(0, 1))
        self.convblock3_1 = ConvBlock(dim2, dim3, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.convblock3_2 = ConvBlock(dim3, dim3, kernel_size=(1, 3), padding=(0, 1))
        self.convblock4_1 = ConvBlock(dim3, dim4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.convblock4_2 = ConvBlock(dim4, dim4, kernel_size=(1, 3), padding=(0, 1))
        # self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock7_1 = ConvBlock(dim4, dim5, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        # self.convblock7_1 = ConvBlock(dim4, dim5, kernel_size=(4, 5), padding = 0)
        self.convblock7_2 = ConvBlock(dim5, dim5, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        #self.convblock7_3 = ConvBlock(dim5, dim5, kernel_size=(2, 4), padding = 0) #breast
        #self.convblock7_3 = ConvBlock(dim5, dim5, kernel_size=(2, 2), padding = 0) #limb transpose
        self.convblock7_3 = ConvBlock(dim5, dim5, kernel_size=(2, 4), padding = 0) # new limb
        self.linear = nn.Linear(512, n_out)
        self.print_bool = print_bool

    def forward(self, x):
        batch_size = x.shape[0]
        size_fun = x.shape[1]
        
        x = x.view(batch_size * size_fun, x.shape[2], x.shape[3], x.shape[4])
        # x = x.permute(0, 3, 1, 2)
        # Encoder Part
        # self.print_bool=True
        # b*L,1,2,64
        if self.print_bool: print(x.shape) 
        x = self.convblock1(x)  # (None, 32, 49, 16) N, 64, 2,32
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)  # (None, 64, 25, 16)  N, 128, 2, 16
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock3_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock3_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        # x = self.convblock3_1(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        # x = self.convblock3_2(x)  # (None, 64, 13, 16)
        # if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock7_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        # x = self.convblock6_2(x)  # (None, 256, 16, 18)
        # if self.print_bool: print(x.shape)
        # x = self.convblock7_1(x)  # (None, 256, 8, 9) 7
        # if self.print_bool: print(x.shape)
        x = self.convblock7_2(x)  # (None, 256, 8, 9)
        if self.print_bool: print("after7_2:",x.shape)
        x = self.convblock7_3(x)  # (None, 256, 8, 9)
        if self.print_bool: print(x.shape)
        # x = self.convblock8(x)  # (None, 512, 1, 1)
        # if self.print_bool: print(x.shape)
        x = nn.Flatten()(x)

        if self.print_bool: print(x.shape)
        x = x.view(batch_size, size_fun, x.shape[1])
        if self.print_bool: print(x.shape)
        # if self.n_out != 512:
        #    x = self.linear(x)
        x = self.linear(x)
        if self.print_bool: print(x.shape)

        if self.print_bool: print(x.shape)
        if self.print_bool: quit()
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams




class Encoder2D(nn.Module):
    def __init__(self, n_out, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, **kwargs):
        super(Encoder2D, self).__init__()
        # self.n_out = n_out
        self.convblock1 = ConvBlock(1, dim1, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))
        self.convblock3_1 = ConvBlock(dim2, dim3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convblock3_2 = ConvBlock(dim3, dim3, kernel_size=(3, 3), padding=(1, 1))
        self.convblock4_1 = ConvBlock(dim3, dim4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convblock4_2 = ConvBlock(dim4, dim4, kernel_size=(3, 3), padding=(1, 1))
      
        self.convblock7_1 = ConvBlock(dim4, dim5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convblock7_2 = ConvBlock(dim5, dim5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convblock7_3 = ConvBlock(dim5, dim5, kernel_size=(2, 1), padding = 0) # new limb
        self.linear = nn.Linear(512, n_out)
        self.print_bool = False
  
    def forward(self, x):
        # x: b,L,1,nx,ny
        batch_size = x.shape[0]
        size_fun = x.shape[1]
        
        x = x.view(batch_size * size_fun, x.shape[2], x.shape[3], x.shape[4])
        if self.print_bool: print(x.shape) 
        x = self.convblock1(x)  # (None, 32, 49, 16) N, 64, 2,32
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)  # (None, 64, 25, 16)  N, 128, 2, 16
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock3_1(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock3_2(x)  # (None, 64, 25, 16)
        if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        if self.print_bool: print(x.shape)
        x = self.convblock7_1(x)  # (None, 256, 16, 18)
        if self.print_bool: print(x.shape)
        # x = self.convblock6_2(x)  # (None, 256, 16, 18)
        # if self.print_bool: print(x.shape)
        # x = self.convblock7_1(x)  # (None, 256, 8, 9) 7
        # if self.print_bool: print(x.shape)
        x = self.convblock7_2(x)  # (None, 256, 8, 9)
        if self.print_bool: print("after7_2:",x.shape)
        x = self.convblock7_3(x)  # (None, 256, 8, 9)
        if self.print_bool: print(x.shape)
        # x = self.convblock8(x)  # (None, 512, 1, 1)
        # if self.print_bool: print(x.shape)
        x = nn.Flatten()(x)

        if self.print_bool: print(x.shape)
        x = x.view(batch_size, size_fun, x.shape[1])
        if self.print_bool: print(x.shape)
        # if self.n_out != 512:
        #    x = self.linear(x)
        x = self.linear(x)
        if self.print_bool: print(x.shape)

        if self.print_bool: print(x.shape)
        if self.print_bool: quit()
        return x




class Encoder(nn.Module):
    def __init__(self, output_dim, dim1=64, dim2=128, dim3=256):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(1, dim1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2 = ConvBlock(dim1, dim2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3 = ConvBlock(dim2, dim3, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.final_conv1 = ConvBlock(dim3, dim3, kernel_size=(1, 5), stride=(1, 1), padding=(0, 1))
        self.final_conv2 = ConvBlock(dim3, dim3, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
        self.final_conv3 = ConvBlock(dim3, dim3, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0))
        self.final_conv4 = ConvBlock(dim3, dim3, kernel_size=(1, 15), stride=(1, 1), padding=(0, 0))
        self.linear = nn.Linear(dim3, output_dim)

    def forward(self, x):
        batch_size, L, N = x.shape

        # Reshape input to (batch*L, 1, 1, N)
        x = x.view(batch_size * L, 1, 1, N)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        x = self.final_conv3(x)
     

        # Flatten and reshape to (batch, L, feature_dim)
        x = x.view(batch_size, L, -1)

        # Apply linear layer to achieve desired output_dim
        x = self.linear(x)

        return x


class Encoder_ode(nn.Module):
    def __init__(self, output_dim, dim1=64, dim2=128, dim3=256):
        super(Encoder_ode, self).__init__()
        self.conv1 = ConvBlock(1, dim1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2 = ConvBlock(dim1, dim2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3 = ConvBlock(dim2, dim3, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.final_conv1 = ConvBlock(dim3, dim3, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1))
        self.final_conv2 = ConvBlock(dim3, dim3, kernel_size=(3, 2), stride=(1, 1), padding=(0, 0))

        self.linear = nn.Linear(dim3, output_dim)

    def forward(self, x):
        batch_size, L, N = x.shape

        # Reshape input to (batch*L, 1, 1, N)
        x = x.view(batch_size * L, 1, 1, N)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_conv1(x)
        x = self.final_conv2(x)

        # Flatten and reshape to (batch, L, feature_dim)
        x = x.view(batch_size, L, -1)

        # Apply linear layer to achieve desired output_dim
        x = self.linear(x)

        return x

class Encoder3D(nn.Module):
    def __init__(self, n_out, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, **kwargs):
        super(Encoder3D, self).__init__()
        # 定义3D卷积块
        self.convblock1 = ConvBlock3D(1, dim1, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.convblock2_1 = ConvBlock3D(dim1, dim2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock2_2 = ConvBlock3D(dim2, dim2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convblock3_1 = ConvBlock3D(dim2, dim3, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock3_2 = ConvBlock3D(dim3, dim3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convblock4_1 = ConvBlock3D(dim3, dim4, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock4_2 = ConvBlock3D(dim4, dim4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convblock7_1 = ConvBlock3D(dim4, dim5, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock7_2 = ConvBlock3D(dim5, dim5, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock7_3 = ConvBlock3D(dim5, dim5, kernel_size=(2, 1, 1), padding=0)  # new limb
        self.linear = nn.Linear(512, n_out)
        self.print_bool = False

    def forward(self, x):
        # x: b, L, C, nx, ny, nz
        batch_size = x.shape[0]
        size_fun = x.shape[1]
        
        x = x.view(batch_size * size_fun, x.shape[2], x.shape[3], x.shape[4], x.shape[5])
        if self.print_bool: print(x.shape)
        x = self.convblock1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)
        if self.print_bool: print(x.shape)
        x = self.convblock3_1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock3_2(x)
        if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)
        if self.print_bool: print(x.shape)
        x = self.convblock7_1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock7_2(x)
        if self.print_bool: print("after7_2:", x.shape)
        x = self.convblock7_3(x)
        if self.print_bool: print(x.shape)

        x = nn.Flatten()(x)
        if self.print_bool: print(x.shape)
        x = x.view(batch_size, size_fun, x.shape[1])
        if self.print_bool: print(x.shape)
        x = self.linear(x)
        if self.print_bool: print(x.shape)

        return x


class Encoder3D_down(nn.Module):
    def __init__(self, n_out, dim1=64, dim2=128, dim3=256, dim4=512, dim5=512, sample_spatial=1.0, **kwargs):
        super(Encoder3D_down, self).__init__()
        # 定义3D卷积块
        self.convblock1 = ConvBlock3D(1, dim1, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.convblock2_1 = ConvBlock3D(dim1, dim2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock2_2 = ConvBlock3D(dim2, dim2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convblock3_1 = ConvBlock3D(dim2, dim3, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock3_2 = ConvBlock3D(dim3, dim3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convblock4_1 = ConvBlock3D(dim3, dim4, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock4_2 = ConvBlock3D(dim4, dim4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convblock7_1 = ConvBlock3D(dim4, dim5, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock7_2 = ConvBlock3D(dim5, dim5, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.convblock7_3 = ConvBlock3D(dim5, dim5, kernel_size=(1, 1, 1), padding=0)  # new limb
        self.linear = nn.Linear(512, n_out)
        self.print_bool = False

    def forward(self, x):
        # x: b, L, C, nx, ny, nz
        batch_size = x.shape[0]
        size_fun = x.shape[1]
        
        x = x.view(batch_size * size_fun, x.shape[2], x.shape[3], x.shape[4], x.shape[5])
        if self.print_bool: print(x.shape)
        x = self.convblock1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock2_1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock2_2(x)
        if self.print_bool: print(x.shape)
        x = self.convblock3_1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock3_2(x)
        if self.print_bool: print(x.shape)
        x = self.convblock4_1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock4_2(x)
        if self.print_bool: print(x.shape)
        x = self.convblock7_1(x)
        if self.print_bool: print(x.shape)
        x = self.convblock7_2(x)
        if self.print_bool: print("after7_2:", x.shape)
        x = self.convblock7_3(x)
        if self.print_bool: print(x.shape)

        x = nn.Flatten()(x)
        if self.print_bool: print(x.shape)
        x = x.view(batch_size, size_fun, x.shape[1])
        if self.print_bool: print(x.shape)
        x = self.linear(x)
        if self.print_bool: print(x.shape)

        return x

if __name__ == "__main__":
    # batch = 10
    # L = 100
    # nx = 61
    # ny = 61
    # n_out = 25
    # x = torch.rand(batch,L,1,nx,ny)
    # model = Encoder2D(n_out = n_out)
    # y = model(x)
    # print(y.shape)
    # batch = 10
    # L = 200
    # nx = 11
   
    # n_out = 25
    # x = torch.rand(batch,L,nx)
    # model = Encoder_ode(output_dim = n_out)
    # y = model(x)
    batch = 5
    L = 100
    nx = 40
    ny = 40 
    nz = 40
    n_out = 25
    x = torch.rand(batch,L,1,nx,ny,nz)
    model = Encoder3D(n_out = n_out) 
    y = model(x)
    print(y.shape)
    
