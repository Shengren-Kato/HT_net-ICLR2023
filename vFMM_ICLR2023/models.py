import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import reduce
from functools import partial
import operator
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils import *


def waveletShrinkage(x, thr, mode='soft'):
    """
    Perform soft or hard thresholding. The shrinkage is only applied to high frequency blocks.

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param thr: thresholds stored in a torch dense tensor with shape [num_hid_features]
    :param mode: 'soft' or 'hard'. Default: 'soft'
    :return: one block of wavelet coefficients after shrinkage. The shape will not be changed
    """
    assert mode in ('soft', 'hard', 'relu'), 'shrinkage type is invalid'

    if mode == 'soft':
        x = torch.mul(torch.sgn(x), (((torch.abs(x) - thr) + torch.abs(torch.abs(x) - thr)) / 2))
    elif mode == 'relu':
        x = torch.mul(x, (F.relu(torch.abs(x) - thr)))
    else:
        x = torch.mul(x, (torch.abs(x) > thr))

    return x


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, mode_threshold=False, init_scale=16):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.fourier_weight1 = nn.Parameter(
            torch.empty(in_channels, out_channels,
                                                modes1, modes2, 2)) 
        self.fourier_weight2 = nn.Parameter(
            torch.empty(in_channels, out_channels,
                                                modes1, modes2, 2)) 
        
        nn.init.xavier_uniform_(self.fourier_weight1, gain=1/(in_channels*out_channels)
                           * np.sqrt((in_channels+out_channels)/init_scale))
        nn.init.xavier_uniform_(self.fourier_weight2, gain=1/(in_channels*out_channels)
                           * np.sqrt((in_channels+out_channels)/init_scale))
        
        self.mode_threshold = mode_threshold
        self.shrink = nn.Softshrink()
    
    # Complex multiplication
    # def compl_mul2d(self, input, weights):
    #     # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    #     return torch.einsum("bixy,ioxy->boxy", input, weights)
    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.fourier_weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.fourier_weight2)
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])
        
        # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batch_size, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
#         if self.mode_threshold:
#             # out_ft = waveletShrinkage(out_ft, thr=1e-1, mode='relu') 
#             out_ft = self.shrink(out_ft)
#         # the 2d Hermmit symmetric refers to two oppsite directions 
#         out_ft[:, :, :self.modes1, :self.modes2] = \
#             self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2] = \
#             self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        
        

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0., activation='gelu', layer=2, LN=True):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'gelu':
            act = nn.GELU
        elif activation == 'tanh':
            act = nn.Tanh
        else: raise NameError('invalid activation')
         
        if LN:
            if layer == 2:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim)
                )
            elif layer == 3:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                    act(),
                    nn.LayerNorm(out_dim)
                )
            else: raise NameError('only accept 2 or 3 layers')
        else:
            if layer == 2:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                )
            elif layer == 3:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                )
            else: raise NameError('only accept 2 or 3 layers')
            
    def forward(self, x):
        return self.net(x)
    def reset_parameters(self):
        for layer in self.children():
            for n, l in layer.named_modules():
                if hasattr(l, 'reset_parameters'):
                    print(f'Reset trainable parameters of layer = {l}')
                    l.reset_parameters()
                    
class FNO2d(nn.Module):
    def __init__(self, modes=12, width=32, num_spectral_layers=4, mlp_hidden_dim=128, 
                output_dim=1, mlp_LN=False, activation='gelu', mode_threshold=False, 
                kernel_type='p', padding=9, resolution=None, init_scale=16, add_pos=True):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes
        self.modes2 = modes
        self.add_pos = add_pos
        if add_pos:
            self.width = width + 2
        else:
            self.width = width        
        self.num_spectral_layers = num_spectral_layers
        self.padding = padding # pad the domain if input is non-periodic
        
        self.Spectral_Conv_List = nn.ModuleList([])
        
            # self.Spectral_Conv_List.append(SpectralConv2d(self.width+2, self.width, self.modes1, self.modes2, mode_threshold, init_scale))
        for _ in range(num_spectral_layers):
            self.Spectral_Conv_List.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2, mode_threshold, init_scale))
           
          
        self.Conv2d_list = nn.ModuleList([])         
            # self.Conv2d_list.append(nn.Conv2d(self.width+2, self.width, kernel_size=3, stride=1, padding=1, dilation=1))
 
  
        if kernel_type == 'p':
            for _ in range(num_spectral_layers):
                self.Conv2d_list.append(nn.Conv2d(self.width, self.width, 1))
        else:         
            for _ in range(num_spectral_layers):
                self.Conv2d_list.append(nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1))
        
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation')
  
        self.mlp = FeedForward(self.width, mlp_hidden_dim, output_dim, LN=mlp_LN)
        self.grid = None
        self.resolution = resolution
        
    def forward(self, x):
        if self.add_pos:
            if self.grid is None:        
                grid = self.get_grid(x.shape, x.device)
                self.grid = grid
                x = torch.cat((x, grid), dim=-1)
            else: x = torch.cat((x, self.grid), dim=-1)

#         x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        if self.padding:
            x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.Spectral_Conv_List[0](x)
        x2 = self.Conv2d_list[0](x)
        x_shortcut = self.act(x1 + x2)
        x = x_shortcut

        for i in range(1, self.num_spectral_layers - 1):
            x1 = self.Spectral_Conv_List[i](x)
            x2 = self.Conv2d_list[i](x)
            x = x1 + x2
            x = self.act(x)

        x1 = self.Spectral_Conv_List[-1](x)
        x2 = self.Conv2d_list[-1](x)
        x = x1 + x2 + x_shortcut

#         x = x[..., :-self.padding, :-self.padding]
        if self.padding:
            x = x[..., :self.resolution, :self.resolution]
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        return x
    
    def get_grid(self, shape, device):
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batch_size, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batch_size, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
class FNO(nn.Module):
    def __init__(self, in_dim,
                 freq_dim,
                 modes=32,
                 dim_feedforward=256,
                 posadd2=True,
                 activation='silu',
                 padding=9,
                 dropout=0.0,):
        super(FNO, self).__init__()

        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.posadd2 = posadd2
        self.padding = padding
        self.dropout = nn.Dropout(dropout)
        self.width = freq_dim

        if self.posadd2:
            self.fc0 = nn.Linear(in_dim+2, self.width)
        else:
            self.fc0 = nn.Linear(in_dim, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, modes, modes)
        self.conv1 = SpectralConv2d(self.width, self.width, modes, modes)
        self.w0 = nn.Linear(self.width, self.width)
        self.w1 = nn.Linear(self.width, self.width)

        self.regressor = nn.Sequential(
            nn.Linear(freq_dim,  dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x, pos):
        if self.posadd2:
            x = torch.cat([x, pos], dim=-1)

        x = self.fc0(x)

        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x = x.permute(0, 2, 3, 1)
        x2 = self.w0(x)
        x2 = x2.permute(0, 3, 1, 2)
        x = x1 + x2
        x = self.activation(x)
        x = self.dropout(x)

        x1 = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x2 = self.w1(x)
        x2 = x2.permute(0, 3, 1, 2)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)

        x = self.regressor(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
#     print(x.shape)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim+2*self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pos=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        x = x.repeat(1,1,3).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]  # make torchscript happy (cannot use tensor as tuple)

        if pos is not None:
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.num_heads, 1, 1])
            q, k, v = [torch.cat([pos, x], dim=-1)
                       for x in (q, k, v)]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if pos is not None:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C+2*self.num_heads)
            x = self.proj2(x)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, pos):
        if len(x.shape)==3:
            H, W = self.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
            outshape = 3
        elif len(x.shape)==4:
            B, H, W, C = x.shape
            outshape = 4

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if pos is not None:
            pos_windows = window_partition(pos, self.window_size)
            pos_windows = pos_windows.view(-1, self.window_size * self.window_size, 2)
        else:
            pos_windows = None

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, pos=pos_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if outshape == 3:
            x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, downtype='purify'):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        if downtype=='purify':
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        elif downtype=='keep':
            self.reduction = nn.Conv2d(in_channels=4 * dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.downtype = downtype
        self.reconstruction = nn.Linear(2 * dim, 4 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, direction='down', uplevel_result1=None, uplevel_result2=None):
        """
        x: B, H*W, C
        """
        if direction=='down':
            
            H, W = self.input_resolution
            if len(x.shape) == 3:
                B, L, C = x.shape
                outshape = 3
            else:
                B, H, W, C = x.shape
                outshape = 4
#             print(x.shape)
#             print(H, W)
            # assert L == H * W, "input feature has wrong size"
            # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

            x = x.view(B, H, W, C)

            x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            if outshape == 3:
                x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

            x = self.norm(x)
            if self.downtype == 'keep':
                x = x.permute(0,3,1,2)
                x = self.reduction(x)
                x = x.permute(0,2,3,1)
            else:
                x = self.reduction(x)
        else:
            H, W = self.input_resolution
#             print(x.shape)
            B = x.shape[0]
            C = x.shape[-1]
#             print(x.shape)
#             print(H, W)
#             assert H_0 == H // 2, "input feature has wrong size"
            x = self.reconstruction(x)
            x = x.view(B, H//2, W//2, 2*C)
            New_C = C//2
            x_target = torch.zeros(B, H, W, self.dim, device=x.device)

            x_target[:, 0::2, 0::2, :] = x[...,:New_C] # B H/2 W/2 C
            x_target[:, 1::2, 0::2, :] = x[...,New_C:2*New_C]  # B H/2 W/2 C
            x_target[:, 0::2, 1::2, :] = x[...,2*New_C:3*New_C]  # B H/2 W/2 C
            x_target[:, 1::2, 1::2, :] = x[...,3*New_C:] # B H/2 W/2 C
            x = x_target + uplevel_result1.view(B, H, W, self.dim) 
           
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
    
class PatchUnMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
#         self.reduction = nn.Linear(dim//4, 1, bias=False)
#         self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        New_C = C//4
        x_target = torch.zeros(B, H*2, W*2, New_C, device=x.device)

        x_target[:, 0::2, 0::2, :] = x[..., :New_C] # B H/2 W/2 C
        x_target[:, 1::2, 0::2, :] = x[..., New_C:2*New_C]  # B H/2 W/2 C
        x_target[:, 0::2, 1::2, :] = x[..., 2*New_C:3*New_C]  # B H/2 W/2 C
        x_target[:, 1::2, 1::2, :] = x[..., 3*New_C:] # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

#         x = self.norm(x)
#         x = self.reduction(x)

        return x_target

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, downtype='purify', use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, downtype=downtype)
        else:
            self.downsample = None

    def forward(self, x, pos=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x)
            else:
                x1 = blk(x, pos)
        if self.downsample is not None:
            x2 = self.downsample(x1)
        else:
            x2 = x1
        if pos is not None:
            pos = pos[:, 1::2, 1::2, :]
        return x1, x2, pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, stride=2, patch_padding=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [(img_size[0]-patch_size[0]+2*patch_padding) // stride + 1, 
                              (img_size[0]-patch_size[0]+2*patch_padding) // stride + 1]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

class RebuildLayer(nn.Module):
    def __init__(self, dim):

        super().__init__()
        self.dim = dim
        self.rebuildnet = nn.Linear(dim, 2*dim)

    def forward(self, x):
        B, H, W, C0 = x.shape
        C = int(C0/2)
        x_temp = self.rebuildnet(x)
        x_target = torch.zeros(B, H*2, W*2, self.dim//2, device=x.device)
        x_target[:, 0::2, 0::2, :] = x_temp[..., :C] # B H/2 W/2 C
        x_target[:, 1::2, 0::2, :] = x_temp[..., C:2*C]  # B H/2 W/2 C
        x_target[:, 0::2, 1::2, :] = x_temp[..., 2*C:3*C]  # B H/2 W/2 C
        x_target[:, 1::2, 1::2, :] = x_temp[..., 3*C:] # B H/2 W/2 C

        return x_target

class Downscaler(nn.Module):
    def __init__(self, in_dim, feature_dim, down_size,
                 downsample_mode='interp',
                 activation_type='silu',
                 dropout=0.05,
                 #====================================
                 patch_size=3,
                 stride=2,
                 patch_padding=1
                 ):
        super(Downscaler, self).__init__()
        self.downsample_mode = downsample_mode
        #choose the way to decrease the number of nodes. Do attention with many nodes may lead GPU out of memory!
        if self.downsample_mode == 'linear':
            self.downsample = nn.Linear(in_dim, feature_dim)
        elif self.downsample_mode == 'interp':
            self.downsample = Interp2dEncoder(in_dim=in_dim,
                                              feature_dim=feature_dim,
                                              interp_size=down_size,
                                              activation_type=activation_type,
                                              dropout=dropout)
        elif self.downsample_mode == 'patchembed':
            self.downsample =PatchEmbed(patch_size=patch_size, in_chans=in_dim,
            embed_dim=feature_dim, stride=stride, patch_padding=patch_padding)
        else:
            raise NotImplementedError("downsample type error")
        self.in_dim = in_dim

    def forward(self, x):
        if self.downsample_mode == 'linear':
            x = self.downsample(x)
        elif self.downsample_mode == 'interp':
            x = x.permute(0, 3, 1, 2)
            x = self.downsample(x)
            x = x.permute(0, 2, 3, 1)
        elif self.downsample_mode == 'patchembed':
            x = x.permute(0,3,1,2)
            x = self.downsample(x)
            B, L, C = x.shape
            H = int(pow(L,0.5))
            x= x.view(B,H,H,-1)
        return x

class UpScaler(nn.Module):
    def __init__(self, in_dim, out_dim, interp_size,
                 dropout=0.0, upsample_mode='interp', activation_type='silu',):
        super(UpScaler, self).__init__()

        '''
        A wrapper for DeConv2d upscaler or interpolation upscaler
        Deconv: Conv1dTranspose
        Interp: interp->conv->interp
        '''
        self.upsample_mode = upsample_mode
        if self.upsample_mode == 'linear':
            self.upsample = nn.Linear(in_dim, out_dim)
        elif self.upsample_mode == 'interp':
            self.upsample = Interp2dUpsample(in_dim=in_dim,
                                             out_dim=out_dim,
                                             interp_size=interp_size,
                                             dropout=dropout,
                                             activation_type=activation_type)
        elif self.upsample_mode == 'identity':
             self.upsample = torch.nn.Identity()

        else:
            raise NotImplementedError("downsample type error")
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n_s, n_s, in_dim)
            Output: (-1, n, n, out_dim)
        '''
        if self.upsample_mode == 'linear' or self.upsample_mode =='patchembed':
            x = self.upsample(x)
        elif self.upsample_mode == 'interp':
            x = x.permute(0, 3, 1, 2)
            x = self.upsample(x)
            x = x.permute(0, 2, 3, 1)
        return x    

class Transformer(nn.Module):
    def __init__(self, img_size=512, in_chans=128, depths=[2, 2], num_heads=[4, 4],
                 window_size=[8, 8], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, downtype='keep',
                 norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.num_features = int(in_chans)
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        if downtype == 'keep':
            layerdim = int(self.num_features)
        for i_layer in range(self.num_layers):
            if downtype == 'purify':
                layerdim = int(self.num_features*2**i_layer)
            layer = BasicLayer(dim=layerdim,
                               input_resolution=(img_size // (2 ** i_layer),
                                                 img_size // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               downtype=downtype,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)



        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if downtype=='keep':
            self.relayers = None
            self.head = nn.Linear(self.num_features*self.num_layers, self.num_features)
        elif downtype=='purify':
            self.relayers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = RebuildLayer(int(self.num_features * 2 ** i_layer))
                self.relayers.append(layer)
            self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, pos):
        list_x_out = []
        i_att = 0
        x_downsample = x
        for layer in self.layers:
            x_att, x_downsample, pos = layer(x_downsample, pos)  # do attention
            list_x_out.append(x_att)
            i_att += 1

        if self.relayers is None:
            for i_att in range(self.num_layers - 1, 0, -1):
                A, B, C = list_x_out[i_att - 1].shape[0:-1]
                D = list_x_out[i_att].shape[-1]
                temp = torch.zeros([A, B, C, D], device=x.device)
                temp[:, 0::2, 0::2, :] = list_x_out[i_att]
                temp[:, 1::2, 0::2, :] = list_x_out[i_att]
                temp[:, 0::2, 1::2, :] = list_x_out[i_att]
                temp[:, 1::2, 1::2, :] = list_x_out[i_att]
                list_x_out[i_att - 1] = torch.cat((list_x_out[i_att - 1], temp), dim=-1)

        else:
            for i_att in range(self.num_layers-1, 0, -1):
                x_temp = self.relayers[i_att](list_x_out[i_att])
                list_x_out[i_att-1] = list_x_out[i_att-1] + x_temp

        x = list_x_out[0]

        return x

    def forward(self, x, pos):
        x = self.forward_features(x, pos)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

class Hnet2d(nn.Module):
    def __init__(self, R_dic):
        super(Hnet2d, self).__init__()
        self.boundary_condition = R_dic['boundary_condition']
        self.posadd1 = R_dic['posadd1']
        self.posadd2 = R_dic['posadd2']
        self.dim_feedforward = R_dic['dim_feedforward']

        self.feature_dim = R_dic['feature_dim']
        self.freq_dim = R_dic['freq_dim']
        self.modes = R_dic['modes']
        self.dropout = 0.0

        self.downscaler_size = R_dic['downscaler_size']
        self.upscaler_size = R_dic['upscaler_size']

        if self.posadd1:
            self.in_dim = 3
        else:
            self.in_dim = 1

        self.downscaler = Downscaler(in_dim=self.in_dim, feature_dim=self.feature_dim, downsample_mode=R_dic['downsample_mode'], down_size=self.downscaler_size, stride=R_dic['subsample_attn'])

        self.attn = Transformer(img_size=R_dic['resolution_coarse'], in_chans=self.feature_dim,depths=R_dic['depths'], num_heads=R_dic['num_heads'],
                window_size=R_dic['window_size'], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, downtype=R_dic['downtype'],
                norm_layer=nn.LayerNorm, use_checkpoint=False)



        self.upscaler = UpScaler(in_dim=self.feature_dim, out_dim=self.feature_dim, upsample_mode=R_dic['upsample_mode'], interp_size=self.upscaler_size)

        self.dpo = nn.Dropout(self.dropout)


        self.regressor = FNO(in_dim=self.feature_dim, freq_dim=self.freq_dim, modes=self.modes,
                                   dim_feedforward=self.dim_feedforward, activation='silu')


        self.normalize_type = R_dic['normalize_type']

        self.__name__ = 'Hnet2d'

    def forward(self, node, pos_fine, pos_coarse, Normfunction):

        # input data
        if self.posadd1:
            x = torch.cat([node, pos_fine], dim=-1)
        else:
            x = node


        x = self.downscaler(x)
        x = self.dpo(x)

        if x.shape[2] == pos_coarse.shape[2]:
            x = self.attn(x, pos_coarse)
        else:
            x = self.attn(x, pos_fine)

        x = self.upscaler(x)
        x = self.dpo(x)

        
        x = self.regressor(x, pos_fine)
        # x = self.net_linear(x)
        if self.normalize_type:
            x = Normfunction.inverse_transform(x)

        if self.boundary_condition == 'dirichlet':
            x = x[:, 1:-1, 1:-1].contiguous()
            x = F.pad(x, (0, 0, 1, 1, 1, 1), "constant", 0)


        return x

class FMMTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, FNO_paras, img_size=224, patch_size=4, in_chans=3, 
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=[7, 7, 7, 7], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, stride=2, patch_padding=1, normalizer=None):
        super().__init__()

        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride,
            norm_layer=norm_layer if self.patch_norm else None, patch_padding=patch_padding)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            
        for i_layer in range(self.num_layers-1):
            layer = PatchMerging(dim=int(embed_dim * 2 ** i_layer),
                                 input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)))
            self.downsamplers.append(layer)
            
        self.patchUnmerge = PatchUnMerging(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)))
        # self.norm = norm_layer(self.num_features//4)
        self.norm = norm_layer(self.embed_dim)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.Decoder = FNO2d(**FNO_paras)
        self.normalizer = normalizer
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        _, y0, _ = self.layers[0](x)
        if self.num_layers==1 :
            B = y0.size(dim=0)
            H, W = self.patches_resolution
            y0 = y0.view(B, H, W, -1)
        else: 
            x1 = self.downsamplers[0](x)
            _, y1, _ = self.layers[1](x1)
            if self.num_layers==3:
                x2 = self.downsamplers[1](x1)
                _, y2, _ = self.layers[2](x2)
                y1 = self.downsamplers[1](y2, direction='up', uplevel_result1=y1)
            y0 = self.downsamplers[0](y1, direction='up', uplevel_result1=y0)
        
        x = self.norm(y0)  # B H W C
        x = torch.squeeze(self.Decoder(x))  # forget why using squeeze 
   
        if self.normalizer:  
            x = self.normalizer.decode(x)
   
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops