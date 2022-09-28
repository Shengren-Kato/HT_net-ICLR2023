import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.loss import _WeightedLoss
import os
import gc
import scipy.io
import h5py
import pickle
from scipy import interpolate
import torch.nn.functional as F


def get_interp2d(x, n_f, n_c):
    '''
    interpolate (N, n_f, n_f) to (N, n_c, n_c)
    '''
    x_f, y_f = np.linspace(0, 1, n_f), np.linspace(0, 1, n_f)
    x_c, y_c = np.linspace(0, 1, n_c), np.linspace(0, 1, n_c)
    x_interp = []
    for i in range(len(x)):
        xi_interp = interpolate.interp2d(x_f, y_f, x[i])
        x_interp.append(xi_interp(x_c, y_c))
    return np.stack(x_interp, axis=0)


class Datareader(Dataset):
    #Load data
    def __init__(self,
                 data_path=None,
                 normalization=True,
                 res_grid_coarse: int = 256,
                 res_grid_fine: int = 512,
                 region = [-1,1,-1,1],
                 train_data=True,
                 res = 1023,
                 data_len=100,
                 normalizer_x=None,
                 noise=0
                ):

        self.data_path = data_path
        self.train_data = train_data
        self.data_len = data_len
        self.res = res
        self.res_grid_fine = res_grid_fine
        self.res_grid_coarse= res_grid_coarse
        self.h = 1/self.res
        self.region = region
        self.return_boundary = True # If false, just output inner points without boundary points
        self.noise = noise
        self.normalization = normalization
        self.normalizer_x = normalizer_x    # It aims to normalize test data and obtained from train data
        if self.data_path is not None:
            self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        try:
            data = scipy.io.loadmat(self.data_path)
            a = data['coeff']  # (N, n, n)
            u = data['sol']  # (N, n, n)
        except:
            data = h5py.File(self.data_path)
            a = np.transpose(data['coeff'])  # (N, n, n)
            u = np.transpose(data['sol'])  # (N, n, n)

        del data
        gc.collect()

        if self.train_data:
            a, u = a[:self.data_len, :, :], u[:self.data_len, :, :]
        else:
            a, u = a[-self.data_len:, :, :], u[-self.data_len:, :, :]
            # a, u = a[:self.data_len, :, :], u[:self.data_len, :, :]
        self.n_samples = len(a)

        s = int((self.res-1)/(self.res_grid_fine-1))

        # get the gradient for H1 loss
        targets = u
        
        targets_gradx, targets_grady = self.central_diff(
            targets, self.h)  # (N, n_f, n_f)
        targets_gradx = targets_gradx[:, ::s, ::s]
        targets_grady = targets_grady[:, ::s, ::s]
        targets_grad = np.stack(
            [targets_gradx, targets_grady], axis=-1)  # (N, n, n, 2)

        #Be careful! If the data_size is not suitable(Loss boundary information during samping), Use pooling!!!!
        targets = get_interp2d(targets, self.res, self.res_grid_fine).reshape(-1, self.res_grid_fine, self.res_grid_fine, 1)

        # nodes = a[:, ::s, ::s].reshape(-1, self.res_grid_fine, self.res_grid_fine, 1)
        if self.res_grid_fine == self.res_grid_coarse:
            nodes = a.reshape(-1, self.res, self.res, 1)
        else:
            nodes = get_interp2d(a, self.res, self.res_grid_fine).reshape(-1, self.res_grid_fine, self.res_grid_fine, 1)

        self.coeff = nodes
        self.pos = self.get_grid(self.res_grid_coarse)
        self.pos_fine = self.get_grid(self.res_grid_fine)


        if self.train_data and self.normalization:
            self.normalizer_x = UnitGaussianNormalizer() #normalize coeffcient
            self.normalizer_y = UnitGaussianNormalizer() #normalize solution
            nodes = self.normalizer_x.fit_transform(nodes)

            if self.return_boundary:
                _ = self.normalizer_y.fit_transform(x=targets)
            else:
                _ = self.normalizer_y.fit_transform(
                    x=targets[:, 1:-1, 1:-1, :])
        elif self.normalization:
            nodes = self.normalizer_x.transform(nodes)

        if self.noise > 0:
            nodes += self.noise*np.random.randn(*nodes.shape)

        self.node = nodes
        self.target = targets
        self.target_grad = targets_grad

    @staticmethod
    def central_diff(x, h, padding=True):
        # x: (batch, n, n)
        # b = x.shape[0]
        if padding:
            x = np.pad(x, ((0, 0), (1, 1), (1, 1)),
                       'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (x[:, d:, s:-s] - x[:, :-d, s:-s]) / d  # (N, S_x, S_y)
        grad_y = (x[:, s:-s, d:] - x[:, s:-s, :-d]) / d  # (N, S_x, S_y)

        return grad_x / h, grad_y / h

    @staticmethod
    def get_grid(n_grid, subsample=1, region=[-1,1,-1,1], return_boundary=True):
        x = np.linspace(region[0], region[1], n_grid)
        y = np.linspace(region[2], region[3], n_grid)
        x, y = np.meshgrid(x, y)
        s = subsample
        if return_boundary:
            x = x[::s, ::s]
            y = y[::s, ::s]
        else:
            x = x[::s, ::s][1:-1, 1:-1]
            y = y[::s, ::s][1:-1, 1:-1]
        grid = np.stack([x, y], axis=-1)
        return grid

    def __getitem__(self, index):

        pos = torch.from_numpy(self.pos)
        pos_fine = torch.from_numpy(self.pos_fine)

        node = torch.from_numpy(self.node[index])
        coeff = torch.from_numpy(self.coeff[index])
        target = torch.from_numpy(self.target[index])
        target_grad = torch.from_numpy(self.target_grad[index])

        return dict(node=node.float(),
                    coeff=coeff.float(),
                    pos=pos.float(),
                    pos_fine=pos_fine.float(),
                    target=target.float(),
                    target_grad=target_grad.float())
        '''
        node: normalized coeffcients    [N,n,n,1]
        coeff: initial coeffcients  [N,n,n,1]
        pos: position of coarse grid(just for attention)    [N,n,n,2] 
        pos_fine: position of fine grid    [N,n,n,2]
        target: "exact" solution    [N,n,n,1]
        target_grad: gradient of "exact" solution   [N,n,n,2]
        '''


class UnitGaussianNormalizer:
    def __init__(self, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()
        '''
        modified from utils3.py in 
        https://github.com/zongyi-li/fourier_neural_operator
        Changes:
            - .to() has a return to polymorph the torch behavior
            - naming convention changed to sklearn scalers 
        '''
        self.eps = eps

    def fit_transform(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0)
        return (x - self.mean) / (self.std + self.eps)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return (x * (self.std + self.eps)) + self.mean

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.float().to(device)
            self.std = self.std.float().to(device)
        else:
            self.mean = torch.from_numpy(self.mean).float().to(device)
            self.std = torch.from_numpy(self.std).float().to(device)
        return self

    def cuda(self, device=None):
        assert torch.is_tensor(self.mean)
        self.mean = self.mean.float().cuda(device)
        self.std = self.std.float().cuda(device)
        return self

    def cpu(self):
        assert torch.is_tensor(self.mean)
        self.mean = self.mean.float().cpu()
        self.std = self.std.float().cpu()
        return self


class Lossfunction(_WeightedLoss):
    def __init__(self,
                 dim=2,
                 dilation=2,  # central diff
                 regularizer=False, #if true, H1 loss; else, L2 loss.
                 h=1/512,  # mesh size
                 gamma=1e-1,  # \|D(N(u)) - Du\|,
                 eps=1e-10,
                 ):
        super(Lossfunction, self).__init__()
        self.regularizer = regularizer
        assert dilation % 2 == 0
        self.dilation = dilation
        self.dim = dim
        self.h = h
        self.gamma = gamma  # H^1
        self.eps = eps


    def central_diff(self, u: torch.Tensor, h=None):
        '''
        u: function defined on a grid (bsz, n, n)
        out: gradient (N, n-2, n-2, 2)
        '''
        d = self.dilation  # central diff dilation
        s = d // 2  # central diff stride

        grad_x = (u[:, d:, s:-s] - u[:, :-d, s:-s])/d
        grad_y = (u[:, s:-s, d:] - u[:, s:-s, :-d])/d
        grad = torch.stack([grad_x, grad_y], dim=-1)
        return grad/h

    def forward(self, preds, targets, targets_prime=None, K=None):
        r'''
        preds: (N, n, n, 1)
        targets: (N, n, n, 1)
        targets_prime: (N, n, n, 1)
        K: (N, n, n, 1)
        '''

        h = self.h
        d = self.dim
        K = torch.tensor(1) if K is None else K


        target_norm = targets.pow(2).mean(dim=(1, 2)) + self.eps

        if targets_prime is not None:
            targets_prime_norm = d * \
                (K*targets_prime.pow(2)).mean(dim=(1, 2, 3)) + self.eps
        else:
            targets_prime_norm = 1

        loss = ((preds - targets).pow(2)).mean(dim=(1, 2))/target_norm

        metric = loss.sqrt().mean().item()
        loss = loss.sqrt().mean()


        if self.regularizer and targets_prime is not None:
            preds_diff = self.central_diff(preds, h=h)
            s = self.dilation // 2
            targets_prime = targets_prime[:, s:-s, s:-s, :].contiguous()

            if K.ndim > 1:
                K = K[:, s:-s, s:-s].contiguous()

            regularizer = self.gamma * h * ((K * (targets_prime - preds_diff))
                                            .pow(2)).mean(dim=(1, 2, 3))/targets_prime_norm

            regularizer = regularizer.sqrt().mean()

        else:
            regularizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)


        return loss, regularizer, metric

class Identity(nn.Module):
    '''
    a placeholder layer similar to tensorflow.no_op():
    https://github.com/pytorch/pytorch/issues/9160#issuecomment-483048684
    not used anymore as
    https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
    edge and grid are dummy inputs
    '''

    def __init__(self, in_features=None, out_features=None,
                 *args, **kwargs):
        super(Identity, self).__init__()

        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)
        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        return self.id(x)



class Shortcut2d(nn.Module):
    '''
    (-1, in, S, S) -> (-1, out, S, S)
    Used in SimpleResBlock
    '''

    def __init__(self, in_features=None,
                 out_features=None,):
        super(Shortcut2d, self).__init__()
        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x, edge=None, grid=None):
        x = x.permute(0, 2, 3, 1)
        x = self.shortcut(x)
        x = x.permute(0, 3, 1, 2)
        return x




class Conv2dResBlock(nn.Module):
    '''
    Conv2d + a residual block
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Modified from ResNet's basic block, one conv less, no batchnorm
    No batchnorm
    '''

    def __init__(self, in_dim, out_dim,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 dropout=0.1,
                 stride=1,
                 bias=False,
                 residual=False,
                 basic_block=False,
                 activation_type='silu'):
        super(Conv2dResBlock, self).__init__()

        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation,
                      stride=stride,
                      bias=bias),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                self.activation,
                nn.Conv2d(out_dim, out_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.Dropout(dropout),
            )
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Shortcut2d(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x):
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)




class Interp2dEncoder(nn.Module):
    r'''
    Using Interpolate instead of avg pool
    interp dim hard coded or using a factor
    old code uses lambda and cannot be pickled
    '''

    def __init__(self, in_dim: int,
                 feature_dim: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 interp_size=None,
                 residual=False,
                 activation_type='silu',
                 dropout=0.1):
        super(Interp2dEncoder, self).__init__()

        out_dim = feature_dim
        conv_dim0 = out_dim // 3
        conv_dim1 = out_dim // 3
        conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
        padding1 = padding//2 if padding//2 >= 1 else 1
        padding2 = padding//4 if padding//4 >= 1 else 1
        self.interp_size = interp_size
        self.is_scale_factor = isinstance(
            interp_size[0], float) and isinstance(interp_size[1], float)
        self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
                                    padding=padding, activation_type=activation_type,
                                    dropout=dropout,
                                    residual=residual)
        self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
                                    padding=padding1,
                                    stride=stride, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
                                    dilation=dilation,
                                    padding=padding2, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
                                    kernel_size=kernel_size,
                                    residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual


    def forward(self, x):
        x = self.conv0(x)
        if self.is_scale_factor:
            x = F.interpolate(x, scale_factor=self.interp_size[0],
                              mode='bilinear',
                              recompute_scale_factor=True,
                              align_corners=True)
        else:
            x = F.interpolate(x, size=self.interp_size[0],
                              mode='bilinear',
                              align_corners=True)
        x = self.activation(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        if self.add_res:
            out += x

        if self.is_scale_factor:
            out = F.interpolate(out, scale_factor=self.interp_size[1],
                                mode='bilinear',
                                recompute_scale_factor=True,
                                align_corners=True,)
        else:
            out = F.interpolate(out, size=self.interp_size[1],
                              mode='bilinear',
                              align_corners=True)
        return out




class Interp2dUpsample(nn.Module):
    '''
    interpolate then Conv2dResBlock
    old code uses lambda and cannot be pickled
    temp hard-coded dimensions
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual=False,
                 conv_block=True,
                 interp_mode='bilinear',
                 interp_size=None,
                 activation_type='silu',
                 dropout=0.1,
                 debug=False):
        super(Interp2dUpsample, self).__init__()

        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if conv_block:
            self.conv = nn.Sequential(Conv2dResBlock(
                in_dim, out_dim,
                kernel_size=kernel_size,
                padding=padding,
                residual=residual,
                dropout=dropout,
                activation_type=activation_type),
                self.dropout,
                self.activation)
        self.conv_block = conv_block
        self.interp_size = interp_size
        self.interp_mode = interp_mode
        self.debug = debug

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.interp_size[0],
                          mode=self.interp_mode,
                          align_corners=True)
        if self.conv_block:
            x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.interp_size[1],
                          mode=self.interp_mode,
                          align_corners=True)
        return x


def get_num_params(model):
    '''
    a single entry in cfloat and cdouble count as two parameters
    see https://github.com/pytorch/pytorch/issues/57518
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = 0
    for p in model_parameters:
        # num_params += np.prod(p.size()+(2,) if p.is_complex() else p.size())
        num_params += p.numel() * (1 + p.is_complex())
    return num_params


def pooling_2d(mat, kernel_size: tuple = (2, 2), method='mean', padding=False):
    '''Non-overlapping pooling on 2D data (or 2D data stacked as 3D array).

    mat: ndarray, input array to pool. (m, n) or (bsz, m, n)
    kernel_size: tuple of 2, kernel size in (ky, kx).
    method: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    pad: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f), padding is nan
           so when computing mean the 0 is counted

    Return <result>: pooled matrix.

    Modified from https://stackoverflow.com/a/49317610/622119
    to handle the case of batch edge matrices
    CC BY-SA 3.0
    '''

    m, n = mat.shape[-2:]
    ky, kx = kernel_size

    def _ceil(x, y): return int(np.ceil(x/float(y)))

    if padding:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = mat.shape[:-2] + (ny*ky, nx*kx)
        sy = (ny*ky - m)//2
        sx = (nx*kx - n)//2
        _sy = ny*ky - m - sy
        _sx = nx*kx - n - sx

        mat_pad = np.full(size, np.nan)
        mat_pad[..., sy:-_sy, sx:-_sx] = mat
    else:
        ny = m//ky
        nx = n//kx
        mat_pad = mat[..., :ny*ky, :nx*kx]

    new_shape = mat.shape[:-2] + (ny, ky, nx, kx)

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(-3, -1))
    elif method == 'mean':
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(-3, -1))
    else:
        raise NotImplementedError("pooling method not implemented.")

    return result


class Colors:
    """Defining Color Codes to color the text displayed on terminal.
    """

    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    end = "\033[0m"


def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"


def save_pickle(var, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(var, f)