import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import os
import operator
from functools import reduce
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from datetime import date
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from Adam import Adam
#################################################
#
# Utilities
#
#################################################
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# I comment the above

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super().__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(nn.Module):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.register_buffer('mean', torch.mean(x, 0))
        self.register_buffer('std', torch.std(x, 0))
        self.register_buffer('std', torch.std(x, 0))
        self.eps = eps
    
    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class GaussianImageNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super().__init__()

        # self.mean = torch.mean(x, [0,1,2])
        self.std = torch.std(x, [0,1,2])
        self.eps = eps

    def encode(self, x):
        x = (x) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) 
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, relative=True):
        super().__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.relative = relative

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                if self.relative:
                    return torch.sum(diff_norms/y_norms)
                else: 
                    return torch.sum(diff_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True, truncation=True, res=256, return_freq=True, return_l2=True):
        super().__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average
        self.res = res
        self.return_freq = return_freq
        self.return_l2 = return_l2
        if a == None:
            a = [1,] * k
        self.a = a
        k_x = torch.cat((torch.arange(start=0, end=res//2, step=1),torch.arange(start=-res//2, end=0, step=1)), 0).reshape(res,1).repeat(1,res)
        k_y = torch.cat((torch.arange(start=0, end=res//2, step=1),torch.arange(start=-res//2, end=0, step=1)), 0).reshape(1,res).repeat(res,1)
        
        if truncation:
            self.k_x = (torch.abs(k_x)*(torch.abs(k_x)<20)).reshape(1,res,res,1) 
            self.k_y = (torch.abs(k_y)*(torch.abs(k_y)<20)).reshape(1,res,res,1) 
        else:
            self.k_x = torch.abs(k_x).reshape(1,res,res,1) 
            self.k_y = torch.abs(k_y).reshape(1,res,res,1) 
            
    def cuda(self, device):
        self.k_x = self.k_x.to(device)
        self.k_y = self.k_y.to(device)

    def cpu(self):
        self.k_x = self.k_x.cpu()
        self.k_y = self.k_y.cpu()

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None, return_l2=True):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], self.res, self.res, -1)
        y = y.view(y.shape[0], self.res, self.res, -1)

        

        x = torch.fft.fftn(x, dim=[1, 2], norm='ortho')
        y = torch.fft.fftn(y, dim=[1, 2], norm='ortho')

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (self.k_x**2 + self.k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
            l2loss = self.rel(x, y)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)
        
        if self.return_freq:
            return loss, l2loss, x[:, :, 0], y[:, :, 0]
        elif self.return_l2:
            return loss, l2loss
        else:
            return loss
    
    

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


def getPath(data, flag):
    
    
    if data=='darcy':
        if flag=='train':
            PATH = 'Please fill in your datapath'
        else:
            PATH = 'Please fill in your datapath'
    elif data=='darcy20':
        TRAIN_PATH = 'Please fill in your datapath'
        TEST_PATH = 'Please fill in your datapath'
    elif data=='darcy20c6':
        if flag=='train':
            PATH = 'Please fill in your datapath'
        elif flag=='test':
            PATH = 'Please fill in your datapath'
        else:
            PATH = 'Please fill in your datapath'
    elif data=='darcy20c6_c3':
        TRAIN_PATH = 'Please fill in your datapath'
        TEST_PATH = 'Please fill in your datapath'
    elif data=='darcy15c10':
        TRAIN_PATH = 'Please fill in your datapath'
        TEST_PATH = 'Please fill in your datapath'

    elif data=='navier':
        TRAIN_PATH = 'Please fill in your datapath'
        TEST_PATH = 'Please fill in your datapath'
    else: raise NameError('invalid data name')
    
    return PATH
   

def getDarcyDataSet(dataOpt, flag, 
return_normalizer=False, normalizer=None):
    PATH = getPath(dataOpt['data'], flag)
    r = dataOpt['sampling_rate']
    nSample = dataOpt['dataSize'][flag]
    GN = dataOpt['GN']

    reader = MatReader(PATH)
    if dataOpt['sample_x']:
        x = reader.read_field('coeff')[:nSample,::r,::r]
    else:
        x = reader.read_field('coeff')[:nSample,...]
    y = reader.read_field('sol')[:nSample,::r,::r]
    
    if return_normalizer:
        x_normalizer = UnitGaussianNormalizer(x)
        y_normalizer = UnitGaussianNormalizer(y)
        if GN:        
            x = x_normalizer.encode(x)
            return x, y, x_normalizer, y_normalizer
        else:
            return x, y, x_normalizer, y_normalizer
    else:
        if GN:
            if normalizer is None:
                raise NameError('No normalizer')
            else:
                return normalizer.encode(x), y
        else:
            return x, y
        
def getNavierDataSet(dataPath, r, ntrain, ntest, T_in, T, device, T_out=None, return_normalizer=False, GN=False, normalizer=None, full_train=False):

    if not T_out:
        T_out = T_in
    reader = MatReader(dataPath)
    temp = reader.read_field('u').to(device)
    if full_train:
        train_a = temp[:ntrain,::r,::r,:T_in+T]
    else:
        train_a = temp[:ntrain,::r,::r,:T_in]
    train_u = temp[:ntrain,::r,::r,T_out:T+T_out]


    test_a = temp[-ntest:,::r,::r,:T_in]
    test_u = temp[-ntest:,::r,::r,T_out:T+T_out]
    
    if return_normalizer:
        x_normalizer = UnitGaussianNormalizer(train_a)
        y_normalizer = UnitGaussianNormalizer(train_u)
        if GN:        
            train_a = x_normalizer.encode(train_a)
            test_a = x_normalizer.encode(test_a)
            # train_u = y_normalizer.encode(train_u)
            # test_u = y_normalizer.encode(test_u)
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
        else:
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
    else:
        if GN:
            if normalizer is None:
                raise NameError('No normalizer')
            else:
                return normalizer.decode(train_a), train_u, normalizer.decode(test_a), test_u
        else:
            return train_a, train_u, test_a, test_u

def getNavierDataSet2(opt, device, return_normalizer=False, GN=False, normalizer=None):
    dataPath, r, ntrain, ntest, T_in, T_out, T = opt['path'], opt['sampling'], opt['ntrain'], opt['ntest'], opt['T_in'], opt['T_out'], opt['T']

    reader = MatReader(dataPath)
    temp = reader.read_field('u').to(device)
    if opt['full_train']:
        train_a = temp[:ntrain,::r,::r,:T_in+T]
    else:
        train_a = temp[:ntrain,::r,::r,:T_in]
    train_u = temp[:ntrain,::r,::r,T_out:T+T_out]


    test_a = temp[-ntest:,::r,::r,:T_in]
    test_u = temp[-ntest:,::r,::r,T_out:T+T_out]

    print(train_u.shape)
    print(test_u.shape)
    assert (opt['r'] == train_u.shape[-2])
 

    
    if return_normalizer:
        x_normalizer = GaussianImageNormalizer(train_a)
        y_normalizer = GaussianImageNormalizer(train_u)
        if GN:        
            train_a = x_normalizer.encode(train_a)
            test_a = x_normalizer.encode(test_a)
            train_u = y_normalizer.encode(train_u)
            test_u = y_normalizer.encode(test_u)
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
        else:
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
    else:
        if GN:
            if normalizer is None:
                raise NameError('No normalizer')
            else:
                return normalizer.decode(train_a), train_u, normalizer.decode(test_a), test_u
        else:
            return train_a, train_u, test_a, test_u

def getOptimizerScheduler(parameters, epochs, optimizer_type='adam', lr=0.001,
 weight_decay=1e-4, final_div_factor=1e1, div_factor=1e1):
    if optimizer_type == 'sgd':
        optimizer =  torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        optimizer =  torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adagrad':
        optimizer =  torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer =  Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamax':
        optimizer =  torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer =  torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                               div_factor=div_factor, 
                               final_div_factor=final_div_factor,
                               pct_start=0.2,
                               steps_per_epoch=1, 
                               epochs=epochs)
    return optimizer, scheduler
        

def getNavierDataLoader(dataPath, r, ntrain, ntest, T_in, T, batch_size, device, model_name='vFMM', return_normalizer=False, GN=False, normalizer=None):
    train_a, train_u, test_a, test_u = getNavierDataSet(dataPath, r, ntrain, ntest, T_in, T, device, return_normalizer, GN, normalizer)
    if model_name=='vFMM':       
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a.permute(0, 3, 1, 2), train_u.permute(0, 3, 1, 2)), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a.permute(0, 3, 1, 2), test_u.permute(0, 3, 1, 2)), batch_size=batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def getSavePath(data, model_name):
    
    MODEL_PATH = os.path.join(os.path.abspath(''), 'model/' + model_name + data + str(date.today()) + '.log')
    return MODEL_PATH

def save_pickle(var, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(var, f)
        

def visual2d(x, y):
   
    fig, (ax1,ax2) = plt.subplots(1,2,subplot_kw={"projection": "3d"})

    # Make data.
    Z = x.cpu().numpy()  
    X = np.linspace(0, 1, Z.shape[-1])
    Y = np.linspace(0, 1, Z.shape[-1])
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax1.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax1.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)

#     plt.show()

    Z = y.cpu().numpy()
    surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax2.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax2.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
def showcontour(z, **kwargs):
    '''
    show 2D solution z of its contour
    '''
    fig = go.Figure(data=go.Contour(z=z,
                       colorscale='RdYlBu',
                       # zmax=14,
                       # zmin=1,
                       # range_color=[2, 12],
                       line_smoothing=0.85,
                       line_width=0.1,
                       contours=dict(
                           coloring='heatmap',
                              # showlabels=True,
                                       ),
                       # colorbar=dict(
                       #              title="Surface Heat",
                       #              # titleside="top",
                       #              # tickmode="array",
                       #              tickvals=[2, 3, 12],
                       #              # ticktext=["Cool", "Mild", "Hot"],
                       #              # ticks="outside"
                       #          ),
                       ),
                    
                    layout={'xaxis': {'title': 'x-label',
                                      'visible': False,
                                      'showticklabels': False},
                            'yaxis': {'title': 'y-label',
                                      'visible': False,
                                      'showticklabels': False}},
                    
                   )
    fig.update_traces(showscale=False)
    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=0, r=0, t=0, b=0),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          **kwargs)
    fig.show()
    return fig