import torch
import torch.nn as nn
from models import *
from utils import *
from torch.optim.lr_scheduler import OneCycleLR
from datetime import date
from tqdm.auto import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_path, 'model')
DATA_PATH = os.path.join(current_path, 'data')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#==========================================================================================
# train_path = os.path.join(DATA_PATH, R_dic['train_path'])
# test_path = os.path.join(DATA_PATH, R_dic['test_path'])
train_path = 'Please fill in your datapath'
test_path = 'Please fill in your datapath'

R_dic = {}
R_dic['train_len'] = 1000
R_dic['test_len'] = 100
R_dic['resolution_datasets'] = 1023  # resolution of data sets
R_dic['epochs'] = 600
R_dic['batch_size'] = 4
R_dic['boundary_condition'] = 'dirichlet'
R_dic['region'] = [-1, 1, -1, 1]  # region [a,b,c,d] means [a,b]*[c,d]

R_dic['subsample_nodes'] = 4  # reslution = data.resolution/subsample_nodes
R_dic['subsample_attn'] = 8  # attention resolution = data.resolution/subsample_attn
R_dic['resolution_fine'] = int((R_dic['resolution_datasets']-1)/R_dic['subsample_nodes']+1)
R_dic['resolution_coarse'] = int((R_dic['resolution_datasets'] - 1) / R_dic['subsample_attn']+1)

R_dic['feature_dim'] = 128  # feature dim, in order to enhance expressiveness
R_dic['freq_dim'] = 32  # width of FNO
R_dic['modes'] = 12  # modes of FNO
R_dic['dim_feedforward'] = 256
R_dic['downscaler_size'] = [0.71094, 0.705]
R_dic['upscaler_size'] = [1.42188, 1.40660]
R_dic['downsample_mode'] = 'interp'
R_dic['upsample_mode'] = 'interp'
R_dic['window_size'] = [8, 8]
R_dic['depths'] = [2, 6]
R_dic['num_heads'] = [4, 4]

R_dic['learning_rate'] = 1e-3
R_dic['posadd1'] = True  # If ture, input = node+pos; Else input = node;
R_dic['posadd2'] = True  # If ture, FNO input = node+pos; Else input = node;
R_dic['downtype'] = 'keep'
R_dic['normalize_type'] = True

R_dic['model_save_path'] = MODEL_PATH
R_dic['model_name'] = 'Hnet2d'+ str(R_dic['resolution_fine'])+'_gamblet' +'_epoch' + str(R_dic['epochs']) \
                        + '_' + str(date.today()) + '.pt'
R_dic['result_name'] = str(R_dic['model_name'][0:-3]) + '.pkl'
#==========================================================================================

train_dataset = Datareader(data_path=train_path, res_grid_coarse=R_dic['resolution_coarse'],
                          res_grid_fine=R_dic['resolution_fine'], region=R_dic['region'],
                          train_data=True, res=R_dic['resolution_datasets'], data_len=R_dic['train_len'])

valid_dataset = Datareader(data_path=test_path,
                              res_grid_coarse=R_dic['resolution_coarse'], res_grid_fine=R_dic['resolution_fine'],
                              region=R_dic['region'], train_data=False, res=R_dic['resolution_datasets'],
                              data_len=R_dic['test_len'], normalizer_x=train_dataset.normalizer_x)
train_loader = DataLoader(train_dataset, batch_size=R_dic['batch_size'], shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=R_dic['batch_size'], shuffle=False, drop_last=False)

sample1 = next(iter(train_loader))
pos_coarse = sample1['pos'].to(device)
pos_fine = sample1['pos_fine'].to(device)
normalizer = train_dataset.normalizer_y.to(device)

model = Hnet2d(R_dic)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=R_dic['learning_rate'])
lr_scheduler = OneCycleLR(optimizer, max_lr=R_dic['learning_rate'], div_factor=1e4, final_div_factor=1e4,
                          pct_start=0.3, steps_per_epoch=len(train_loader), epochs=R_dic['epochs'])

h = (R_dic['region'][1] - R_dic['region'][0]) / R_dic['resolution_fine']
loss_func = Lossfunction(regularizer=True, h=h, gamma=0.5)
metric_func = Lossfunction(regularizer=False, h=h)
epochs = R_dic['epochs']

loss_train = []
loss_val = []
lossl2_epoch = []
reg_epoch = []
lr_history = []
stop_counter = 0
best_val_metric = np.inf
best_val_epoch = None
with tqdm(total=epochs) as pbar:
    for epoch in range(epochs):
        # ==================== train model ====================
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            node = batch["node"].to(device)
            u = batch['target'].to(device)
            gradu = batch['target_grad'].to(device)

            out = model(node, pos_fine=pos_fine, pos_coarse=pos_coarse, Normfunction=normalizer)
            loss_l2, reg, _ = loss_func(preds=out[..., 0], targets=u[..., 0], targets_prime=gradu)
            loss = loss_l2 + reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.99)
            optimizer.step()
            lr_scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            lr_history.append(lr)

            lossl2_epoch.append(loss_l2.item())
            reg_epoch.append(reg.item())

        _lossl2_mean = np.mean(lossl2_epoch)
        _reg_mean = np.mean(reg_epoch)
        loss_train.append([_lossl2_mean, _reg_mean])
        lossl2_epoch = []
        reg_epoch = []

        # ==================== compute test loss ====================
        model.eval()
        metric_val = []
        for _, data in enumerate(valid_loader):
            with torch.no_grad():
                x = data['node'].to(device)
                exact = data['target'].to(device)
                approx = model(x, pos_fine=pos_fine, pos_coarse=pos_coarse, Normfunction=normalizer)
                _, _, metric = metric_func(approx[..., 0], exact[..., 0])
                metric_val.append(metric)

        lossval = np.mean(metric_val, axis=0)
        loss_val.append(lossval)
        val_metric = lossval.sum()

        # ========================================
        if val_metric < best_val_metric:
            best_val_epoch = epoch
            best_val_metric = val_metric
            stop_counter = 0

            torch.save(model, os.path.join(R_dic['model_save_path'], R_dic['model_name']))

        else:
            stop_counter += 1

        # print('val metric:', val_metric)
        desc = color(f"| val metric: {val_metric:.3e} ", color=Colors.blue)
        desc += color(f"| best val: {best_val_metric:.3e} at epoch {best_val_epoch + 1}", color=Colors.yellow)
        desc += color(f" | early stop: {stop_counter} ", color=Colors.red)
        desc += color(f" | current lr: {lr:.3e}", color=Colors.magenta)

        desc_ep = color("", color=Colors.green)
        desc_ep += color(f"| L2 loss : {_lossl2_mean:.3e} ", color=Colors.green)
        desc_ep += color(f"| reg loss : {_reg_mean:.3e} ", color=Colors.green)

        desc_ep += desc
        pbar.set_description(desc_ep)
        pbar.update()

    result = dict(
        best_val_epoch=best_val_epoch,
        best_val_metric=best_val_metric,
        loss_train=np.asarray(loss_train),
        loss_val=np.asarray(loss_val),
        lr_history=np.asarray(lr_history),
        optimizer_state=optimizer.state_dict()
    )
    save_pickle(result, os.path.join(R_dic['model_save_path'], R_dic['result_name']))