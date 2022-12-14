import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from unet_model import UNet
from models import FMMTransformer, SpectralDecoder, HTransformer

import os, logging, copy
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from Adam import Adam
from utilities3 import *
from tqdm.auto import tqdm

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


torch.set_printoptions(threshold=100000)
      

    
def objective(dataOpt, FMM_paras, optimizerScheduler_args,
              show_conv=False, tqdm_disable=True, 
              log_if=False, parallel=False, 
              validate=False, generalization=False, model_type='FMM', 
              model_save=False, extrapolate=False):
    
    ################################################################
    # configs
    ################################################################

    print(os.path.basename(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = getSavePath(dataOpt['data'], model_type)
    MODEL_PATH_PARA = getSavePath(dataOpt['data'], model_type, flag='para')
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename=MODEL_PATH,
                    filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"FMM_paras={FMM_paras}")
    logging.info(f"dataOpt={dataOpt}")
    logging.info(f"optimizerScheduler_args={optimizerScheduler_args}")
  


    
    ################################################################
    # load data and data normalization
    ################################################################
   
    
    x_train, y_train, x_normalizer, y_normalizer = getDarcyDataSet(dataOpt, flag='train', return_normalizer=True)
    x_test, y_test = getDarcyDataSet(dataOpt, flag='test', return_normalizer=False, normalizer=x_normalizer)
    x_val, y_val = getDarcyDataSet(dataOpt, flag='val', return_normalizer=False, normalizer=x_normalizer)
    if generalization:
        x_gel, y_gel = getDarcyDataSet(dataOpt, flag='gel', return_normalizer=False, normalizer=x_normalizer)

    if extrapolate:
        dataOpt2 = copy.deepcopy(dataOpt)
        dataOpt2['sampling_rate'] //= 2
        _, y_test_extra = getDarcyDataSet(dataOpt2, flag='test', return_normalizer=False, normalizer=x_normalizer)
    
 

    if model_type in ('FMM', 'HMM', 'Unet'):
        x_train = x_train[:, np.newaxis, ...]
        x_test = x_test[:, np.newaxis, ...]
        x_val = x_val[:, np.newaxis, ...]  
        if generalization:
            x_gel = x_gel[:, np.newaxis, ...] 
    else:
        x_train = x_train[:, ..., np.newaxis]
        x_test = x_test[:, ..., np.newaxis]
        x_val = x_val[:, ..., np.newaxis] 
        if generalization:
            x_gel = x_gel[:, ..., np.newaxis]  

    
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train.contiguous().to(device), y_train.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test.contiguous().to(device), y_test.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val.contiguous().to(device), y_val.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)

    if extrapolate:
        test_extra_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test.contiguous().to(device), y_test_extra.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)

    if generalization:
        gel_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_gel.contiguous().to(device), y_gel.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)
  
    ################################################################
    # training and evaluation
    ################################################################
    FMM_paras['Decoder_paras']['resolution'] = y_train.size(1)
    FMM_paras['Decoder_paras']['normalizer'] =  y_normalizer
  
        
    if dataOpt['data'] in ('darcy', 'darcy20', 'darcy20c6', 'darcy15c10', 'darcy20c6_c3', 'helmholtz','a4f1'):
        if model_type == 'FMM':
            model = FMMTransformer(**FMM_paras).to(device)
        elif model_type == 'HMM':
            model = HTransformer(**FMM_paras).to(device)
        elif model_type == 'Unet':
            model = UNet(1,1,bilinear=True).to(device)
        elif model_type =='FNO':
            model = SpectralDecoder(**FMM_paras['Decoder_paras'])
        else: 
            raise NameError('invalid model type')

    elif dataOpt['data']=='navier':
        model = FMMTransformer(Decoder_paras, img_size=128, patch_size=3, in_chans=1,
                     embed_dim=Decoder_paras['width'], depths=[1, 1, 1], num_heads=[1, 1, 1],
                     window_size=[4, 4, 4], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                     norm_layer=nn.LayerNorm, ape=False, patch_norm=patch_norm,
                     use_checkpoint=False, stride=sampling_rate, patch_padding=1,
                      normalizer=None).to(device)
        
    else:
        model = FMMTransformer(Decoder_paras, img_size=1023, patch_size=4, in_chans=1,
                     embed_dim=Decoder_paras['width'], depths=[1, 1, 1], num_heads=[1, 1, 1],
                     window_size=[8, 8, 8], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                     norm_layer=nn.LayerNorm, ape=False, patch_norm=patch_norm,
                     use_checkpoint=False, stride=sampling_rate, 
                     normalizer=y_normalizer).to(device)
        

    
    if parallel:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = model.to(f'cuda:{model.device_ids[0]}')
    
    optimizer, scheduler = getOptimizerScheduler(model.parameters(), **optimizerScheduler_args)
    
    h1loss = HsLoss(d=2, p=2, k=1, size_average=False, res=y_train.size(1), a=dataOpt['loss_weight'])
    h1loss.cuda(device)
    # h1loss = HSloss_d()
    l2loss = LpLoss(size_average=False)  
    # l2loss_abs = LpLoss(size_average=False, relative=False)  
    ############################
    def train(train_loader):
        model.train()
        train_l2, train_h1 = 0, 0
        train_f_dist = torch.zeros(y_train.size(1))

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            if dataOpt['loss_type']=='h1':
                with torch.no_grad():
                    train_l2loss = l2loss(out, y)

                train_h1loss, train_f_l2loss, f_l2x, f_l2y = h1loss(out, y)
                train_h1loss.backward()
            else:
                with torch.no_grad():
                    train_h1loss, train_f_l2loss, f_l2x, f_l2y = h1loss(out, y)

                train_l2loss = l2loss(out, y)
                train_l2loss.backward()

            optimizer.step()
            train_h1 += train_h1loss.item()
            train_l2 += train_l2loss.item()
            train_f_dist += sum(torch.squeeze(torch.abs(f_l2x-f_l2y))).cpu()

        
        train_l2/= len(dataOpt['dataSize']['train'])
        train_h1/= len(dataOpt['dataSize']['train'])
        train_f_dist/= len(dataOpt['dataSize']['train'])
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        
        return lr, train_l2, train_h1, train_f_dist
            
    @torch.no_grad()
    def test(test_loader):
        model.eval()
        test_l2, test_h1 = 0., 0.

        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                test_l2 += l2loss(out, y).item()
                test_h1 += h1loss(out, y)[0].item()
                # logging.info(l2loss(out, y))
                # logging.info(h1loss(out, y)[1])
        test_l2/= len(dataOpt['dataSize']['test'])
        test_h1/= len(dataOpt['dataSize']['test'] )          
        

        return  test_l2, test_h1

    @torch.no_grad()
    def test_extra(test_extra_loader):
        model.eval()
        test_extra_l2, test_extra_h1 = 0., 0.

        with torch.no_grad():
            for x, y in test_extra_loader:
                out = model(x, (512, 512))
                test_extra_l2 += l2loss(out, y).item()
                test_extra_h1 += h1loss(out, y)[0].item()
                # logging.info(l2loss(out, y))
                # logging.info(h1loss(out, y)[1])
        test_extra_l2/= len(dataOpt['dataSize']['test'])
        test_extra_h1/= len(dataOpt['dataSize']['test'])         
        

        return  test_extra_l2, test_extra_h1
        
    ############################  
    ###### start to train ######
    ############################
    
    train_h1_rec, train_l2_rec, train_f_dist_rec, test_l2_rec, test_h1_rec = [], [], [], [], []
    val_l2_rec, val_h1_rec = [], [],
    if extrapolate:
        extra_l2_rec, extra_h1_rec = [], [],
    best_l2, best_test_l2, best_test_h1, arg_min_epoch = 1.0, 1.0, 1.0, 0  
    with tqdm(total=optimizerScheduler_args['epochs'], disable=tqdm_disable) as pbar_ep:
                            
        for epoch in range(optimizerScheduler_args['epochs']):
            desc = f"epoch: [{epoch+1}/{optimizerScheduler_args['epochs']}]"
            lr, train_l2, train_h1, train_f_dist = train(train_loader)
            test_l2, test_h1 = test(test_loader)
            val_l2, val_h1 = test(val_loader)
            if generalization:
                gel_l2, gel_h1 = test(gel_loader)
            train_l2_rec.append(train_l2); train_h1_rec.append(train_h1) 
            train_f_dist_rec.append(train_f_dist)
            test_l2_rec.append(test_l2); test_h1_rec.append(test_h1)
            val_l2_rec.append(val_l2); val_h1_rec.append(val_h1)

            if val_l2 < best_l2:
                best_l2 = val_l2
                arg_min_epoch = epoch
                best_test_l2 = test_l2
                best_test_h1 = test_h1
            if extrapolate:
                test_extra_l2, test_extra_h1 = test_extra(test_extra_loader)
                extra_l2_rec.append(test_extra_l2); extra_h1_rec.append(test_extra_h1)

            desc += f" | current lr: {lr:.3e}"
            desc += f"| train l2 loss: {train_l2:.3e} "
            desc += f"| train h1 loss: {train_h1:.3e} "
            desc += f"| test l2 loss: {test_l2:.3e} "
            desc += f"| test h1 loss: {test_h1:.3e} "
            desc += f"| val l2 loss: {val_l2:.3e} "
            desc += f"| val h1 loss: {val_h1:.3e} "
            if generalization:
                desc += f"| val l2 loss: {gel_l2:.3e} "
                desc += f"| val h1 loss: {gel_h1:.3e} "
            if extrapolate:
                desc += f"| extra l2 loss: {test_extra_l2:.3e} "
                desc += f"| extra h1 loss: {test_extra_h1:.3e} "
            pbar_ep.set_description(desc)
            pbar_ep.update()
            if log_if:
                logging.info(desc)  
        logging.info(f" test h1 loss: {best_test_h1:.3e}, test l2 loss: {best_test_l2:.3e}")
                 
        if log_if:
            logging.info('train l2 rec:')
            logging.info(train_l2_rec)
            logging.info('train h1 rec:')
            logging.info(train_h1_rec)
            logging.info('test l2 rec:')
            logging.info(test_l2_rec)
            logging.info('test h1 rec:')
            logging.info(test_h1_rec)
            if generalization:
                logging.info('gel l2 rec:')
                logging.info(gel_l2_rec)
                logging.info('gel h1 rec:')
                logging.info(gel_h1_rec)
            if extrapolate:
                logging.info('extra l2 rec:')
                logging.info(extra_l2_rec)
                logging.info('extra h1 rec:')
                logging.info(extra_h1_rec)
            logging.info('train_f_dist_rec')
            logging.info(torch.stack(train_f_dist_rec))
            


        if show_conv:
            plt.figure(1)
            plt.semilogy(np.array(train_l2_rec), label='train l2 loss')
            plt.semilogy(np.array(train_h1_rec), label='train h1 loss')
            plt.semilogy(np.array(test_l2_rec), label='test l2 loss')
            plt.semilogy(np.array(test_h1_rec), label='test h1 loss')
            if validate:
                plt.semilogy(np.array(val_l2_rec), label='valid l2 loss')
                plt.semilogy(np.array(val_h1_rec), label='valid h1 loss')
            plt.grid(True, which="both", ls="--")
            plt.legend()
            plt.show()

            temp = torch.stack(train_f_dist_rec[:, :100])
            plt.figure()
            plt.semilogy(temp[:, 0:6].detach().cpu().numpy())
            plt.grid(True, which="both", ls="--")
            plt.legend(range(6))
            plt.show()

            plt.figure()
            plt.semilogy(temp[:, 10:16].detach().cpu().numpy())
            plt.grid(True, which="both", ls="--")
            plt.legend(range(10,16))
            plt.show()
    if model_save:
        torch.save(model, MODEL_PATH_PARA)
        
    
            
    return test_l2


def ray_objective(opt, show_conv=False,   
tqdm_disable=True, parallel=False, validate=False, checkpoint_dir=None):
    
    ################################################################
    # configs
    ################################################################
    dataOpt = {'data': opt['data'], 'GN': opt['GN'], 
    'sampling_rate': opt['sampling_rate'],
    'dataSize': opt['dataSize'], 'sample_x': opt['sample_x'],
     'batch_size':opt['batch_size']
     }
    

    Decoder_paras={"modes": opt['modes'],
           "width": opt['width'],
           "padding": opt['padding'],        
           "kernel_type": opt['kernel_type'],
           "num_spectral_layers": opt['num_spectral_layers'],
           "init_scale": opt['init_scale'],
           "add_pos": opt['add_pos'],
          }

    FMM_paras = {    
            'img_size': opt['img_size'], 'patch_size': opt['patch_size'], 
            'in_chans':opt['in_chans'], 
            'embed_dim': Decoder_paras['width'], 'depths': opt['depths'], 
            'num_heads':opt['num_heads'],
            'window_size': opt['window_size'], 'mlp_ratio': opt['mlp_ratio'],
            'qkv_bias': opt['qkv_bias'], 'qk_scale': opt['qk_scale'],
            'norm_layer': opt['norm_layer'], 'patch_norm': opt['patch_norm'],
            'stride': dataOpt['sampling_rate'],
            'patch_padding': opt['patch_padding'], 
            'Decoder_paras': Decoder_paras,
            }

    optimizerScheduler_args = {
        "optimizer_type": opt['optimizer_type'],
        "lr": opt['lr'],
        "weight_decay": opt['weight_decay'],        
        "epochs": opt['epochs'],
        "final_div_factor": opt['final_div_factor'],  
        "div_factor": opt['div_factor'],
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    ################################################################
    # load data and data normalization
    ################################################################
    MODEL_PATH = getSavePath(dataOpt['data'], 'vFMM-')
    
    x_train, y_train, x_normalizer, y_normalizer = getDarcyDataSet(dataOpt, flag='train', return_normalizer=True)
    x_test, y_test = getDarcyDataSet(dataOpt, flag='test', return_normalizer=False, normalizer=x_normalizer)
    if validate:
        # _, VAL_PATH = getPath('darcy20c6_c3')
        x_val, y_val = getDarcyDataSet(dataOpt, flag='val', return_normalizer=False, normalizer=x_normalizer)
    
    # just for transformer
    
    x_train = x_train[:, np.newaxis, ...]
    x_test = x_test[:, np.newaxis, ...]
    if validate:
        x_val = x_val[:, np.newaxis, ...]   

        
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train.contiguous().to(device), y_train.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test.contiguous().to(device), y_test.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)
    if validate:
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val.contiguous().to(device), y_val.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)
    
    ################################################################
    # training and evaluation
    ################################################################
    FMM_paras['Decoder_paras']['resolution'] = y_train.size(1)

    model = FMMTransformer(**FMM_paras, normalizer=y_normalizer).to(device)
    
    if parallel:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = model.to(f'cuda:{model.device_ids[0]}')
    
    optimizer, scheduler = getOptimizerScheduler(model.parameters(), **optimizerScheduler_args)
    
    h1loss = HsLoss(d=2, p=2, k=1, size_average=False, res=y_train.size(1), a=[2.,])
    h1loss.cuda(device)
    l2loss = LpLoss(size_average=False)  
    l2loss_abs = LpLoss(size_average=False, relative=False)  
    
    ############################
    def train(train_loader):
        model.train()
        train_l2, train_h1 = 0, 0
        train_f_dist = torch.zeros(y_train.size(1))

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            if dataOpt['loss_type']=='h1':
                with torch.no_grad():
                    train_l2loss = l2loss(out, y)

                train_h1loss, train_f_l2loss, f_l2x, f_l2y = h1loss(out, y)
                train_h1loss.backward()
            else:
                with torch.no_grad():
                    train_h1loss, train_f_l2loss, f_l2x, f_l2y = h1loss(out, y)

                train_l2loss = l2loss_abs(out, y)
                train_l2loss.backward()

            optimizer.step()
            train_h1 += train_h1loss.item()
            train_l2 += train_l2loss.item()
            train_f_dist += sum(torch.squeeze(torch.abs(f_l2x-f_l2y))).cpu()

        
        train_l2/= dataOpt['dataSize']['train']
        train_h1/= dataOpt['dataSize']['train']
        train_f_dist/= dataOpt['dataSize']['train']
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        
        return lr, train_l2, train_h1, train_f_dist
            
    @torch.no_grad()
    def test(test_loader):
        model.eval()
        test_l2, test_h1 = 0., 0.

        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                test_l2 += l2loss(out, y).item()
                test_h1 += h1loss(out, y)[0].item()

        test_l2/= dataOpt['dataSize']['test']
        test_h1/= dataOpt['dataSize']['test']           
        
        
        return  test_l2, test_h1
        
    ############################  
    ###### start to train ######
    ############################
    
    train_h1_rec, train_l2_rec, train_f_dist_rec, test_l2_rec, test_h1_rec = [], [], [], [], []
    if validate:
        val_l2_rec, val_h1_rec = [], [],
    best_l2, arg_min_epoch = 0.0, 0
    with tqdm(total=optimizerScheduler_args['epochs'], disable=tqdm_disable) as pbar_ep:
                            
        for epoch in range(optimizerScheduler_args['epochs']):
            desc = f"epoch: [{epoch+1}/{optimizerScheduler_args['epochs']}]"
            lr, train_l2, train_h1, train_f_dist = train(train_loader)
            test_l2, test_h1 = test(test_loader)
            
            train_l2_rec.append(train_l2)
            train_h1_rec.append(train_h1) 
            train_f_dist_rec.append(train_f_dist)
            test_l2_rec.append(test_l2); test_h1_rec.append(test_h1)
            
            tune.report(loss=test_l2)
            
            if validate:
                val_l2, val_h1 = test(val_loader)
                val_l2_rec.append(val_l2); val_h1_rec.append(val_h1)
                if val_l2 < best_l2 - 0.0001:
                    best_l2 = val_l2
                    arg_min_epoch = epoch
            desc += f" | current lr: {lr:.3e}"
            desc += f"| train l2 loss: {train_l2:.3e} "
            desc += f"| train h1 loss: {train_h1:.3e} "
            desc += f"| test l2 loss: {test_l2:.3e} "
            desc += f"| test h1 loss: {test_h1:.3e} "
            if validate:
                desc += f"| val l2 loss: {val_l2:.3e} "
                desc += f"| val h1 loss: {val_h1:.3e} "
            pbar_ep.set_description(desc)
            pbar_ep.update()

        if show_conv:
            plt.figure(1)
            plt.semilogy(np.array(train_l2_rec), label='train l2 loss')
            plt.semilogy(np.array(train_h1_rec), label='train h1 loss')
            plt.semilogy(np.array(test_l2_rec), label='test l2 loss')
            plt.semilogy(np.array(test_h1_rec), label='test h1 loss')
            if validate:
                plt.semilogy(np.array(val_l2_rec), label='valid l2 loss')
                plt.semilogy(np.array(val_h1_rec), label='valid h1 loss')
            plt.grid(True, which="both", ls="--")
            plt.legend()
            plt.show()

            temp = torch.stack(train_f_dist_rec)
            plt.figure()
            plt.semilogy(temp[:, 0:6].detach().cpu().numpy())
            plt.grid(True, which="both", ls="--")
            plt.legend(range(6))
            plt.show()

            plt.figure()
            plt.semilogy(temp[:, 10:16].detach().cpu().numpy())
            plt.grid(True, which="both", ls="--")
            plt.legend(range(10,16))
            plt.show()

    print(f" train h1 loss: {train_h1:.3e}, test h1 loss: {test_h1:.3e}, train l2 loss: {train_l2 :.3e}, test l2 loss: {test_l2:.3e}")    
        
    return test_l2

    
def test_model(data, model_path, modes=12, width=32, mode_threshold=False, kernel_type='c', padding=9, init_scale=16, 
mlp_hidden_dim=128, num_spectral_layers=4, activation='gelu', add_pos=True, final_div_factor=1e1,  learning_rate=0.001,
 weight_decay=1e-4, batch_size=20, optimizer_type='Adam', show_conv=False,  sampling_rate=2, 
   GN=True, parallel=False):
    
    ################################################################
    # configs
    ################################################################
    Decoder_paras={"modes": modes,
           "width": width,
           "padding": padding,
           "mode_threshold": mode_threshold,
           "kernel_type": kernel_type,
           "num_spectral_layers": num_spectral_layers,
           "activation": activation,
           "mlp_hidden_dim": mlp_hidden_dim,
           "init_scale": init_scale,
           "add_pos": add_pos,
          }
    print(os.path.basename(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN_PATH, TEST_PATH = getPath(data)
    MODEL_PATH = getSavePath(data, 'vFMM-')
    

    ntrain = 1024
    ntest = 112

    dataOpt['epochs'] = dataOpt['epochs']
    step_size = 100
    gamma = 0.5
    
    r = sampling_rate

    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain,...]
    y_train = reader.read_field('sol')[:ntrain,::r,::r]
    s = y_train.size(1)
    # s = int(((1023 - 1)//r) + 1)
    
    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest,...]
    y_test = reader.read_field('sol')[:ntest,::r,::r]

    if GN:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    
    x_train = x_train[:, np.newaxis, ...]
    x_test = x_test[:, np.newaxis, ...]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train.contiguous().to(device), y_train.contiguous().to(device)), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test.contiguous().to(device), y_test.contiguous().to(device)), batch_size=batch_size, shuffle=False)

    
    ################################################################
    # training and evaluation
    ################################################################
    Decoder_paras['resolution'] = s
    if data=='darcy':
        model = FMMTransformer(Decoder_paras, img_size=421, patch_size=4, in_chans=1, num_classes=2,
                     embed_dim=Decoder_paras['width'], depths=[1, 2, 1], num_heads=[1, 1, 1],
                     window_size=[9, 4, 4], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                     norm_layer=nn.LayerNorm, ape=False, patch_norm=None,
                     use_checkpoint=False, stride=sampling_rate, patch_padding=6, normalizer=y_normalizer).to(device)
        
    else:
        # data in ('darcy20', 'darcy20c6', 'darcy15c10', 'darcy20c6_c3'):
        model = FMMTransformer(Decoder_paras, img_size=512, patch_size=4, in_chans=1, num_classes=2,
                     embed_dim=Decoder_paras['width'], depths=[1, 1, 1], num_heads=[1, 1, 1],
                     window_size=[4, 4, 4], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                     norm_layer=nn.LayerNorm, ape=False, patch_norm=None,
                     use_checkpoint=False, stride=sampling_rate, patch_padding=1, normalizer=y_normalizer).to(device)
      
    h1loss = HsLoss(d=2, p=2, k=1, size_average=False, res=s, a=[2.,])
    h1loss.cuda(device)
    l2loss = LpLoss(size_average=False) 
   
    
    # if checkpoint_dir:
    #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    
    #############################################
    ####### DEFINE TRAIN AND TEST FUNC #########
            
    @torch.no_grad()
    def test(test_loader):
        model.eval()
        test_l2, test_h1 = 0., 0.

        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                test_l2 += l2loss(out, y).item()
                test_h1 += h1loss(out, y)[0].item()

        test_l2 /= ntest
        test_h1/= ntest           

        return  test_l2
    ############################################
    test_l2 = test(test_loader)
    
        
            
    return test_l2


if __name__ == "__main__":

    import argparse
    import vFMM_multi

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--data", type=str, default="darcy", help="data name, darcy, darcy20, darcy20c6, darcy15c10, darcy20c6_c3"
    )
    parser.add_argument(
            "--model_path", type=str, default="darcy20c6_c3.pth", help="model path"
    )
    parser.add_argument(
            "--batch_size", type=int, default=4, help="batch size"
    )
    parser.add_argument(
            "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
            "--epochs", type=int, default=100, help="number of epochs"
    )
    parser.add_argument(
            "--sampling_rate", type=int, default=2, help="sampling rate"
    )
    parser.add_argument(
            "--checkpoint_dir", type=str, default=None, help="checkpoint directory"
    )
    parser.add_argument(
            "--normalizer_type", type=str, default="PGN", help="normalizer type, PGN, GN, etc"
    )
    parser.add_argument(
            "--loss_type", type=str, default="h1", help="loss type, l2, h1, etc"
    )
    parser.add_argument(
            "--optimizer_type", type=str, default="adam", help="optimizer type, adam, adamW, etc"
    )

    # decoder paras
    parser.add_argument(
        "--modes", type=int, default=12, help="the number of modes for fno decoder"
    )
    parser.add_argument(
        "--width", type=int, default=64, help="feature dimension"
    )
    parser.add_argument(
        "--num_spectral_layers", type=int, default=5, help="number of layers of decoder"
    )
    parser.add_argument(
        "--padding", type=int, default=5, help="padding in decoder"
    )
    parser.add_argument(
        "--kernel_type", type=str, default='c', help="pointwise or convolution in decoder"
    )
    parser.add_argument(
        "--add_pos", type=str, default=False, help="add position in decoder or not"
    )
    parser.add_argument(
        "--lift", type=str, default=False, help="lift in decoder or not"
    )
    parser.add_argument(
        "--activation", type=str, default='gelu', help="activation in decoder"
    )

    # FMM, HMM paras
    parser.add_argument(
        "--model_type", type=str, default="HMM", help="model type, FMM, HMM, etc"
    )
    # parser.add_argument(
    #     "--window_size", type=list, default=[4, 4, 4], help="window size in FMM"
    # )
    # parser.add_argument(
    #     "--depths", type=list, default=[1, 1, 1], help="depths in FMM"
    # )
    # parser.add_argument(
    #     "--num_heads", type=list, default=[1, 1, 1], help="num_heads in FMM"
    # )
    # parser.add_argument(    
    #     "--patch_size", type=int, default=4, help="patch_size in FMM"
    # )
    # parser.add_argument(
    #     "--in_chans", type=int, default=1, help="in_chans in FMM"
    # )
    # parser.add_argument(
    #     "--patch_padding", type=int, default=1, help="patch_padding in FMM" 
    # )
    parser.add_argument(
        "--qkv",   default=True, action=argparse.BooleanOptionalAction, help="qkv in FMM"
    )

    args = parser.parse_args()
    args = vars(args)
        

    Decoder_paras={  
                "modes": 12,
                "width": 64,
                "lift": False,
                "padding": 5,
                "mode_threshold": False,
                "kernel_type": 'c',
                "num_spectral_layers": 5,
                "activation": 'gelu',
                "mlp_hidden_dim": 128,
                "init_scale": 16,
                "add_pos": False,
                "shortcut": True,
                }
 

    if args['data']=='darcy':
        dataOpt = {}
        dataOpt['data'] = "darcy"
        dataOpt['GN'] = False
        dataOpt['sampling_rate'] = 2
        dataOpt['dataSize'] = {'train': range(1000), 'test': range(100), 'val':range(100, 200)}
        dataOpt['batch_size'] = 4
        dataOpt['sample_x'] = False
        dataOpt['loss_type']='h1'
        dataOpt['loss_weight'] = [2,]
        dataOpt['normalizer_type'] = 'PGN'
        FMM_paras = {    
                    'img_size': 421, 'patch_size': 6, 'in_chans':1, 
                    'embed_dim': Decoder_paras['width'], 'depths': [1, 2, 1], 
                    'num_heads':[1, 1, 1],
                    'window_size': [4, 4, 4], 'mlp_ratio': 4.,
                    'qkv':True, 'qkv_bias': False, 'qk_scale': None,
                    'patch_norm': False,
                    'stride': dataOpt['sampling_rate'], 'patch_padding': 0, 
                    'Decoder_paras': Decoder_paras,
                     }
  
        
    elif dataOpt['data'] == 'darcy20c6':
        dataOpt = {}
        dataOpt['data'] = "darcy20c6"
        dataOpt['GN'] = True
        dataOpt['sampling_rate'] = 2
        dataOpt['dataSize'] = {'train': range(1280), 'test': range(112), 'val':range(1280, 1280+112)}
        dataOpt['batch_size'] = 4
        dataOpt['sample_x'] = False
        dataOpt['loss_type']='h1'
        dataOpt['loss_weight'] = [2,]
        dataOpt['normalizer_type'] = 'PGN'
        FMM_paras = {    
                    'img_size': 512, 'patch_size': 4, 'in_chans':1, 
                    'embed_dim': Decoder_paras['width'], 'depths': [1, 1, 1], 
                    'num_heads':[1, 1, 1],
                    'window_size': [8, 8, 4], 'mlp_ratio': 4.,
                    'qkv':True, 'qkv_bias': False, 'qk_scale': None,
                    'patch_norm': True,
                    'stride': dataOpt['sampling_rate'], 'patch_padding': 1, 
                    'Decoder_paras': Decoder_paras,
                    }


    elif dataOpt['data'] == 'a4f1':
        dataOpt = {}
        dataOpt['data'] = "a4f1"
        dataOpt['GN'] = True
        dataOpt['sampling_rate'] = 4
        dataOpt['dataSize'] = {'train': range(1000), 'val': range(100), 'test': range(100)} #, 'val':112
        dataOpt['batch_size'] = 4
        dataOpt['sample_x'] = False
        dataOpt['loss_type']='h1'
        dataOpt['loss_weight'] = [2,]
        dataOpt['normalizer_type'] = 'GN'
        FMM_paras = {    
                    'img_size': 1023, 'patch_size': 4, 'in_chans':1, 
                    'embed_dim': Decoder_paras['width'], 'depths': [1, 1, 1], 
                    'num_heads':[1, 1, 1],
                    'window_size': [8, 8, 8], 'mlp_ratio': 4.,
                    'qkv':True, 'qkv_bias': False, 'qk_scale': None,
                    'patch_norm': False,
                    'stride': dataOpt['sampling_rate'], 'patch_padding': 1, 
                    'Decoder_paras': Decoder_paras,
                     }
       
    optimizerScheduler_args = {'optimizer_type': 'adam', 'lr': 8e-4, 'weight_decay': 0.0001, 'epochs': 100, 'final_div_factor': 20, 'div_factor': 4}

    optimizerScheduler_args, FMM_paras, dataOpt, Decoder_paras = merge_params(args, optimizerScheduler_args, FMM_paras, dataOpt, Decoder_paras)
    
    vFMM_multi.objective(dataOpt, FMM_paras, optimizerScheduler_args, model_type=args['model_type'],
    validate=True, generalization=False, tqdm_disable=True, log_if=True, 
    model_save=True, show_conv=False, parallel=True, extrapolate=False)



    # sampling rate 1
    # FMM_paras = {    
    #         'img_size': 512, 'patch_size': 3, 'in_chans':1, 
    #         'embed_dim': Decoder_paras['width'], 'depths': [1, 1, 1], 
    #         'num_heads':[1, 1, 1],
    #         'window_size': [4, 4, 4], 'mlp_ratio': 4.,
    #         'qkv_bias': False, 'qk_scale': None,
    #         'norm_layer': nn.LayerNorm, 'patch_norm': False,
    #         'stride': dataOpt['sampling_rate'],
    #         'patch_padding': 1, 
    #         'Decoder_paras': Decoder_paras,
    #         }
    # FMM_paras = {    
    #         'img_size': 1023, 'patch_size': 4, 'in_chans':1, 
    #         'embed_dim': Decoder_paras['width'], 'depths': [1, 1, 1], 
    #         'num_heads':[1, 1, 1],
    #         'window_size': [8, 8, 8], 'mlp_ratio': 4.,
    #         'qkv_bias': False, 'qk_scale': None,
    #         'norm_layer': nn.LayerNorm, 'patch_norm': True,
    #         'stride': dataOpt['sampling_rate'],
    #         'patch_padding': 2, 
    #         'Decoder_paras': Decoder_paras,
    #         }

    # darcy's
    # FMM_paras = {    
    #             'img_size': 421, 'patch_size': 6, 'in_chans':1, 
    #             'embed_dim': Decoder_paras['width'], 'depths': [1, 2, 1], 
    #             'num_heads':[1, 1, 1],
    #             'window_size': [4, 4, 4], 'mlp_ratio': 4.,
    #             'qkv_bias': False, 'qk_scale': None,
    #             'norm_layer': nn.LayerNorm, 'patch_norm': None,
    #             'stride': dataOpt['sampling_rate'], 'patch_padding': 0, 
    #             'Decoder_paras': Decoder_paras,
    #             }


    