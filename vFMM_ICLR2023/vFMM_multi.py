import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# from unet_model import UNet
from models import FMMTransformer

import os, logging
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from Adam import Adam
from utilities3 import *
from tqdm.auto import tqdm

# import ray
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

torch.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(threshold=100000)
      

    
def objective(dataOpt, FMM_paras, optimizerScheduler_args,
              show_conv=False,  loss_type='h1', 
              tqdm_disable=True, log_if=False, parallel=False, validate=False, model_type='FMM'):
    
    ################################################################
    # configs
    ################################################################

    print(os.path.basename(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = getSavePath(dataOpt['data'], 'vFMM-')
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename=MODEL_PATH,
                    filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(FMM_paras)
    logging.info(dataOpt)
    logging.info(optimizerScheduler_args)


    
    ################################################################
    # load data and data normalization
    ################################################################
   
    
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
    FMM_paras['FNO_paras']['resolution'] = y_train.size(1)

    # if dataOpt['data']=='darcy':
    #     model = FMMTransformer(FNO_paras, img_size=421, patch_size=4, in_chans=1, 
    #                  embed_dim=FNO_paras['width'], depths=[1, 2, 1], num_heads=[1, 1, 1],
    #                  window_size=[9, 4, 4], mlp_ratio=4., qkv_bias=False, qk_scale=None,
    #                  norm_layer=nn.LayerNorm, ape=False, patch_norm=patch_norm,
    #                  use_checkpoint=False, stride=sampling_rate, patch_padding=6, 
    #                  normalizer=y_normalizer).to(device)
        
    if dataOpt['data'] in ('darcy', 'darcy20', 'darcy20c6', 'darcy15c10', 'darcy20c6_c3'):
        if model_type == 'FMM':
            model = FMMTransformer(**FMM_paras, normalizer=y_normalizer).to(device)
        elif model_type == 'Unet':
            model = UNet(1,1,bilinear=True).to(device)
        else: 
            raise NameError('invalid model type')

    elif dataOpt['data']=='navier':
        model = FMMTransformer(FNO_paras, img_size=128, patch_size=3, in_chans=1,
                     embed_dim=FNO_paras['width'], depths=[1, 1, 1], num_heads=[1, 1, 1],
                     window_size=[4, 4, 4], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                     norm_layer=nn.LayerNorm, ape=False, patch_norm=patch_norm,
                     use_checkpoint=False, stride=sampling_rate, patch_padding=1,
                      normalizer=None).to(device)
        
    else:
        model = FMMTransformer(FNO_paras, img_size=1023, patch_size=4, in_chans=1,
                     embed_dim=FNO_paras['width'], depths=[1, 1, 1], num_heads=[1, 1, 1],
                     window_size=[8, 8, 8], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                     norm_layer=nn.LayerNorm, ape=False, patch_norm=patch_norm,
                     use_checkpoint=False, stride=sampling_rate, 
                     normalizer=y_normalizer).to(device)
        

    
    if parallel:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = model.to(f'cuda:{model.device_ids[0]}')
    
    optimizer, scheduler = getOptimizerScheduler(model.parameters(), **optimizerScheduler_args)
    
    h1loss = HsLoss(d=2, p=2, k=1, size_average=False, res=y_train.size(1), a=[2.,])
    h1loss.cuda(device)
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
            if loss_type=='h1':
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
        
    with tqdm(total=optimizerScheduler_args['epochs'], disable=tqdm_disable) as pbar_ep:
                            
        for epoch in range(optimizerScheduler_args['epochs']):
            desc = f"epoch: [{epoch+1}/{optimizerScheduler_args['epochs']}]"
            lr, train_l2, train_h1, train_f_dist = train(train_loader)
            test_l2, test_h1 = test(test_loader)
            
            train_l2_rec.append(train_l2)
            train_h1_rec.append(train_h1) 
            train_f_dist_rec.append(train_f_dist)
            test_l2_rec.append(test_l2); test_h1_rec.append(test_h1)
            
            if validate:
                val_l2, val_h1 = test(val_loader)
                val_l2_rec.append(val_l2); val_h1_rec.append(val_h1)
                
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
            if log_if:
                logging.info(desc)           
        if log_if:
            logging.info('train l2 rec:')
            logging.info(train_l2_rec)
            logging.info('train h1 rec:')
            logging.info(train_h1_rec)
            logging.info('test l2 rec:')
            logging.info(test_l2_rec)
            logging.info('test h1 rec:')
            logging.info(test_h1_rec)
            if validate:
                logging.info('val l2 rec:')
                logging.info(val_l2_rec)
                logging.info('val h1 rec:')
                logging.info(val_h1_rec)
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
    MODEL_PATH = '/dssg/home/acct-nislz/nislz/xubo/vFMM/model/checkboard.pt'
    torch.save(model, MODEL_PATH)
    print(f" test h1 loss: {test_h1:.3e}, test l2 loss: {test_l2:.3e}")
            
    return test_l2


    
def test_model(data, model_path, modes=12, width=32, mode_threshold=False, kernel_type='c', padding=9, init_scale=16, 
mlp_hidden_dim=128, num_spectral_layers=4, activation='gelu', add_pos=True, final_div_factor=1e1,  learning_rate=0.001,
 weight_decay=1e-4, batch_size=20, optimizer_type='Adam', show_conv=False,  loss_type='h1', sampling_rate=2, 
   GN=True, parallel=False):
    
    ################################################################
    # configs
    ################################################################
    FNO_paras={"modes": modes,
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
    FNO_paras['resolution'] = s
    if data=='darcy':
        model = FMMTransformer(FNO_paras, img_size=421, patch_size=4, in_chans=1, num_classes=2,
                     embed_dim=FNO_paras['width'], depths=[1, 2, 1], num_heads=[1, 1, 1],
                     window_size=[9, 4, 4], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                     norm_layer=nn.LayerNorm, ape=False, patch_norm=None,
                     use_checkpoint=False, stride=sampling_rate, patch_padding=6, normalizer=y_normalizer).to(device)
        
    else:
        # data in ('darcy20', 'darcy20c6', 'darcy15c10', 'darcy20c6_c3'):
        model = FMMTransformer(FNO_paras, img_size=512, patch_size=4, in_chans=1, num_classes=2,
                     embed_dim=FNO_paras['width'], depths=[1, 1, 1], num_heads=[1, 1, 1],
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

 
    # dataOpt = {}
    # dataOpt['data'] = "darcy20c6"
    # dataOpt['GN'] = False
    # dataOpt['sampling_rate'] = 1
    # dataOpt['dataSize'] = {'train': 1280, 'test': 112, 'val':112}
    # dataOpt['batch_size'] = 8

    # FNO_paras={  
    #             "modes": 12,
    #             "width": 64,
    #             "padding": 5,
    #             "mode_threshold": False,
    #             "kernel_type": 'c',
    #             "num_spectral_layers": 5,
    #             "activation": 'gelu',
    #             "mlp_hidden_dim": 128,
    #             "init_scale": 16,
    #             "add_pos": False,
    #             }


    # # parser = argparse.ArgumentParser()
    # # parser.add_argument(
    # #         "--optimizer_type", type=str, default="adam", help="optimizer type, adam, adamW, etc"
    # #                     )
    # # args = parser.parse_args()
    # # optimizerScheduler_args = vars(args)

    # # parser = argparse.ArgumentParser()
    # # parser.add_argument(
    # #     "--modes", type=int, default=12, help="the number of modes for fno decoder"
    # # )
    # # parser.add_argument(
    # #     "--width", type=int, default=64, help="feature dimension"
    # # )
    # # parser.add_argument(
    # #     "--num_spectral_layers", type=int, default=5, help="number of layers of fno decoder"
    # # )
    # # parser.add_argument(
    # #     "--padding", type=int, default=5, help="padding in fno decoder"
    # # )
    # # parser.add_argument(
    # #     "--kernel_type", type=str, default='c', help="pointwise or convolution in fno decoder"
    # # )
    # # parser.add_argument(
    # #     "--add_pos", type=str, default=False, help="add position in fno decoder or not"
    # # )

    # if dataOpt['data']=='darcy':
    #     FMM_paras = {    
    #                 'img_size': 421, 'patch_size': 4, 'in_chans':1, 
    #                 'embed_dim': FNO_paras['width'], 'depths': [1, 2, 1], 
    #                 'num_heads':[1, 1, 1],
    #                 'window_size': [9, 4, 4], 'mlp_ratio': 4.,
    #                 'qkv_bias': False, 'qk_scale': None,
    #                 'norm_layer': nn.LayerNorm, 'patch_norm': False,
    #                 'stride': dataOpt['sampling_rate'],
    #                 'patch_padding': 6, 
    #                 'FNO_paras': FNO_paras,
    #                  }
        
    # elif dataOpt['data'] in ('darcy20', 'darcy20c6', 'darcy15c10', 'darcy20c6_c3'):
    #     FMM_paras = {    
    #                 'img_size': 512, 'patch_size': 3, 'in_chans':1, 
    #                 'embed_dim': FNO_paras['width'], 'depths': [1, 1, 1], 
    #                 'num_heads':[1, 1, 1],
    #                 'window_size': [4, 4, 4], 'mlp_ratio': 4.,
    #                 'qkv_bias': False, 'qk_scale': None,
    #                 'norm_layer': nn.LayerNorm, 'patch_norm': False,
    #                 'stride': dataOpt['sampling_rate'],
    #                 'patch_padding': 1, 
    #                 'FNO_paras': FNO_paras,
    #                  }
     

        
    # else:
    #     FMM_paras = {    
    #                 'img_size': 1023, 'patch_size': 4, 'in_chans':1, 
    #                 'embed_dim': FNO_paras['width'], 'depths': [1, 1, 1], 
    #                 'num_heads':[1, 1, 1],
    #                 'window_size': [8, 8, 8], 'mlp_ratio': 4.,
    #                 'qkv_bias': False, 'qk_scale': None,
    #                 'norm_layer': nn.LayerNorm, 'patch_norm': False,
    #                 'stride': dataOpt['sampling_rate'],
    #                 'patch_padding': 1, 
    #                 'FNO_paras': FNO_paras,
    #                  }
      

    
    
    

    # optimizerScheduler_args = {
    #     "optimizer_type": 'adam',
    #     "lr": 8e-4,
    #     "weight_decay": 1e-4,        
    #     "epochs": 100,
    #     "final_div_factor": 1e1,  

    #     # "loss_type": 'h1',
    #     # "show_conv": True,  
    #     # "tqdm_disable": False,
    #     # "parallel": False,
    #     # "validate": False,
    #     }


    # objective(dataOpt, FMM_paras, optimizerScheduler_args, show_conv=True, tqdm_disable=False, model_type='Unet')
    import vFMM_multi
    
    dataOpt = {}
    dataOpt['data'] = "darcy20c6"
    dataOpt['GN'] = False
    dataOpt['sampling_rate'] = 2
    dataOpt['dataSize'] = {'train': 1280, 'test': 112, 'val':112}
    dataOpt['batch_size'] = 8
    dataOpt['sample_x'] = False

    # # darcy best parameters
    # dataOpt = {}
    # dataOpt['data'] = "darcy"
    # dataOpt['GN'] = True
    # dataOpt['sampling_rate'] = 2
    # dataOpt['dataSize'] = {'train': 1000, 'test': 100}
    # dataOpt['batch_size'] = 20
    # dataOpt['sample_x'] = False

   

    FNO_paras={  
                "modes": 12,
                "width": 64,
                "padding": 5,
                "mode_threshold": False,
                "kernel_type": 'c',
                "num_spectral_layers": 5,
                "activation": 'gelu',
                "mlp_hidden_dim": 128,
                "init_scale": 16,
                "add_pos": False,
                }


    FMM_paras = {    
                'img_size': 512, 'patch_size': 4, 'in_chans':1, 
                'embed_dim': FNO_paras['width'], 'depths': [1, 1, 1], 
                'num_heads':[1, 1, 1],
                'window_size': [4, 4, 4], 'mlp_ratio': 4.,
                'qkv_bias': False, 'qk_scale': None,
                'norm_layer': nn.LayerNorm, 'patch_norm': False,
                'stride': dataOpt['sampling_rate'], 'patch_padding': 1, 
                'FNO_paras': FNO_paras,
                }

    # # darcy's
    # FMM_paras = {    
    #             'img_size': 421, 'patch_size': 6, 'in_chans':1, 
    #             'embed_dim': FNO_paras['width'], 'depths': [1, 2, 1], 
    #             'num_heads':[1, 1, 1],
    #             'window_size': [4, 4, 4], 'mlp_ratio': 4.,
    #             'qkv_bias': False, 'qk_scale': None,
    #             'norm_layer': nn.LayerNorm, 'patch_norm': None,
    #             'stride': dataOpt['sampling_rate'], 'patch_padding': 0, 
    #             'FNO_paras': FNO_paras,
    #             }


    optimizerScheduler_args = {
            "optimizer_type": 'adam',
            "lr": 8e-4,
            "weight_decay": 1e-4,        
            "epochs": 100,
            "final_div_factor": 8,  
            "div_factor": 4,
            }
    
    
    vFMM_multi.objective(dataOpt, FMM_paras, optimizerScheduler_args, loss_type='h1', validate=False, tqdm_disable=True, log_if=True, show_conv=False)