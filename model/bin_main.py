import torch
import pandas as pd
from scipy.io import arff

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import os
import time

# snntorch

from tqdm import tqdm
from snntorch import functional as SF
from sklearn.preprocessing import MinMaxScaler

from bin_config import parse_arguments, set_random_seed
from bin_utils import *
from bin_model import MODEL
from bin_train import train, val       

def training(args):
    
    # print(f"device using {args.device}")
    
    # transform = get_transform
    transform = None
    data_set = CustomDataset(root=args.data_root, transform=transform, class_imbalance=True) 
    
    args.data_num_steps = data_set[1][0].shape[0]
    
    model = MODEL[args.model_name](args).to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size)
    criterion = args.criterion()
    
    tq_epoch = tqdm(range(args.epoch), leave=False)
    
    loaders = data_loader(data_set=data_set, 
                          batch_size=args.batch_size, 
                          num_samples=args.num_samples, 
                          num_test_samples=args.num_test_samples)
        
    train_loader = loaders['train']; test_loader = loaders['val']
    
    tot_train_time = []; tot_val_time = []
    
    for epoch in tq_epoch:      
        
        tq_epoch.set_description(f"EPOCH {epoch+1} | ")
        
        epoch_start_time = time.time()
        
        train_infos = train(args, model, train_loader, criterion, optimizer) 
        
        epoch_end_time  = time.time()
        
        tot_train_time.append(epoch_end_time-epoch_start_time)
        
        # print(f"train spk count {(train_infos['class_spike_count'][0]+train_infos['class_spike_count'][1])/25000}")
        
        scheduler.step()
        
        epoch_start_time = time.time()
        
        val_infos = val(args, model, test_loader, criterion)
        
        epoch_end_time = time.time()
        
        tot_val_time.append(epoch_end_time-epoch_start_time)
        
        # print(f"test spk count {(val_infos['t_class_spike_count'][0] + val_infos['t_class_spike_count'][1])/5000}")
        

        # save epoch info
        epoch_infos = epoch_info(args.num_classes, train_infos, val_infos)

        # print status and save info
        epoch_print_save(args, epoch, epoch_infos, optimizer.param_groups[0]['lr'], model.state_dict())
        print(f"training run time = {tot_train_time[-1]:.2f} | inference time = {tot_val_time[-1]:.2f}")
        
    print(f"total run time = {np.sum(tot_train_time):.2f} | total inference time = {np.sum(tot_val_time):.2f}")
    
    # final model information       
    save_info(model.state_dict(), args, epoch_infos)

    
# def test(**kargs):
#     ''' 
#     Test method for visualization and checking results
#     ---
#     - args : arguments,
#     - model_root : already trained model state dict,
#     - show_frame : if you want to check data details 
    
#     if you want to test model for data set, we need:  
#         - need_loader=True,
#         - data_root : csv file root, 
        
#     else:  
#         - need_loader=False,
#         - sample : one sample of data(A returned tensor of dataset class)
        
#     '''
    
#     args = kargs['args']
#     model_root = kargs['model_root']
#     need_loader = kargs['need_loader']
#     show_frame = kargs['show_frame'] if kargs['show_frame'] else None
    
#     trained_model_info = torch.load(model_root, map_location=args.device)
    
#     model = MODEL[args.model_name](args).to(args.device)
#     model.load_state_dict(trained_model_info['model_state'])
    
#     if need_loader:
#         data_root = kargs['data_root']
        
#         test_set = CustomDataset(data_root, class_imbalance=False, transform=None)
#         test_loader = torch.utils.data.DataLoader(test_set, batchsize=1, shuffle=False)
        
#         with torch.no_grad():
#             model.eval()
#             for img, label in test_loader:
                
#                 img = img.to(args.device)
#                 label = label.to(args.device)
                
#                 spk_rec, _ = model(img)
                
#                 _, predicted_class = spk_rec.sum(dim=0).max(1)
        
#                 for i in range(len(predicted_class)): print(f"Model predict {predicted_class[i]} (target class : {label[i]})")
        
#     else:
#         sample = kargs['sample']

#         X = sample[0]; y = sample[1]
        
#         if show_frame: show_data_frame(X, y)
        
#         with torch.no_grad():
#             model.eval()
            
#             X = X.to(args.device)
#             y = y.to(args.device)
            
#             spk_rec, _ = model(X)
            
#             _, predicted_class = spk_rec.sum(0).max(1)
            
#             plot_data(X.cpu().numpy().squeeze(0), predicted_class, y)


def test(args):
    data_root = args.test_data_root
    
    test_set = CustomDataset(data_root, class_imbalance=True, transform=None)
    test_loader = torch.utils.data.DataLoader(test_set, batchsize=1, shuffle=False)
    
    args.data_num_steps = test_set[1][0].shape[0]
    
    model_root = args.model_root
    
    trained_model_info = torch.load(model_root, map_location=args.device)
    
    model = MODEL[args.model_name](args).to(args.device)
    model.load_state_dict(trained_model_info['model_state'])

    
    criterion = args.criterion()
    
    test_info = val(args, model, test_loader, criterion)
        
    
    

if __name__ == '__main__':
    
    set_random_seed(2023)
    args = parse_arguments()
    training(args)