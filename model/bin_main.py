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
 
        scheduler.step()
        
        epoch_start_time = time.time()
        
        val_infos = val(args, model, test_loader, criterion)
        
        epoch_end_time = time.time()
        
        tot_val_time.append(epoch_end_time-epoch_start_time)

        # save epoch info
        epoch_infos = epoch_info(args.num_classes, train_infos, val_infos)

        # print status and save info
        epoch_print_save(args, epoch, epoch_infos, optimizer.param_groups[0]['lr'], model.state_dict())
        print(f"training run time = {tot_train_time[-1]:.2f} | inference time = {tot_val_time[-1]:.2f}")
        
    print(f"total run time = {np.sum(tot_train_time):.2f} | total inference time = {np.sum(tot_val_time):.2f}")
    
    # final model information       
    save_info(model.state_dict(), args, epoch_infos)
    

if __name__ == '__main__':
    
    set_random_seed(2023)
    args = parse_arguments()
    training(args)