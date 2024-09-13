import torch
import pandas as pd
from scipy.io import arff

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import os
import time
import datetime

# snntorch

from tqdm import tqdm
from snntorch import functional as SF
from sklearn.preprocessing import MinMaxScaler

from bin_config import parse_arguments, set_random_seed
from bin_utils import *
from bin_model import MODEL
from bin_train import val    

def test(args):
    
    model_root = f"log/{args.date_dir}/{args.model_name}/model_info_save_dict/{args.model_name}_{args.date_dir}.pth"
    data_root = args.test_data_root
    
    test_set = TestCustomDataset(data_root, class_imbalance=True, transform=None)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # args.data_num_steps = test_set[1][0].shape[0]
    args.data_num_steps = test_set[1][0].shape[0]

    trained_model_info = torch.load(model_root, map_location=args.device)
    
    model = MODEL[args.model_name](args).to(args.device)
    model.load_state_dict(trained_model_info['model_state'])

    criterion = args.criterion()
    
    test_start_time = time.time()

    test_info = val(args, model, test_loader, criterion)
    
    test_end_time = time.time()
    
    print(f"test spk count {(test_info['t_class_spike_count'][0] + test_info['t_class_spike_count'][1])/6000}")
    print(f"inference time = {datetime.timedelta(seconds=test_end_time-test_start_time)}")
    
    
    test_epoch_infos = test_epoch_info(2, test_info)
    
    t_epoch_acc = test_epoch_infos['t_epoch_acc']
    t_epoch_spike_count = test_epoch_infos['t_epoch_spike_count']
    
    print(" * [val_acc]\t", end="")
    # validation score
    for i in range(2):print(f"{i} {t_epoch_acc[i][-1]*100:.2f}% | ", end="")
    
    t_total_acc = 0
    for i in range(2): t_total_acc += t_epoch_acc[i][-1]
    print(f" >> total {(t_total_acc*100)/2:.2f}%")
    
    print()
    
    print(" * [val_spike_count]\t", end="")
    # validation score
    for i in range(2):print(f"{i} {t_epoch_spike_count[i][-1]} | ", end="")
    
    t_total_spike_count = 0
    for i in range(2): t_total_spike_count += t_epoch_spike_count[i][-1]
    print(f" >> total {(t_total_spike_count/2)}")
    
    print()
    
    
    
    
        

def test_epoch_info(num_classes, test_infos):
    
    epoch_infos = {
        't_epoch_acc' : {i: list() for i in range(num_classes)},
        't_epoch_spike_count' : {i: list() for i in range(num_classes)},
    }
    
    for i in range(num_classes):
        # accuracy
        if test_infos['t_class_acc'][i]:
            epoch_infos['t_epoch_acc'][i].append(test_infos['t_class_acc'][i]/test_infos['t_class_count'][i])
        else: 
            epoch_infos['t_epoch_acc'][i].append(0)
        
        # spike count
        if test_infos['t_class_spike_count'][i]:
            epoch_infos['t_epoch_spike_count'][i].append(test_infos['t_class_spike_count'][i]/test_infos['t_class_count'][i])
        else: 
            epoch_infos['t_epoch_spike_count'][i].append(0)
        
    return epoch_infos
    

if __name__ == '__main__':
    
    set_random_seed(2023)
    args = parse_arguments()
    test(args)