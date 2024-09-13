import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.classification import MulticlassAccuracy

import pandas as pd
from scipy.io import arff

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import os
import time
import datetime
from tqdm import tqdm

# snntorch
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils

from bin_config import parse_arguments
from bin_utils import *


def train(args, model, train_loader, criterion, optimizer):
    
    model.train()
        
    train_loss = []
    class_acc = {i: 0 for i in range(args.num_classes)}; 
    class_count = {i: 0 for i in range(args.num_classes)};
    class_spike_count = {i: 0 for i in range(args.num_classes)}
    
    # train
    for data, label in tqdm(train_loader, desc='loader | '):
        data = data.to(args.device) # [batch size, len_data]
        label = label.to(args.device) # [batch size, ]
        # label = F.one_hot(label, num_classes=args.num_classes).to(torch.float32) # [batch size, 5]
        
        args.data_num_steps = data.shape[1]

        spk_rec, spike_count_rec = model(data)
        
        loss = criterion(spk_rec, label)
        
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.neural_encoding == 'TTFS':
            acc, count, spike_count = class_split_acc_ttfs(spk_rec, spike_count_rec, label, args.num_classes)
        else:
            acc, count, spike_count = class_split_acc(spk_rec, spike_count_rec, label, args.num_classes)
        
        for i in range(args.num_classes):
            class_acc[i] += acc[i]
            class_count[i] += count[i]
            class_spike_count[i] += spike_count[i]
        
        train_loss.append(loss.item())
    
    return {'train_loss' : train_loss, 
            'class_acc' : class_acc,
            'class_count' : class_count,
            'class_spike_count' : class_spike_count}
    
def val(args, model, test_loader, criterion):
    
    test_loss = []
    t_class_acc = {i: 0 for i in range(args.num_classes)}
    t_class_count = {i: 0 for i in range(args.num_classes)}
    t_class_spike_count = {i: 0 for i in range(args.num_classes)}
    
    spk_count = 0

    with torch.no_grad():
        model.eval()
        for t_data, t_label in test_loader:
            t_data = t_data.to(args.device)
            t_label = t_label.to(args.device)
            
            # t_label = F.one_hot(t_label, num_classes=args.num_classes).to(torch.float32)
            
            t_spk_rec, t_spike_count_rec = model(t_data)  
            
            # print(torch.sum(t_spike_count_rec, dim=0)[0])
            
            spk_count += torch.mean(t_spike_count_rec)
            
            t_loss = criterion(t_spk_rec, t_label)
            
            test_loss.append(t_loss.item())
            
            if args.neural_encoding == 'TTFS':
                t_acc, t_count, t_spike_count = class_split_acc_ttfs(t_spk_rec, t_spike_count_rec, t_label, args.num_classes)
            else:
                t_acc, t_count, t_spike_count = class_split_acc(t_spk_rec, t_spike_count_rec, t_label, args.num_classes)
            
            for i in range(args.num_classes):
                t_class_acc[i] += t_acc[i]
                t_class_count[i] += t_count[i]
                t_class_spike_count[i] += t_spike_count[i]
                
        print(spk_count/len(test_loader))
                
    return {'test_loss': test_loss,
            't_class_acc' : t_class_acc,
            't_class_count' : t_class_count,
            't_class_spike_count' : t_class_spike_count}


    