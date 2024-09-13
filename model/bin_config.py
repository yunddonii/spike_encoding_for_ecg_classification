import torch
import pandas as pd
from scipy.io import arff

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import time
import datetime
from tqdm import tqdm

import os

import argparse
import random

import snntorch.functional as SF

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# def parse_arguments(args):
def parse_arguments():
    
    parser = argparse.ArgumentParser(description='the hyperparameters for training')
    
    # parser = argparse.Namespace(args)
    
    ####### neural encoding #########
    
    parser.add_argument('-ne', '--neural_encoding', dest='neural_encoding', nargs='?', default=None, 
                        help='neural encoding : [brust, ttfs, hsa, bsa, rate(possion), None] \
                        \n\tNone is to use just data not encoded (default: %(default)s)')
    
    parser.add_argument('-n', '--neuron', dest='neuron', nargs='?', type=str, default='lif', help='kind of neuron : [lif, syn] (default: %(default)s)')
    parser.add_argument('--bias', dest='bias', nargs='?', type=bool, default=True, help='bias in neuron (default: %(default)s)')
    
    # filter parameter
    parser.add_argument('-f1', '--fil_args1', dest='fil_args1', nargs='?', type=int, help='filter window size')
    parser.add_argument('-f2', '--fil_args2', dest='fil_args2', nargs='?', type=int)
    parser.add_argument('-na', '--new_amp', dest='new_amp', nargs='?', type=float)
    parser.add_argument('-th', '--threshold', dest='threshold', nargs='?', type=float, help='threshold')
    parser.add_argument('--tau', dest='tau', nargs='?', type=int, help='tau for ttfs coding')
    parser.add_argument('--beta', dest='beta', nargs='?', type=float, help='beta for burst coding')

    # parser.add_argument('-m', '--model', dest='model_name', nargs='?', choices=['SNN', 'SNNT'], required=True,
    #                     help='ANN or SNN')
    
    parser.add_argument('-s', '--save', dest='save', nargs='?', type=bool, default=False,
                        help='if you want to save model state : True\nelse : False (default: %(default)s)')
    
    parser.add_argument('-e', '--epoch', dest='epoch', nargs='?', type=int,
                        help='# of total epoch')
    
    parser.add_argument('-nd', '--num_device', dest='num_device', nargs='?', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Device number : 0 or 1 (default: %(default)s)')
    
    parser.add_argument('-b', '--batch_size', dest='batch_size', nargs='?', type=int,
                        help='Batch size')
    
    parser.add_argument('-lr', '--learning_rate', dest='lr', nargs='?', type=float,
                        help='Learning rate')
    
    parser.add_argument('--print_epoch', nargs='?', type=int,
                        help='epoch number to want to print infomation (default: %(default)s)')
    
    parser.add_argument('--log_dir', nargs='?', default='log',
                        help='The directory name for saving log file \n\t default: %(default)s')
    
    parser.add_argument('--data_root', nargs='?', type=argparse.FileType('r'), default='../data/bin_3_mitbih_train.csv',
                        help='The root of train data root saved \n\t default: %(default)s')
    
    parser.add_argument('--test_data_root', nargs='?', type=argparse.FileType('r'), default='../data/bin_3_mitbih_test.csv',
                        help='The root of test data root saved \n\t default: %(default)s')
    
    parser.add_argument('-ns','--num_samples', dest='num_samples', nargs='?', type=int, default=25000,
                        help='# of samples to train (default: %(default)s)')
    
    parser.add_argument('-nts','--num_test_samples', dest='num_test_samples', nargs='?', type=int, default=5000,
                        help='# of samples to test (default: %(default)s)') 
    
    parser.add_argument('-t', '--time_step', dest='time_step', nargs='?', type=int, default=100, 
                        help='Total time steps for snn simulation (default: %(default)s)')
    
    parser.add_argument('--step_size', dest='step_size', nargs='?', type=int)
    
    parser.add_argument('--model_root', nargs='?', type=argparse.FileType('r'))


    config = parser.parse_args()
    args = Config(config)
    
    args.print_info()
    
    return args


class Config():
    def __init__(self, args):
        
        self.neural_encoding    = args.neural_encoding.upper() if args.neural_encoding else "vanilla"
        
        self.neuron = args.neuron
        if self.neuron == 'lif' : 
            self.model_name         = self.neural_encoding +'_SNN'
        elif self.neuron == 'syn' : 
            self.model_name         = self.neural_encoding + '_SYN_SNN'
            
        self.device             = torch.device(f'cuda:{args.num_device}' if torch.cuda.is_available() else 'cpu')

        # log / model weights save
        self.save               = args.save
        self.log_dir            = args.log_dir
        self.data_root          = args.data_root
        self.date_dir           = time.strftime("%y%m%d", time.localtime(time.time()))
        self.test_data_root     = args.test_data_root
        self.model_root         = "model_info_save_dict" if not args.model_root else args.model_root
        self.save_log_path      = os.path.join(self.log_dir, self.date_dir, self.model_name)
        
        os.makedirs(f"{self.log_dir}/{self.date_dir}/{self.model_name}/{self.model_root}", exist_ok=True)
        os.makedirs(os.path.join(self.save_log_path, "epoch_save"), exist_ok=True)
        
        self.date               = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
        
        # data
        self.num_samples        = int(args.num_samples)
        self.num_test_samples   = int(args.num_samples*0.15) if not args.num_test_samples else args.num_test_samples
        self.num_classes        = 2
        
        # training parameter
        self.bias               = args.bias
        self.epoch              = int(50) if not args.epoch else args.epoch
        self.lr                 = 1e-4 if not args.lr else args.lr
        self.step_size          = int(7) if not args.step_size else args.step_size
        self.batch_size         = int(16) if not args.batch_size else args.batch_size
        
        self.criterion          = self.select_criterion()
        
        # snn parameter
        self.beta               = 0.5
        self.num_steps          = args.time_step
        self.print_epoch        = int(5) if not args.print_epoch else args.print_epoch
        
        
        if self.neural_encoding == 'HSA':
            self.fil_args1 = 5 if not args.fil_args1 else args.fil_args1
            self.fil_args2 = 3 if not args.fil_args2 else args.fil_args2
            self.new_amp = 0.09 if not args.new_amp else args.new_amp
        
        # elif self.neural_encoding == 'MHSA':
        #     self.fil_args1 = 5 if not args.fil_args1 else args.fil_args1
        #     self.fil_args2 = 3 if not args.fil_args2 else args.fil_args2
        #     self.threshold = 0.9 if not args.threshold else args.threshold
        #     self.new_amp = 0.3 if not args.new_amp else args.new_amp
        
        elif self.neural_encoding == 'BSA':
            self.fil_args1 = 3 if not args.fil_args1 else args.fil_args1
            self.fil_args2 = 3 if not args.fil_args2 else args.fil_args2
            self.threshold = 0.96 if not args.threshold else args.threshold
            self.new_amp = 0.01 if not args.new_amp else args.new_amp
            
        elif self.neural_encoding == 'BURST':
            self.threshold = 0.125 if not args.threshold else args.threshold
            self.step_size = int(5)
            self.beta = 2.0 if not args.beta else args.beta 
 
        elif self.neural_encoding == 'TTFS':
            # self.tau =  int(self.num_steps/5) if not args.tau else args.tau
            self.tau =  20 if not args.tau else args.tau
            self.step_size = int(3)
            
            # self.data_root = 'MIT_BIH_ECG/bin_ttfs_3_mitbih_train.csv'
            # self.test_data_root = 'MIT_BIH_ECG/bin_ttfs_3_mitbih_test.csv'

    
    def print_info(self):
        
        args = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        print(f"{' Argument List ' : =^100s}")
        
        for k, v in args.items():
            arg = f"{k:-<30s}{str(v):->70s}"
            print(arg)
        
        print('\n')
        
        # print("_"*55)
        # print()
        # print("\t\t\t>>  binary classification  << \n")
        # if self.neural_encoding: print(f" * spike encoding algorithm  [{self.neural_encoding}]")
        # print(f" * type of neuron  [{self.neuron}]")
        # print(f" * bias  [{self.bias}]")
        # print(f" * device using cuda  [{self.device.index}]")
        # print(f" * save  [{self.save}]")
        # print(f" * print_epoch  [{self.print_epoch}]")
        # print(f" * total # of epoch  [{self.epoch}]")
        # print(f" * mini batch size  [{self.batch_size}]")
        # print(f" * total simulation # of time steps  [{self.num_steps}]")
        # print(f" * initial learning rate  [{self.lr}]")
        # print(f" * scheduler step size  [{self.step_size}]")
        # print(f" * # train samples  [{self.num_samples}]")
        # print(f" * # label classes  [{self.num_classes}]")
        
        # if (self.neural_encoding == "HSA") | \
        #     (self.neural_encoding =="MHSA") | \
        #     (self.neural_encoding =="BSA"): 
        #     print("\n+++++++++++++++ filter information")
        #     print(f" * filter argument 1  [{self.fil_args1}]")
        #     print(f" * filter argument 2  [{self.fil_args2}]")
        #     print(f" * filter amplitude  [{self.new_amp}]")
        # if (self.neural_encoding == "BURST") |\
        #     (self.neural_encoding == "MHSA") |\
        #     (self.neural_encoding == "BSA"): 
            
        #     print(f"\n * algorithm threshold  [{self.threshold}]")
            
        # if self.neural_encoding == "TTFS": print(f"\n * tau in TTFS  [{self.tau}]")
        # if self.neural_encoding == "BURST" : print(f"\n * beta in Burst coding [{self.beta}]")
        
        # print(f"\n+++++++++++++++ log save path is  [{self.save_log_path}]")
        # print("_"*55)
        # print()
        
    def select_criterion(self):
        return {
            'TTFS' : SF.ce_temporal_loss,
            'BSA' : SF.ce_count_loss,
            'HSA' : SF.ce_count_loss,
            'BURST' : SF.ce_count_loss
        }.get(self.neural_encoding)
        
            