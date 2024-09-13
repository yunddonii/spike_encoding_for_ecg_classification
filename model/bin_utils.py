import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

import pandas as pd
from scipy.io import arff

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from tqdm import tqdm

from sklearn.utils import resample
import os

from scipy import signal

# snntorch
# import snntorch as snn
# from snntorch import surrogate
# from snntorch import functional as SF
# from snntorch import spikeplot as splt
# from snntorch import utils

def arff2csv(train_root, test_root, save_dir, split=False, index=False):
    """
        train_root : arff training set root,
        test_root : arff test set root,
        split=True : save file splited into train and test,
        index : index
    """
    arff_trainset, _ = arff.loadarff(train_root)
    arff_testset, _ = arff.loadarff(test_root)
    
    df_train = pd.DataFrame(arff_trainset)
    df_test = pd.DataFrame(arff_testset)
    
    num_targets = int(1 + len(df_train['target'].unique()))
    
    df_train['target'] = df_train['target'].str.decode('utf-8')
    df_test['target'] = df_test['target'].str.decode('utf-8')
    
    for i in range(num_targets):
        df_train.loc[df_train['target'] == str(i), 'target'] = i-1
        df_test.loc[df_test['target'] == str(i), 'target'] = i-1 

    if split:
        df_train.to_csv(save_dir + '/ECG5000_train.csv', index=index)
        df_test.to_csv(save_dir + '/ECG5000_test.csv', index=index)
    
    df_tot = pd.concat([df_train, df_test], axis=0) # 5000 X 140
    
    # label: str -> integer
    # df_tot['target'] = df_tot['target'].str.decode('utf-8')
    
    df_tot.to_csv(save_dir + '/ECG5000_tot.csv', index=index)
    
    return df_train, df_test, df_tot


def oversampling(data, sampling_size_class_num=4):
    
    # num_classes = len(data.target.unique())
    num_classes = len(data.iloc[:, -1].unique())
    
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    
    X_sampled = []
    y_sampled = []

    for i in range(0, num_classes):
        X_, y_ = resample(X[y==i], y[y==i], replace=True, n_samples=X[y==sampling_size_class_num].shape[0])
        X_sampled.append(X_)
        y_sampled.append(y_)
        
    df_X = np.vstack(X_sampled)
    df_y = np.hstack(y_sampled)

    # df_X = np.concatenate([X[y==0], X_sampled], axis=0)
    # df_y = {'target': np.concatenate([y[y==0], y_sampled])} 

    df_X = pd.DataFrame(df_X)
    df_y = pd.DataFrame(df_y)
    
    df = pd.concat([df_X, df_y], axis=1, ignore_index=False)
    
    return df



def undersampling(data, sampling_size):
    
    
    num_classes = len(data.iloc[:, -1].unique())
    
    if sampling_size:
        min_value = sampling_size
    else:
        min_value = data.iloc[:, -1].value_counts().min()
    
    X_sampled = []
    y_sampled = []

    start_idx = 0
    for i in range(0, num_classes):
        
        max_idx = len(data[data.iloc[:, -1] == i])
        mask = np.random.permutation(range(start_idx, start_idx + max_idx))
        mask = mask[:min_value]
        sampled_data = data.iloc[mask, :]
        
        X_sampled.append(sampled_data.iloc[:, :-1].values)
        y_sampled.append(sampled_data.iloc[:, -1].values)
        
        start_idx = start_idx + max_idx
        
    df_X = np.vstack(X_sampled)
    df_y = np.hstack(y_sampled)

    # df_X = np.concatenate([X[y==0], X_sampled], axis=0)
    # df_y = {'target': np.concatenate([y[y==0], y_sampled])} 

    df_X = pd.DataFrame(df_X)
    df_y = pd.DataFrame(df_y)
    
    df = pd.concat([df_X, df_y], axis=1, ignore_index=False)
    
    return df



class CustomDataset(Dataset):
    def __init__(self, root=None, class_imbalance=True, transform=None) -> None:
        """
            root: csv file root,
            class_imbalance: True => take resample to make samples of lack class same with largest class
        """
        super().__init__()
        
        if root is None:
            root = '../ecg_data/ECG4500_train.csv'
        
        self.data = pd.read_csv(root)
        self.transform = transform
        self.undersampling_size = 15000
        
        if class_imbalance:
            # self.data = oversampling(self.data)
            self.data = undersampling(self.data, self.undersampling_size)
        
        num_steps = int(len(self.data.columns) - 1)
        
        self.x = self.data.iloc[:, :num_steps].to_numpy(dtype=np.float64) 
        self.y = self.data.iloc[:, num_steps].to_numpy(dtype=np.compat.long) 
    
    def __getitem__(self, idx):
            
        img = torch.tensor(self.x[idx][:], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
            
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        
        return self.x.shape[0] 
    
    
class TestCustomDataset(Dataset):
    def __init__(self, root=None, class_imbalance=True, transform=None) -> None:
        """
            root: csv file root,
            class_imbalance: True => take resample to make samples of lack class same with largest class
        """
        super().__init__()
        
        if root is None:
            root = './MIT_BIH_ECG/bin_3_mitbih_test.csv'
        
        self.data = pd.read_csv(root)
        self.transform = transform
        self.undersampling_size = 3000
        
        if class_imbalance:
            # self.data = oversampling(self.data)
            self.data = undersampling(self.data, self.undersampling_size)
        
        num_steps = int(len(self.data.columns) - 1)
        
        self.x = self.data.iloc[:, :num_steps].to_numpy(dtype=np.float64) 
        self.y = self.data.iloc[:, num_steps].to_numpy(dtype=np.compat.long) 
    
    def __getitem__(self, idx):
            
        img = torch.tensor(self.x[idx][:], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
            
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        
        return self.x.shape[0] 
    

def get_transform(data):
    """normalization
    ---
    Args:
        data (torch.Tensor)

    Returns:
        torch.Tensor : normalized data
    """
    # min_data = torch.min(data)
    # max_data = torch.max(data)
    # norm_data = (data - min_data)/(max_data - min_data) 
    norm_data = data - data.mean()
    norm_data *= 1/data.std()
    assert (min(norm_data) == 0) & (max(norm_data) == 1),  "Minimum and Maximums is not correct, check please!"
    
    return norm_data
    
    
def class_split_acc(spk_rec, spike_count, label, num_classes=2):
    
    if spike_count.shape[0] == 100:
        spike_count = torch.sum(spike_count, dim=0)
    
    count = {i: 0 for i in range(num_classes)}
    class_acc = {i: 0 for i in range(num_classes)}
    class_spike_count = {i: 0 for i in range(num_classes)}
    
    _, idx = spk_rec.sum(dim=0).max(1)
    accuracy = (label == idx).detach().cpu().numpy()
    spike_count = spike_count.detach().cpu().numpy()
    
    for i in range(len(label)):
        count[label[i].item()] += 1
        class_acc[label[i].item()] += accuracy[i]
        class_spike_count[label[i].item()] += spike_count[i]

    return class_acc, count, class_spike_count

def class_split_acc_ttfs(spk_rec, spike_count, label, num_classes=2):
    
    device = spk_rec.device
    
    count = {i: 0 for i in range(num_classes)}
    class_acc = {i: 0 for i in range(num_classes)}
    class_spike_count = {i: 0 for i in range(num_classes)}
    
    spike_count = spike_count.detach().cpu().numpy()
    
    fire_time = (spk_rec.transpose(0, -1) * (torch.arange(0, spk_rec.shape[0]).detach().to(device)+1)).transpose(0, -1)
    
    first_fire_time = torch.zeros_like(fire_time[0])
    
    for step in range(fire_time.shape[0]):
        first_fire_time += (
            fire_time[step] * ~first_fire_time.bool()
        )
        
    first_fire_time += ~ first_fire_time.bool() * (fire_time.shape[0])
    first_fire_time -= 1
    
    pred_idx = first_fire_time.min(1)[1]
    accuracy = (label == pred_idx).detach().cpu().numpy()
    
    for i in range(len(label)):
        count[label[i].item()] += 1
        class_acc[label[i].item()] += accuracy[i]
        class_spike_count[label[i].item()] += spike_count[i]

    return class_acc, count, class_spike_count


def data_loader(data_set, batch_size:int, num_samples:int, num_test_samples:int):
    
    indice = list(range(len(data_set)))
    np.random.shuffle(indice)
    
    train_sampler = torch.utils.data.SubsetRandomSampler(indice[:num_samples])
    val_sampler = torch.utils.data.SubsetRandomSampler(indice[num_samples:num_samples+num_test_samples])
    # test_sampler = torch.utils.data.SubsetRandomSampler(indice[num_test_samples:num_test])

    train_loader = DataLoader(data_set, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=8)
    val_loader = DataLoader(data_set, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=8)

    return {
        'train' : train_loader,
        'val' : val_loader,
        # 'test' : test_loader,
    }
    


def print_class_score(epoch_acc, t_epoch_acc, epoch_spike_count, t_epoch_spike_count, num_classes=2):
    # training score
    print(" * [train_acc]\t", end="")
    for i in range(num_classes): print(f"{i} {epoch_acc[i][-1]*100:.2f}% | ", end="")
    
    total_acc = 0
    for i in range(num_classes): total_acc += epoch_acc[i][-1]
    print(f" >> total {(total_acc*100)/num_classes:.2f}%")
    
    print(" * [val_acc]\t", end="")
    # validation score
    for i in range(num_classes):print(f"{i} {t_epoch_acc[i][-1]*100:.2f}% | ", end="")
    
    t_total_acc = 0
    for i in range(num_classes): t_total_acc += t_epoch_acc[i][-1]
    print(f" >> total {(t_total_acc*100)/num_classes:.2f}%")
    
    print()
    
    # spike average count
    print(" * [train_spike_count]\t", end="")
    for i in range(num_classes): print(f"{i} {epoch_spike_count[i][-1]} | ", end="")
    
    total_spike_count = 0
    for i in range(num_classes): total_spike_count += epoch_spike_count[i][-1]
    print(f" >> total {(total_spike_count/num_classes):.2f}")
    
    print(" * [val_spike_count]\t", end="")
    # validation score
    for i in range(num_classes):print(f"{i} {t_epoch_spike_count[i][-1]} | ", end="")
    
    t_total_spike_count = 0
    for i in range(num_classes): t_total_spike_count += t_epoch_spike_count[i][-1]
    print(f" >> total {(t_total_spike_count/num_classes)}")
    
    print()
    

def save_info(model, args, epoch_infos):
    
    save_dict = {
                    'model_state' : model,
                    'epoch_loss' : epoch_infos['epoch_loss'],
                    't_epoch_loss' : epoch_infos['t_epoch_loss'],
                    'epoch_acc' : epoch_infos['epoch_acc'], 
                    't_epoch_acc' : epoch_infos['t_epoch_acc'],
                    'epoch_spike_count' : epoch_infos['epoch_spike_count'],
                    't_epoch_spike_count' : epoch_infos['t_epoch_spike_count']
                    }
    
    save_root = os.path.join(args.save_log_path, "model_info_save_dict", f"{args.model_name}_{args.date_dir}.pth")
    torch.save(save_dict, save_root)
    print('All saves completed successfully and model training completed \n\b\bsave root:', save_root)

    
def epoch_print_save(args, epoch, epoch_infos, optimizer_param, model_state_dict=None):
    
    if (epoch+1) % args.print_epoch == 0:
        
        print(f"\nEPOCH {epoch+1} | train_loss {epoch_infos['epoch_loss'][-1]:.4f} | test_loss {epoch_infos['t_epoch_loss'][-1]:.4f}\n")
        print_class_score(epoch_infos['epoch_acc'], epoch_infos['t_epoch_acc'], epoch_infos['epoch_spike_count'], epoch_infos['t_epoch_spike_count'])
        
        if args.save:
            save_path = os.path.join(args.save_log_path, "epoch_save", f"{args.model_name}_{epoch+1:03d}_{args.date}.pth")
            torch.save(model_state_dict, save_path)
           
    if (epoch+1) % args.step_size == 0:
        print(f"\n++ sheduler lerning rate : {optimizer_param}\n")
        

def epoch_info(num_classes:int, train_infos:dict, val_infos:dict):
    
    epoch_infos = {
        'epoch_loss' : list(),
        't_epoch_loss' : list(),
        'epoch_acc' : {i: list() for i in range(num_classes)},
        't_epoch_acc' : {i: list() for i in range(num_classes)},
        'epoch_spike_count' : {i: list() for i in range(num_classes)},
        't_epoch_spike_count' : {i: list() for i in range(num_classes)},
    }
    
    epoch_infos['epoch_loss'].append(np.mean(train_infos['train_loss']))
    epoch_infos['t_epoch_loss'].append(np.mean(val_infos['test_loss']))
    
    for i in range(num_classes):
        # accuracy
        if val_infos['t_class_acc'][i]:
            epoch_infos['t_epoch_acc'][i].append(val_infos['t_class_acc'][i]/val_infos['t_class_count'][i])
        else: 
            epoch_infos['t_epoch_acc'][i].append(0)
        if train_infos['class_acc'][i]: 
            epoch_infos['epoch_acc'][i].append(train_infos['class_acc'][i]/train_infos['class_count'][i])
        else: epoch_infos['epoch_acc'][i].append(0)
        
        # spike count
        if val_infos['t_class_spike_count'][i]:
            epoch_infos['t_epoch_spike_count'][i].append(val_infos['t_class_spike_count'][i]/val_infos['t_class_count'][i])
        # else: 
            # epoch_infos['t_epoch_spike_count'][i].append(0)
        if train_infos['class_spike_count'][i]: 
            epoch_infos['epoch_spike_count'][i].append(train_infos['class_spike_count'][i]/train_infos['class_count'][i])
        # else: epoch_infos['epoch_spike_count'][i].append(0)
        
    return epoch_infos


def plot_data(data, predicted_class, target_class):
    
    if isinstance(predicted_class, torch.Tensor):
        predicted_class = predicted_class.item()

    if isinstance(target_class, torch.Tensor):
        target_class = target_class.item()
    
    class_dict = {0: 'normal',
                  1: 'PVC',
                  2: 'R-on-T PVC',
                  3: 'SP or EB',
                  4: 'Unclassified Beat'}
    
    plt.title(f"predicted class : {class_dict[predicted_class]} | target class : {class_dict[target_class]}")
    plt.plot(data)
    plt.xlabel("time")
    plt.show()
    

def show_data_frame(data, target):
    data = pd.DataFrame(data.cpu().numpy())
    target = {'target' : target.cpu().numpy()}
    label = pd.DataFrame(target)
    
    sample = pd.concat([data, label], axis=1)
    
    print(sample)

    
def get_one_sample(root, random=True, **kargs):
    test_set = CustomDataset(root, True, None)
    if random:
        idx = np.random.randint(low=0, high=len(test_set), size=(1,))
    else:
        idx = kargs['idx']
        
    return idx, (test_set[idx][0], test_set[idx][1])


# def get_filter(filter, args1, args2=0):
#     if filter == 'gaussian':
#         args1 = 51 if not args1 else args1
#         args2 = 7 if not args2 else args2
#         new_filter = signal.windows.gaussian(args1, args2)
#     elif filter == 'triang':
#         args1 = 12 if not args1 else args1
#         new_filter = signal.windows.triang(args1)
    
#     return new_filter

