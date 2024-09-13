import torch
from torch import Tensor

import torch.nn as nn
import snntorch as snn
import snntorch.spikegen as gen

import numpy as np

from plot_spike import *
from copy import deepcopy


class HSA(nn.Module):
    def __init__(self, filter=False, new_amp=0.2, device='cuda'):
        super().__init__()

        self.filter = filter
        self.new_amp = new_amp
        self.device = device
           
        if type(self.filter) != torch.Tensor:
            self.filter = torch.FloatTensor(self.filter).to(self.device) 
        
        self.filter = self.filter * self.new_amp
            
        self.count = 0
            
        
    def hsa_encode(self, sig:Tensor):
        
        data = deepcopy(sig)
            
        size_input = data.shape[1] # 140
        size_filter = self.filter.shape[0]  # 11
        
        self.output = torch.zeros(data.shape).to(self.device)
                        
        for i in range(size_input):
            if (i+size_filter - 1) <= (size_input - 1):
                    mask = torch.all(data[:, i:i+size_filter] >= self.filter, dim=1) # batch_size
                    data[mask, i:i+size_filter] -= self.filter
                        
                    self.output[mask, i] = 1
                        
            else: break
            
        # if self.count == 0:
        #     plot_batch_spike(algorithm='hsa', output_spike=self.output, save_root='hsa_plot')
            
        self.count += 1
        self.num_spike = torch.sum(self.output, dim=1)
        return self.output.detach()
    
    def forward(self, input:Tensor) -> Tensor:
        return self.hsa_encode(input)
    
    def count_spike(self):
        # self.output.shape = [batch_size, len_time_seies]
        return self.num_spike.clone().detach()
        
    
# 안고침
class MHSA(nn.Module):
    def __init__(self, filter=False, new_amp=0.2, threshold=0.5, device='cuda') -> None:
        super().__init__()

        self.filter = filter
        self.new_amp = new_amp
        self.threshold = threshold
        self.device = device
        
        if type(self.filter) != torch.Tensor:
            self.filter = torch.tensor(self.filter).to(self.device)
    
    def mhsa_encode(self, data):

        size_input = data.shape[1]
        size_filter = self.filter.shape[0]
        self.output = torch.zeros(data.shape).to(self.device)
        
        self.filter = self.filter * self.new_amp
        
        for batch in range(data.shape[0]):
            for i in range(size_input):
                error=0
                for j in range(size_filter):
                    if (i+size_filter-1) < size_input:
                        if data[batch][i+j] < self.filter[j]:
                            error = error + self.filter[j] - data[batch][i+j]
                    if error <= self.threshold:
                        self.output[batch][i] = 1
                        if (i+size_filter) <= size_input:
                            data[batch][i:i+size_filter] -= self.filter[:]
      
        # plot(orign_sig=orign_sig, time=i, filter=filter, data=data, output_spike=self.output, algorithm='modified HSA')
            
        self.num_spike = torch.sum(self.output, dim=1)
        
        return self.output.clone().detach()
    
    def forward(self, input:Tensor) -> Tensor:
        return self.mhsa_encode(input)
    
    def count_spike(self):
        # self.output.shape = [batch_size, len_time_seies]
        return self.num_spike.clone().detach()
    


class BSA(nn.Module):
    def __init__(self, filter, new_amp=1, threshold=0.9952, device='cuda') -> None:
        super().__init__()
        
        self.filter = filter
        self.new_amp = new_amp
        self.threshold = threshold
        self.device = device
        
        if type(self.filter) != torch.Tensor:
            self.filter = torch.tensor(self.filter).to(self.device)
            
        self.filter = self.filter * self.new_amp
            
    def bsa_encode(self, sig):
        
        data = deepcopy(sig)
        
        size_input = data.shape[1]
            
        size_filter = self.filter.shape[0]
        self.output = torch.zeros(data.shape).to(self.device)
            
        for i in range(size_input-size_filter+1):
            
            if i > size_input - size_filter - 1:
                break
            else:
                error1 = torch.zeros(data.shape[0], device=self.device)
                error2 = torch.zeros(data.shape[0], device=self.device)
                
                # for j in range(size_filter):
                error1 += abs((data[:, i:i+size_filter] - self.filter).sum(1))
                error2 += abs(data[:, i:i+size_filter].sum(1))
                    
                # if error1 <= (error2 - threshold):
                # if error1 <= (error2 * self.threshold):
                mask = (error1 <= (error2 * self.threshold))
                if torch.any(mask):
                    self.output[mask, i] = 1
                    data[mask, i+1:i+size_filter+1] -= self.filter
                        
        self.num_spike = torch.sum(self.output, dim=1)
        
        return self.output.clone().detach()
    
    def forward(self, input:Tensor) -> Tensor:
        return self.bsa_encode(input)
    
    def count_spike(self):
        # self.output.shape = [batch_size, len_time_seies]
        return self.num_spike.clone().detach()


class TTFS(nn.Module):
    def __init__(self, data_num_steps, t_d=0, init_th=1, tau=5, device='cuda') -> None:
        super().__init__()
        
        self.t_d = t_d
        self.init_th = init_th
        self.tau = tau
        self.th = 0
        self.device = device
        
        self.mem = torch.zeros(data_num_steps).to(self.device) # membrane potential initialization
        self.fire_time = list()
        
    def encoding_kernel(self, t):
        """
        The encoding kernel for TTFS coding
        ---
        t : current time step
        """

        time = t - self.t_d
        kernel = np.exp(-time / self.tau)
        self.th = self.init_th * kernel
        
    def ttfs_encode(self, data, t):
        r""" 
        only for input data encoding
        ---
        :return: shape: [data time step length, total simulation time step]
            if data.shape is `[140,]` and total_timestep is `[100,]`, 
                then return spike train shape is going to `[140, 100]`
        - this neuron takes reset by zero
        """
        
        if t==0: 
            self.mem = data.clone().detach().to(self.device)
            self.th = 0
            
            
        # self.output = torch.zeros(self.mem.shape).to(self.device)
        
        self.encoding_kernel(t)
    
        if self.th >= 1e-5:
            fire = (self.mem >= self.th)
        else:
            fire = torch.zeros(self.mem.shape, device=self.device).bool()

        # self.output[fire] = 1
        self.output = torch.where(fire, 
                                  torch.ones(self.mem.shape, device=self.device),
                                  torch.zeros(self.mem.shape, device=self.device))
        
        # reset (it must be modified)
        # self.mem[fire] = 0
        self.mem = torch.where(fire, torch.zeros(self.mem.shape, device=self.device), self.mem)
        
        self.fire_time.append(torch.where(self.output == 1))
        
        self.num_spike = torch.sum(self.output, dim=1)
        
        return self.output.clone().detach()

    def get_fire_time(self):
        return self.fire_time
    
    def forward(self, input:Tensor, t:int) -> Tensor:
        return self.ttfs_encode(input, t)

    def count_spike(self):
        # self.output.shape = [batch_size, len_time_seies]
        return self.num_spike.clone().detach()


class BURST(nn.Module):
    def __init__(self, data_num_steps, beta=2.0, init_th=0.125, device='cuda') -> None:
        super().__init__()
        """Burst coding at the time step

        Args:
            data (torch.Tensor): the data transfomed into range `[0, 1]`
            mem (torch.Tensor): data transfomed into range `[0, 1]`
            t (int): time step
            beta (float, optional): . Defaults to 2.0.
            th (float, optional): _description_. Defaults to 0.125.
        """
        
        self.beta = beta
        self.init_th = init_th
        self.device = device
        
        # self.th = torch.tensor([]).to(self.device)
        # self.mem = torch.zeros(data_num_steps).to(self.device) # membrane potential initialization
        
    def burst_encode(self, data, t):
        if t==0:
            self.mem = data.clone().detach().to(self.device)
            self.th = torch.ones(self.mem.shape, device=self.device) * self.init_th
            
        self.output = torch.zeros(self.mem.shape).to(self.device)
        
        fire = (self.mem >= self.th)
        self.output = torch.where(fire, torch.ones(self.output.shape, device=self.device), self.output)
        out = torch.where(fire, self.th, torch.zeros(self.mem.shape, device=self.device))
        self.mem -= out
        
        # th update
        
        self.th = torch.where(fire, self.th * self.beta, torch.ones(self.th.shape, device=self.device)*self.init_th)

        # if you want to repeat input
        
        if self.output.max() == 0:
            self.mem = data.clone().detach().to(self.device) ###########################################################################

        
        # plot2(orign_sig=orign_sig, data=data, output_spike=self.output, algorithm='burst')
        
        # self.num_spike = torch.sum(self.output, dim=1)
        self.num_spike = torch.sum(self.output)/self.mem.shape[0]
        
        return self.output.clone().detach()
    
    def forward(self, input:Tensor, t:int) -> Tensor:
        return self.burst_encode(input, t)
    
    @staticmethod
    def count_spike(input_spike):
        # self.output.shape = [batch_size, len_time_seies]
        # return self.num_spike.clone().detach()
        return torch.sum(input_spike, dim=1)
        
        
