import torch
import torch.nn as nn

# snntorch
import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF

from scipy import signal
from bin_coding import *

class SNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        self.code = args.neural_encoding
        spike_grad = surrogate.atan()

        # initialize layers
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)
        
        
    def forward(self, x):
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []  # Record the output trace of spikes
        mem_rec = []  # Record the output trace of membrane potential

        for _ in range(self.num_steps):
            
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    
class HsaSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        self.filter = signal.windows.gaussian(args.fil_args1, args.fil_args2)
        
        # initialize layers
        self.encoder = HSA(filter=self.filter, new_amp=args.new_amp, device=args.device)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes, bias=args.bias)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)
        
    def forward(self, x):
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []  # Record the output trace of spikes
        # spike_count = []  # Record the output trace of membrane potential

        input_spike = self.encoder(x)
        # input_spike = input_spike.required_grad(False)

        for _ in range(self.num_steps):
            
            cur1 = self.fc1(input_spike)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0), self.encoder.count_spike()

    
class MhsaSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        self.filter = signal.windows.gaussian(args.fil_args1, args.fil_args2)
        
        # initialize layers
        self.encoder = MHSA(filter=self.filter, new_amp=args.new_amp, threshold=args.threshold, device=args.device)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes, bias=args.bias)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)

    def forward(self, x):
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()


        spk_rec = []  # Record the output trace of spikes
        # spike_count = []  # Record the output trace of membrane potential

        
        for _ in range(self.num_steps): # 100
            
            input_spike = self.encoder(x) # (140,)
            
            cur1 = self.fc1(input_spike)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0), self.encoder.count_spike()
    
    
class BsaSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        self.filter = signal.windows.gaussian(args.fil_args1, args.fil_args2)
        
        # initialize layers
        self.encoder = BSA(filter=self.filter, new_amp=args.new_amp, threshold=args.threshold, device=args.device)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes, bias=args.bias)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)

    def forward(self, x):
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()


        spk_rec = []  # Record the output trace of spikes
        # spike_count = []  # Record the output trace of membrane potential

        input_spike = self.encoder(x)
        # input_spike = input_spike.requires_grad(False)

        
        for _ in range(self.num_steps):
            
            cur1 = self.fc1(input_spike)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0), self.encoder.count_spike()


# class TtfsSnn2(nn.Module):
#     def __init__(self, args):
#         super().__init__()
        
#         self.num_steps = args.num_steps
#         spike_grad = surrogate.atan()
        
#         # initialize layers
#         self.encoder = TTFS(data_num_steps=args.data_num_steps, tau=args.tau, device=args.device)
        
#         self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
#         self.lif1 = snn.Synaptic(alpha=0.9, beta=args.beta, spike_grad=spike_grad)
        
#         self.fc2 = nn.Linear(50, args.num_classes)
#         self.lif2 = snn.Synaptic(alpha=0.9, beta=args.beta, spike_grad=spike_grad)
        
#         # self.fc1 = nn.Linear(args.data_num_steps, args.num_classes)
#         # # self.lif1 = snn.Leaky(beta=0.5, spike_grad=spike_grad)
#         # self.lif1 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        

#     def forward(self, x):
        
#         syn1, mem1 = self.lif1.init_synaptic()
#         syn2, mem2 = self.lif2.init_synaptic()
#         # mem2 = self.lif2.init_leaky()


#         spk_rec = []  # Record the output trace of spikes
#         spike_count = []  # Record the output trace of membrane potential

#         for t in range(self.num_steps):
            
#             x = self.encoder(x, t) # (140, )
            
#             cur1 = self.fc1(x)
#             spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            
#             cur2 = self.fc2(spk1)
#             spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            
#             spk_rec.append(spk2)
#             spike_count.append(self.encoder.count_spike())

#         return torch.stack(spk_rec, dim=0), torch.stack(spike_count, dim=0).sum(0)
    

class TtfsSnn1(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        # initialize layers
        self.encoder = TTFS(data_num_steps=args.data_num_steps, tau=args.tau, device=args.device)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Leaky(beta=args.beta, threshold=0.1, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes, bias=args.bias)
        self.lif2 = snn.Leaky(beta=args.beta, threshold=0.1, spike_grad=spike_grad)
        
        # self.fc1 = nn.Linear(args.data_num_steps, args.num_classes)
        # # self.lif1 = snn.Leaky(beta=0.5, spike_grad=spike_grad)
        # self.lif1 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        

    def forward(self, x):
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        # mem2 = self.lif2.init_leaky()


        spk_rec = []  # Record the output trace of spikes
        spike_count = []  # Record the output trace of membrane potential

        for t in range(self.num_steps):
            
            x = self.encoder(x, t) # (140, )
            
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk_rec.append(spk2)
            spike_count.append(self.encoder.count_spike())

        return torch.stack(spk_rec, dim=0), torch.stack(spike_count, dim=0).sum(0)
    

class BurstSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        # initialize layers
        self.encoder = BURST(data_num_steps=args.data_num_steps, init_th=args.threshold, device=args.device)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes, bias=args.bias)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)

    def forward(self, x):
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()


        spk_rec = []  # Record the output trace of spikes
        spike_count = []  # Record the output trace of membrane potential
        input_rec = []
        
        count=0
        for t in range(self.num_steps):
            
            input_spike = self.encoder(x, t)
            input_rec.append(input_spike)
        
            cur1 = self.fc1(input_spike)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk_rec.append(spk2)
            spike_count.append(input_spike.sum(1))
            # spike_count.append(self.encoder.count_spike(input_spike))

        # print(torch.stack(spike_count, dim=0).sum(0).mean())
        
        # plot_4_temporal(orign_sig=x[0].cpu().numpy(), data=x[0].cpu().numpy(), output_spike=(torch.stack(spike_count, dim=0).T)[0].cpu().numpy(), save_root='burst', algorithm='burst')
        # if count == 0:
        #     fig, ax = plt.subplots(1, 1)
        #     splt.raster(torch.stack(input_rec, dim=0)[:, 5, :], ax, s=1.5, c="black", marker="|")
        #     ax.set_xlim([0, 100])
        #     fig.savefig('burst.png')
        #     plt.close()
        # count+=1

        return torch.stack(spk_rec, dim=0), torch.stack(spike_count, dim=0).sum(0)
    
    
# # class SNN2nd(nn.Module):
#     def __init__(self, args):
#         super().__init__()
        
#         self.num_steps = args.num_steps
#         self.code = args.neural_encoding
#         spike_grad = surrogate.atan()

#         # initialize layers
#         self.conv1 = nn.Conv1d(1, 32, 3)
#         self.pool1 = nn.MaxPool1d(kernel_size=2)
#         self.lif1 = snn.Synaptic(alpha=args.alpha, beta=args.beta, spike_grad=spike_grad)
        
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=4, kernel_size=3)
#         self.pool2 = nn.MaxPool1d(kernel_size=2)
#         self.lif2 = snn.Synaptic(alpha=args.alpha, beta=args.beta, spike_grad=spike_grad)
        
#         self.fc1 = nn.Linear(4*45, 32)
#         self.lif3 = snn.Synaptic(alpha=args.alpha, beta=args.beta, spike_grad=spike_grad)
        
#         self.fc2 = nn.Linear(32, args.num_classes)
#         self.lif4 = snn.Synaptic(alpha=args.alpha, beta=args.beta, spike_grad=spike_grad)

#     def forward(self, x):
        
#         if self.code:
#             x = NEURAL_CODE[self.code](x)
#         # else: real coding (use real velue)
        
#         mem1 = self.lif1.init_synaptic()
#         mem2 = self.lif2.init_synaptic()
#         mem3 = self.lif3.init_synaptic()
#         mem4 = self.lif4.init_synaptic()

#         spk_rec = []  # Record the output trace of spikes
#         mem_rec = []  # Record the output trace of membrane potential

#         for _ in range(self.num_steps):
            
#             cur1 = self.pool1(self.conv1(x.unsqueeze(1)))
#             spk1, mem1 = self.lif1(cur1, mem1)
            
#             cur2 = self.pool2(self.conv2(spk1))
#             spk2, mem2 = self.lif2(cur2, mem2)
            
#             cur3 = self.fc1(spk2.view(-1, 4*42)) # spk1 = [16, 32, 68]
#             spk3, mem3 = self.lif3(cur3, mem3)
            
#             cur4 = self.fc2(spk3)
#             spk4, mem4 = self.lif4(cur4, mem4)

#             spk_rec.append(spk4)
#             mem_rec.append(mem4)

#         return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

################################## Synatic neuron ######################################

class SynSNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        self.code = args.neural_encoding
        spike_grad = surrogate.atan()

        # initialize layers
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes)
        self.lif2 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
        
    def forward(self, x):
        
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()

        spk_rec = []  # Record the output trace of spikes
        mem_rec = []  # Record the output trace of membrane potential

        for _ in range(self.num_steps):
            
            cur1 = self.fc1(x)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    
class HsaSynSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        self.filter = signal.windows.gaussian(args.fil_args1, args.fil_args2)
        
        # initialize layers
        self.encoder = HSA(filter=self.filter, new_amp=args.new_amp, device=args.device)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes)
        self.lif2 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
    def forward(self, x):
        
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()

        spk_rec = []  # Record the output trace of spikes
        # spike_count = []  # Record the output trace of membrane potential

        input_spike = self.encoder(x)

        for _ in range(self.num_steps):
            
            cur1 = self.fc1(input_spike)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0), self.encoder.count_spike()

    
class MhsaSynSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        self.filter = signal.windows.gaussian(args.fil_args1, args.fil_args2)
        
        # initialize layers
        self.encoder = MHSA(filter=self.filter, new_amp=args.new_amp, threshold=args.threshold, device=args.device)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes)
        self.lif2 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)

    def forward(self, x):
        
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()


        spk_rec = []  # Record the output trace of spikes
        # spike_count = []  # Record the output trace of membrane potential

        
        for _ in range(self.num_steps): # 100
            
            input_spike = self.encoder(x) # (140,)
            
            cur1 = self.fc1(input_spike)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0), self.encoder.count_spike()
    
    
class BsaSynSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        self.filter = signal.windows.gaussian(args.fil_args1, args.fil_args2)
        
        # initialize layers
        self.encoder = BSA(filter=self.filter, new_amp=args.new_amp, threshold=args.threshold, device=args.device)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes)
        self.lif2 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)

    def forward(self, x):
        
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()


        spk_rec = []  # Record the output trace of spikes
        # spike_count = []  # Record the output trace of membrane potential

        input_spike = self.encoder(x)
        
        for _ in range(self.num_steps):
                       
            cur1 = self.fc1(input_spike)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0), self.encoder.count_spike()


class TtfsSynSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        # initialize layers
        self.encoder = TTFS(data_num_steps=args.data_num_steps, tau=args.tau, device=args.device)
        
        # self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        # self.lif1 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
        # self.fc2 = nn.Linear(50, args.num_classes)
        # self.lif2 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        # self.lif1 = snn.Leaky(beta=0.5, spike_grad=spike_grad)
        self.lif1 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes)
        self.lif2 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        

    def forward(self, x):
        
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()


        spk_rec = []  # Record the output trace of spikes
        spike_count = []  # Record the output trace of membrane potential

        for t in range(self.num_steps):
            
            x = self.encoder(x, t) # (140, )
            
            cur1 = self.fc1(x)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            
            spk_rec.append(spk1)
            spike_count.append(self.encoder.count_spike())

        return torch.stack(spk_rec, dim=0), torch.stack(spike_count, dim=0).sum(0)
    

class BurstSynSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_steps = args.num_steps
        spike_grad = surrogate.atan()
        
        # initialize layers
        self.encoder = BURST(data_num_steps=args.data_num_steps, beta=args.beta, init_th=args.threshold, device=args.device)
        
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=args.bias)
        self.lif1 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(50, args.num_classes)
        self.lif2 = snn.Synaptic(alpha=0.9, beta=0.5, spike_grad=spike_grad)

    def forward(self, x):
        
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()


        spk_rec = []  # Record the output trace of spikes
        spike_count = []  # Record the output trace of membrane potential
        input_rec = []
        

        for t in range(self.num_steps):
            
            input_spike = self.encoder(x, t)
            input_rec.append(input_spike)
            
            cur1 = self.fc1(x)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            
            spk_rec.append(spk2)
            spike_count.append(input_spike.sum(1))


        return torch.stack(spk_rec, dim=0), torch.stack(spike_count, dim=0).sum(0)
   
    
MODEL = {
    'VANILLA_SNN' : SNN,
    'BURST_SNN' : BurstSnn,
    'TTFS_SNN' : TtfsSnn1,
    # 'TTFS_SNN' : TtfsSnn2,
    'HSA_SNN' : HsaSnn,
    'MHSA_SNN' : MhsaSnn,
    'BSA_SNN' : BsaSnn,
    # extra models
    'VANILLA_SYN_SNN' : SynSNN,
    'BURST_SYN_SNN' : BurstSynSnn,
    'TTFS_SYN_SNN' : TtfsSynSnn,
    'HSA_SYN_SNN' : HsaSynSnn,
    'MHSA_SYN_SNN' : MhsaSynSnn,
    'BSA_SYN_SNN' : BsaSynSnn,
}