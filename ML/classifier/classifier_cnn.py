import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import deque, namedtuple
import torch.nn.functional as F
import torch.utils.data as Data
from timeit import default_timer as timer
import random
from LoadRFFDataset import LoadRFFDataset
from sklearn.decomposition import PCA
import pywt
from scipy import signal

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        #For 800 length 
        self.conv0 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5,stride=1, padding=2) # => 8*800
        # self.conv0_1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5,stride=1, padding=2) # => 8*800
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8,stride=2, padding=3) # => 16*400
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, stride=2, padding=2) # => 16*200
        # self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=6, stride=2, padding=2) # => 16*100
        # self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        self.linear1 = nn.Linear(in_features=32*200, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=12)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        #For 800 length 
        x=F.relu(self.conv0(x))
        # x=F.relu(self.conv0_1(x))
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        # x=F.relu(self.conv3(x))

        # x=self.maxpool1(x)
        flatten = nn.Flatten()
        x=flatten(x)
        x=F.relu(self.linear1(x))
        x=F.sigmoid(self.linear2(x))

        return x

class ReConstructor(nn.Module):
    def __init__(self):
        super().__init__()

        #For 800 length 
        self.conv0 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=5,stride=1, padding=2) # => 8*800
        # self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5,stride=1, padding=2) # => 16*800
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=2, kernel_size=5, stride=1, padding=2) # => 2*800

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        #For 800 length 
        x=F.relu(self.conv0(x))
        # x=F.relu(self.conv1(x))
        x=F.tanh(self.conv2(x)) # [-1, 1]

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        #For 800 length 
        self.conv0 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5,stride=1, padding=2) # => 8*800
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8,stride=2, padding=3) # => 32*400
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, stride=2, padding=2) # => 32*200
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=6, stride=2, padding=2) # => 16*100

        self.linear1 = nn.Linear(in_features=16*100, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=2)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        #For 800 length 
        x=F.relu(self.conv0(x))
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))

        flatten = nn.Flatten()
        x=flatten(x)
        x=F.relu(self.linear1(x))
        x=F.sigmoid(self.linear2(x))

        return x
    
class Classifier2D(nn.Module):
    def __init__(self, harmonics):
        super().__init__()
        self.harmonics = harmonics

        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 5), padding=(1, 2), stride=(1,1)) 
        # self.conv2 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 5), padding=(1, 2), stride=(1,1)) 
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(4, 6), padding=(1, 2), stride=(2, 2)) #=> (20, 400)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4, 6), padding=(1, 2), stride=(2, 2)) #=> (10, 200)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=self.harmonics, kernel_size=(4, 6), padding=(1, 2), stride=(2, 2)) #=> (5, 100)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.linear1 = nn.Linear(in_features=self.harmonics*5*100, out_features=64)
        # self.linear2 = nn.Linear(in_features=512, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=12)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        # x=F.relu(self.conv1(x))
        # x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.relu(self.conv5(x))
        # x=self.maxpool1(x)
        flatten = nn.Flatten()
        x=flatten(x)
        x=F.leaky_relu(self.linear1(x))
        # x=F.relu(self.linear2(x))
        x=F.sigmoid(self.linear3(x))

        return x

def addNoise2Batch(batch_data, snr, shape, norm=False, device='cuda'):
    # seed=0
    # np.random.seed(seed)
    snr = 10**(snr/10)
    batch_size = batch_data.shape[0]
    data_noised = np.zeros_like(batch_data)
    for i in range(batch_size):
        # snr = np.random.randint(10, 36, 1)     
       
        sig = batch_data[i]
        if torch.is_tensor(sig):
            sig = batch_data[i].cpu().detach().numpy().copy()
        sigPower = np.sum(sig[0]**2+sig[1]**2)/len(sig[0])
        noisePower = sigPower/snr
        noiseSigma = np.sqrt(noisePower/2) # 2 way signal
        noise = np.zeros(shape)
        noise[0] = noiseSigma*np.random.randn(shape[-1])
        noise[1] = noiseSigma*np.random.randn(shape[-1])
        # batch_data[i] = torch.tensor(sig + noise).to(device)
        # data_noised[i] = torch.tensor(sig + noise)
        data_noised[i] = sig + noise
    if norm:
        x=np.abs(np.min(data_noised))
        y=np.abs(np.max(data_noised))
        data_max = np.max((x, y))
        data_noised = data_noised/data_max
    return data_noised

def getCfgs(num=220):
    # Enable elements
    enable=1
    enables=[enable]*12
    configs=[]
    for i in range(12):
        enables[i]=0
        for j in range(i+1, 12):
            enables[j]=0
            for k in range(j+1, 12):
                enables[k]=0
                configs.append(enables[:])
                enables[k]=enable
            enables[j]=enable
        enables[i]=enable
    # print(len(configs))
    return configs[:num]

def newAccDict(configs):
    acc_dict={}
    for cfg in configs:
        cfg=tuple(cfg)
        if cfg not in acc_dict:
            acc_dict[cfg]=[0, 0] # For each cfg, [num_correct, num] record the correct num and total num => acc_cfg = num_correct/num
    return acc_dict


def getBatchCWT(data, device, corr=False, corr_cwt=None):
    data = data.detach().cpu().numpy()
    batch_size=data.shape[0]
    chs=data.shape[1]
    length=data.shape[-1]
    width=40
    
    data_cwt=np.zeros((batch_size, chs, width, length))
    for b in range(batch_size):
        for ch in range(chs):

            cwtmatr, freqs=pywt.cwt(data[b,ch,:],np.arange(1,width+1),'gaus1')
            cwtmatr_yflip1 = np.flipud(cwtmatr)
            data_cwt[b][ch] = cwtmatr_yflip1
            if corr:
                data_cwt[b][ch] = signal.correlate2d(cwtmatr_yflip1, corr_cwt[ch], mode='same')
    data_cwt = torch.tensor(data_cwt, dtype=torch.float32).to(device)
    return data_cwt