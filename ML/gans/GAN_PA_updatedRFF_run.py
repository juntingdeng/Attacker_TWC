import os
import sys
import time
import numpy as np
import random
# import cupy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.io
from ML.gans.VAE import myVAE
from classifier_cnn import Classifier, Classifier2D, addNoise2Batch
from GAN_PA_updatedRFF import *

if __name__ == "__main__":
    seed=128
    SNR=25
    LR=5e-4
    n_epoch=2000
    batch_size=128
    loss_weights=[1, 0.1, 1]
    pretrained=False
    save_ckp=True
    save_time=True

    if pretrained:
        load_path="./checkpoint2/gan/gan_vae_PA_transfer_fixed/snr"+str(SNR)+"/retrain_seed"+str(seed)+"/"
        save_path = "./checkpoint2/gan/gan_vae_PA_transfer_fixed/snr"+str(SNR)+"/retrain_seed"+str(seed)+"/"
    else:
        load_path=None
        save_path = "./checkpoint2/gan/gan_vae_PA_fixed/snr"+str(SNR)+"/retrain_seed"+str(seed)+"/"

    main(seed=seed, 
         sys_argv=sys.argv, 
         SNR=SNR, 
         LR=LR, 
         n_epoch=n_epoch, 
         batch_size=batch_size,
         loss_weights=loss_weights,
         save_ckp=save_ckp,
         save_time=save_time,
         save_path=save_path,
         pretrained=pretrained,
         load_path=load_path,
         )