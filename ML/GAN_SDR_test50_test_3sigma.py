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
import csv
from VAE import myMemPolyVAE, fastPolyPrefix, fastPolyPrefixOut
from classifier_cnn import Classifier, Classifier2D, addNoise2Batch
from GAN_SDR import ClassifierSDR, Discriminator
# import matlab.engine
import __main__

def load_allSDR_norm(matFiles, distance=0.25):
    datasets_allSDR=None
    targets_allSDR=None
    samples_perSDR=300
    for matfilei in range(len(matFiles)):
        matfile = matFiles[matfilei]
        datasets_load=scipy.io.loadmat("./data/"+str(distance)+"meters/"+matfile)['packet_equalized_allrecord']
        datasets_load = datasets_load[:samples_perSDR]
        targets_load = matfilei*np.ones(datasets_load.shape[0])
        # print("SDR: {}, range: {}~{}".format(matfilei, datasets_load.min(), datasets_load.max()))
        datasets_allSDR = np.concatenate([datasets_allSDR, datasets_load], axis=0) if matfilei!=0 else datasets_load
        targets_allSDR = np.concatenate([targets_allSDR, targets_load], axis=0) if matfilei!=0 else targets_load
    datasets_allSDR_magn = np.max((np.abs(datasets_allSDR.min()), np.abs(datasets_allSDR.max())))
    datasets_allSDR = datasets_allSDR/datasets_allSDR_magn
    # print("datasets load all sdr shape: ", datasets_allSDR.shape)
    # print("targets load all sdr shape: ", targets_allSDR.shape)
    # print("datasets load all sdr and norm range: {}~{}".format(datasets_allSDR.min(), datasets_allSDR.max()))
    return datasets_allSDR, targets_allSDR

matFiles25=["pluto1_0.25meters_run2.mat",
    "pluto2_0.25meters_run2.mat",
    "pluto3_0.25meters_run2.mat",
    "AD9082_0.25meters_run2.mat",
    "AD9082_CC2595_amp1_0.25meters_run2.mat",
    "TI_TRF3705_0.25meters.mat",
    "TI_TRF3705_CC2595_amp1_0.25meters.mat",
    "TI_TRF3722_CC2595_amp1_0.25meters.mat",
    "TI_TRF3722_0.25meters.mat",
    "Pluto1RX_9082TX_0.25meters_NoAGC10Gain_3pkthreshold_348samples.mat",
    "Pluto1RX_9082_CC2595_TX_0.25meters_3pkthreshold_582samples.mat"]

matFiles50=["pluto1_0.5meters_run2.mat",
    "pluto2_0.5meters_run2.mat",
    "pluto3_0.5meters_run2.mat",
    "AD9082_0.5meters_run2.mat",
    "AD9082_CC2595_amp1_0.5meters_run2.mat",
    "TI_TRF3705_0.5meters.mat",
    "TI_TRF3705_CC2595_amp1_0.5meters.mat",
    "TI_TRF3722_CC2595_amp1_0.5meters.mat",
    "TI_TRF3722_0.5meters.mat",
    "Pluto1RX_9082TX_0.5meters_NoAGC10Gain_3pkthreshold_403samples.mat",
    "Pluto1RX_9082_CC2595_TX_0.5meters_3pkthreshold_319samples.mat"]

if __name__ == "__main__":
    print("+++++++++++++++++++Start main()+++++++++++++++++++") 

    # seed = sys.argv[1:]
    seed = 128 #[int(x) for x in seed][0]
    SNR = 25
    sigma = 3
    samples_perSDR=300
    start = 0
    nSDR = 11
    batch_size = 32
    print("seed: {}, sigma:{}".format(seed, sigma))
    print("start:{}, nSDR:{}, SNR:{}dB".format(start, nSDR, SNR))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    setattr(__main__, "ClassifierSDR", ClassifierSDR)
    classifier = ClassifierSDR(nSDR=nSDR)
    clf_load=torch.load("./checkpoint2/classifier/11SDR_nosnr_normall_epoch200.cnn", map_location=device)
    classifier=clf_load["classifier"]
    train_acc=clf_load["train_acc"]
    val_acc=clf_load["val_acc"]
    test_acc=clf_load["test_acc"]
    print("Load clf, train_acc:{:.2f}%, val_acc:{:.2f}%, test_acc:{:.2f}%".format(train_acc.item()*100, val_acc.item()*100, test_acc.item()*100))

    samples_perSDR=300
    duplicate=4
    datasets_allSDR, targets_allSDR = load_allSDR_norm(matFiles25, distance=0.25)
    avg_test_acc=0
    avg_test_acc_fake=0
    test_acc_sdrs=np.zeros(nSDR)
    test_acc_fake_sdrs=np.zeros(nSDR)

    h_sigma=[]
    h_sigma_load=np.load("./checkpoint2/gan/gan_poly_ray_SDR_normAll_nosnrclf_bn_dup/h_sigma"+str(sigma)+"_snr30_seed"+str(seed)+".npy")
    h_sigma_load = torch.tensor(h_sigma_load).to(device)
    hi=0
    for SDRlabel in range(start, start+nSDR):
        # print("\n-----------------SNR:{}, SDRlabel:{}-----------------".format(SNR, SDRlabel))
        timer_curve={}
        max_train_acc=0
        max_val_acc=0

        datasets = datasets_allSDR[SDRlabel*samples_perSDR: (SDRlabel+1)*samples_perSDR]
        targets = targets_allSDR[SDRlabel*samples_perSDR: (SDRlabel+1)*samples_perSDR]
        datasets_dup = np.zeros((datasets.shape[0]*duplicate, datasets.shape[1], datasets.shape[2]))
        targets_dup = np.zeros((targets.shape[0]*duplicate,))
        for d in range(1, duplicate):
            datasets_dup[d*datasets.shape[0]: (d+1)*datasets.shape[0]] = addNoise2Batch(datasets, snr=SNR, norm=False, device=device)
            targets_dup[d*datasets.shape[0]: (d+1)*datasets.shape[0]] = targets

        datasets=torch.tensor(datasets_dup, dtype=float).to(device)
        targets=torch.tensor(targets_dup, dtype=torch.int64).to(device)
        # print("datasets: ", datasets.shape)
        # print("targets: ", targets.shape)
        # print("datasets range: ", datasets.max(), datasets.min())

        rffDataset=Data.TensorDataset(datasets, targets)
        datasetLoader = Data.DataLoader(rffDataset, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.manual_seed(seed))
        # print("len datasetLoader", len(datasetLoader))

        hidden_dim=256
        latent_dim=2
        poly_dim=15
        
        path = "./checkpoint2/gan/gan_poly_ray_SDR_normAll_nosnrclf_bn_dup/snr"+str(SNR)+"/seed"+str(seed)+"/ckp/"
        ckps = os.listdir(path)
        for ckp in ckps:
            if len(ckp.split("_"))<2:
                continue
            sdr_load = int(ckp.split("_")[0][3:])
            if sdr_load == SDRlabel:
                break
        SNR_load = int(ckp.split("_")[1][3:])
        assert SNR_load == SNR
        train_acc = float(ckp.split("_")[3][8:])
        val_acc = float(ckp.split("_")[4][6:])
        # print("Load sdr:{}, snr:{}, tran_acc:{}, val_acc:{}".format(SDRlabel, SNR, train_acc, val_acc))
        ckp_load=torch.load(path+ckp, map_location=device)
        generator = ckp_load["generator"]
        discriminator = ckp_load["discriminator"]
        classifier = ckp_load["classifier"]

        test_acc=[]
        test_acc_fake=[]
        for batch_idx, (data, labels) in enumerate(datasetLoader):
            # print("batch_idx: ", batch_idx)
            data, labels = data.to(device, dtype=torch.float), labels.to(device,  dtype=torch.int64)

            fake_out_g, mu, logvar = generator.forward(data, device=device)

            x_encoder = generator.encoder(data)

            poly_coef = generator.poly_layer(x_encoder)
            mu = generator.mean_layer(x_encoder)
            logvar = generator.logvar_layer(x_encoder)
            
            poly_coef=poly_coef.detach().cpu().numpy().reshape(data.shape[0], 3, 5)
            ycoef2 = fastPolyPrefixOut(generator.tx.detach().cpu().numpy(), generator.txTerms, poly_coef, memLen=3)
            ycoef = torch.tensor(ycoef2).to(device)

            # h_real=torch.randn((batch_size, 800), generator=torch.manual_seed(seed)).to(device)
            # h_imag=torch.randn((batch_size, 800), generator=torch.manual_seed(seed+1)).to(device)
            # h_comp=[h_real, h_imag]
            # for epsilon in h_comp:
            #     torch.manual_seed(seed)
            #     epsilon_num = epsilon.detach().cpu().numpy()
            #     for bi in range(epsilon_num.shape[0]):
            #         for i in range(epsilon_num[bi].shape[0]):
            #             # print("i:{}".format(i))
            #             while not(((-sigma<epsilon_num[bi][i]) and (epsilon_num[bi][i]<=-(sigma-1))) or (((sigma-1)<epsilon_num[bi][i]) and (epsilon_num[bi][i]<=sigma))):
            #                 epsilon[bi][i] = torch.randn(1).to(device)
            #                 epsilon_num[bi][i] = epsilon[bi][i].detach().cpu().numpy()
            
            h_real = h_sigma_load[hi][0]
            h_imag = h_sigma_load[hi][1]
            h_comp=[h_real, h_imag]
            hi+=1
            h_sigma.append([h_.detach().cpu().numpy() for h_ in h_comp])
            h_real = mu[:,0].reshape(-1,1) + torch.exp(logvar[:,0]/2).reshape(-1,1)*h_comp[0]
            h_imag = mu[:,1].reshape(-1,1) + torch.exp(logvar[:,1]/2).reshape(-1,1)*h_comp[1]
            
            h = torch.stack([h_real, h_imag], axis=1)

            h1 = (h[:,0,...]**2+h[:,1,...]**2)[:, None, ...]
            h1=torch.sqrt(h1)
            y=h1*ycoef
            
            data_test_hat = generator.decoderForPoly(y)

            labels_ = classifier.forward(data)
            labels_arg = torch.argmax(labels_, dim=1)
            for i in range(batch_size):
                test_acc.append(labels_arg[i].detach().cpu().numpy()==labels[i].detach().cpu().numpy())

            labels_fake = classifier.forward(data_test_hat)
            labels_fake_arg = torch.argmax(labels_fake, dim=1)
            for i in range(batch_size):
                test_acc_fake.append(labels_fake_arg[i].detach().cpu().numpy()==labels[i].detach().cpu().numpy())
        
        # print("SDR:{}, test_acc:{}, test_acc_fake:{}".format(SDRlabel, sum(test_acc)/len(test_acc), sum(test_acc_fake)/len(test_acc_fake)))
        test_acc_sdrs[SDRlabel] = np.round(sum(test_acc)/len(test_acc), 4)
        test_acc_fake_sdrs[SDRlabel] = np.round(sum(test_acc_fake)/len(test_acc_fake), 4)
        avg_test_acc+=test_acc_sdrs[SDRlabel]
        avg_test_acc_fake += test_acc_fake_sdrs[SDRlabel]
    print("\navg_test_acc:{}, avg_test_acc_fake:{}".format(avg_test_acc/nSDR, avg_test_acc_fake/nSDR))
    print("test_acc_sdrs:{}".format(test_acc_sdrs))
    print("test_acc_fake_sdrs:{}".format(test_acc_fake_sdrs))
    print("+++++++++++++++++++End main()+++++++++++++++++++\n") 
    h_sigma = np.array(h_sigma)
    # np.save("./checkpoint2/gan/gan_poly_ray_SDR_normAll_nosnrclf_bn_dup/h_sigma"+str(sigma)+"_snr"+str(SNR)+"_seed"+str(seed)+".npy", h_sigma)