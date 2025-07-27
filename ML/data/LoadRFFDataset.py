import numpy as np 
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch

def LoadRFFDataset(PATH, dataFile):
    """
    :param: PATH: dictionary
    :param: dataFile: .npz file
    :return: datasets: np.array with a shape (None, 2, len), len=>"len" in dataFile name
    :return: targets: mp.array with a shape (None, 12)

    To load data:
    PATH="D:\CMU\Research\RF Data\Data_for_TCAS\ChipC_AIR_processed\ChipC_AIR_processed/"
    dataFile="ChipCAIRprocessedType3BlockOS4len160comb220V1p2Quant8.npz"
    datasets, targets=LoadRFFDataset(PATH, dataFile)
    """

    file = np.load(PATH+dataFile)
    items = file.files
    datasets, targets = file[items[0]], file[items[1]]

    return datasets, targets

def RFFDatasetSNRs(PATH, dataFile, snrdBs):
    datasets, targets = LoadRFFDataset(PATH, dataFile)
    print(datasets.shape)
    print(targets.shape)

    for snrdB in snrdBs:
        datasets_snr = datasets.copy()
        snr = 10**(snrdB/10)
        for i in range(targets.shape[0]):
            cfg = tuple(targets[i])
            sig = datasets_snr[i]
            sigPower = np.sum(sig[0]**2+sig[1]**2)/len(sig[0])
            noisePower = sigPower/snr
            noiseSigma = np.sqrt(noisePower/2) # 2 way signal
            noise = np.zeros((2, 800))
            noise[0] = noiseSigma*np.random.randn(800)
            noise[1] = noiseSigma*np.random.randn(800)
            datasets_snr[i] = sig + noise
            datasets_snr[i] = datasets_snr[i].astype(np.float32)
        
        np.savez(PATH+"RFF_snr"+str(snrdB)+".npz", datasets=datasets_snr, targets=targets)

def RFFPoolSNRs(PATH, snrdBs):
    for snrdB in snrdBs:
        snr = 10**(snrdB/10)
        dataFile = "RFF_snr"+str(snrdB)+".npz"
        datasets, targets = LoadRFFDataset(PATH, dataFile)
        print(datasets.shape)
        print(targets.shape)
        
        rffPool_dict={}

        for i in range(targets.shape[0]):
            cfg = tuple(targets[i])
            sig = datasets[i]
            if cfg in rffPool_dict.keys():
                rffPool_dict[cfg].append(sig)
            else:
                rffPool_dict[cfg] =[sig]
        
        for cfg in list(rffPool_dict.keys()):
            rffPool_dict[cfg].append(sum(rffPool_dict[cfg])/len(rffPool_dict[cfg]))
        
            if snrdB==35:
                print(len(rffPool_dict[cfg]))

        np.save(PATH+"poolRFF_snr"+str(snrdB)+"_dict.npy", rffPool_dict)

def RFFPool(PATH, dataFile):
    datasets, targets = LoadRFFDataset(PATH, dataFile)
    print(datasets.shape)
    print(targets.shape)
    
    rffPool_dict={}

    for i in range(targets.shape[0]):
        cfg = tuple(targets[i])
        sig = datasets[i]
        if cfg in rffPool_dict.keys():
            rffPool_dict[cfg].append(sig)
        else:
            rffPool_dict[cfg] =[sig]
    
    for cfg in list(rffPool_dict.keys()):
        rffPool_dict[cfg].append(sum(rffPool_dict[cfg])/len(rffPool_dict[cfg]))

    np.save(PATH+"poolRFF_dict.npy", rffPool_dict)

def RFFPoolRandomNoise(PATH, dataFile):
    datasets, targets = LoadRFFDataset(PATH, dataFile)
    print(datasets.shape)
    print(targets.shape)
    snr_start=0
    snr_end=35
    for i in range(datasets.shape[0]):
        snr = np.random.randint(snr_start, snr_end+1, 1)
        # snr = 10
        snr = 10**(snr/10)
        # sig = batch_data[i].cpu().detach().numpy().copy()
        sig = datasets[i]
        sigPower = np.sum(sig[0]**2+sig[1]**2)/len(sig[0])
        noisePower = sigPower/snr
        noiseSigma = np.sqrt(noisePower/2) # 2 way signal
        noise = np.zeros((2, 800))
        noise[0] = noiseSigma*np.random.randn(800)
        noise[1] = noiseSigma*np.random.randn(800)
        # batch_data[i] = torch.tensor(sig + noise).to(device)
        datasets[i] = sig + noise
    
    rffPool_dict={}

    for i in range(targets.shape[0]):
        cfg = tuple(targets[i])
        sig = datasets[i]
        if cfg in rffPool_dict.keys():
            rffPool_dict[cfg].append(sig)
        else:
            rffPool_dict[cfg] =[sig]
    
    for cfg in list(rffPool_dict.keys()):
        rffPool_dict[cfg].append(sum(rffPool_dict[cfg])/len(rffPool_dict[cfg]))

    np.save(PATH+"poolRFF_"+"randomSNR"+str(snr_start)+"-"+str(snr_end)+"_dict.npy", rffPool_dict)

def AvgRFFDataset(PATH, dataFile):
    datasets, targets = LoadRFFDataset(PATH, dataFile)
    print(datasets.shape)
    print(targets.shape)
    print(targets[0])
    avg_dict = {}
    for i in range(targets.shape[0]):
        cfg = tuple(targets[i])
        sig = datasets[i]

        if cfg in avg_dict.keys():
            avg_dict[cfg].append(sig)
        else:
            avg_dict[cfg] =[sig]
    
    for cfg in list(avg_dict.keys()):
        avg_dict[cfg] = sum(avg_dict[cfg])/len(avg_dict[cfg])
    
    np.save(PATH+"avgRFF_dict.npy", avg_dict)
    return avg_dict

def AvgRFFDatasetSNRs(PATH, dataFile, snrdBs):
    datasets, targets = LoadRFFDataset(PATH, dataFile)
    print(datasets.shape)
    print(targets.shape)

    for snrdB in snrdBs:
        avg_dict = {}
        snr = 10**(snrdB/10)
        for i in range(targets.shape[0]):
            cfg = tuple(targets[i])
            sig = datasets[i]
            sigPower = np.sum(sig[0]**2+sig[1]**2)/len(sig[0])
            noisePower = sigPower/snr
            noiseSigma = np.sqrt(noisePower/2) # 2 way signal
            noise = np.zeros((2, 800))
            noise[0] = noiseSigma*np.random.randn(800)
            noise[1] = noiseSigma*np.random.randn(800)
            sig = sig + noise
            if cfg in avg_dict.keys():
                avg_dict[cfg].append(sig)
            else:
                avg_dict[cfg] = [sig]
    
        for cfg in list(avg_dict.keys()):
            avg_dict[cfg] = sum(avg_dict[cfg])/len(avg_dict[cfg])
            avg_dict[cfg] = avg_dict[cfg].astype(np.float32)
        
        np.save(PATH+"avgRFF_snr"+str(snrdB)+"_dict.npy", avg_dict)


if __name__ == "__main__":
    PATH = "./TcasRFFData/"
    dataFile = "ChipCAIRprocessedType3BlockOS20len800comb220V1p2Quant16.npz"
    # avg_dict = AvgRFFDataset(PATH, dataFile)

    snrdBs = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
    # RFFDatasetSNRs(PATH, dataFile, snrdBs)
    # RFFPoolSNRs(PATH, snrdBs)
    # RFFPool(PATH, dataFile)
    RFFPoolRandomNoise(PATH, dataFile)