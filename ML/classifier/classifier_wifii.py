import torch
import torch.nn as  nn
import torch.optim as optim
import numpy as np
import scipy.io
import torch.utils.data as Data
import matplotlib.pyplot as plt
from GAN_SDR import ClassifierSDR
from classifier_cnn import addNoise2Batch

def train(batch_size, nRFFs, nrepeat, nepochs, lr, train_loader, val_loader, save, save_path, device):
    train_loss_curve = []
    train_acc_curve = []
    val_loss_curve = []
    val_acc_curve = []
    best_acc = 0
    classfier = ClassifierSDR(nSDR=nRFFs).to(device)
    classfier = classfier
    optimizer = optim.Adam(classfier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(nepochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        for _, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels_ = classfier(data)
            loss = criterion(labels_, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += torch.sum(torch.tensor(torch.argmax(labels_, -1) == labels, dtype=torch.float), 0)

        for _, (data, labels) in enumerate(val_loader):
            labels_ = classfier(data)
            loss = criterion(labels_, labels)

            val_loss += loss.item()
            val_acc += torch.sum(torch.tensor(torch.argmax(labels_, -1) == labels, dtype=torch.float), 0)
        
        train_loss = train_loss/len(train_loader)
        train_acc = train_acc/(batch_size*len(train_loader))
        val_loss = val_loss/len(val_loader)
        val_acc = val_acc/(batch_size*len(val_loader))

        train_loss_curve.append(train_loss)
        train_acc_curve.append(train_acc)
        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_acc)

        ## Archive best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = {
                'classifier': classfier,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'batch_size': batch_size
            }
        
        ## Print and save
        if (epoch+1)%200 == 0:
            print("Epoch:{}:".format(epoch+1))
            print("Train loss:{}, acc:{}".format(train_loss, train_acc))
            print("Val loss:{}, acc:{}".format(val_loss, val_acc))
            if save:
                torch.save(best_model, save_path+"wifi6e_{}RFF_nrepeat{}_epoch{}.cnn".format(nRFFs,nrepeat,epoch+1))

def main(batch_size, nlabels, nrepeat, nepochs, lr, SNR, seed, save, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preamble_path = "./data/wifi6e16qam/txSelectedPreambleDecimateDefault.mat"
    preamble_comp = scipy.io.loadmat(preamble_path)
    print(preamble_comp.keys())
    keys = list(preamble_comp.keys())
    preamble_comp = preamble_comp[keys[-1]].reshape(-1)

    preamble = np.zeros((2, 480))
    preamble[0] = np.real(preamble_comp)
    preamble[1] = np.imag(preamble_comp)
    print("preamble shape:{}".format(preamble.shape))
    # plt.plot(preamble[0])
    # plt.plot(preamble[1])

    datasetdict_load = np.load("./data/wifi6e16qam/wifi6e{}RFFs.npy".format(nlabels), allow_pickle=True).item()
    print(datasetdict_load.keys())
    dataset_load, target_load = datasetdict_load['dataset'], datasetdict_load['target']
    
    datasets = np.zeros((nlabels*nrepeat, 2, 480))
    targets = np.zeros(nlabels*nrepeat)
    for labeli in range(nlabels):
        for repeati in range(nrepeat):
            datasets[labeli*nrepeat+repeati] = dataset_load[labeli]
            targets[labeli*nrepeat+repeati] = target_load[labeli]
    
    dataset_magn = np.max((np.abs(datasets.min()), np.abs(datasets.max())))
    datasets /= dataset_magn
    datasets = addNoise2Batch(datasets, SNR, shape=datasets.shape[1:], norm=False, device=device)
    datasets = torch.tensor(datasets, dtype=torch.float).to(device)
    targets = torch.tensor(targets, dtype=torch.int64).to(device)
    wifiDataset = Data.TensorDataset(datasets, targets)
    ratio = [0.6, 0.4]
    len_trainset = len(wifiDataset)*ratio[0]
    len_valset = len(wifiDataset) - len_trainset
    wifiTrain, wifiVal = Data.random_split(wifiDataset, ratio, generator=torch.manual_seed(seed))
    trainLoader = Data.DataLoader(wifiTrain, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.manual_seed(seed))
    valLoader = Data.DataLoader(wifiVal, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.manual_seed(seed))
    print("len trainLoader", len(trainLoader))
    print("len valLoader", len(valLoader))

    train(batch_size=batch_size,
          nRFFs=nlabels,
          nrepeat=nrepeat,
          nepochs=nepochs,
          lr = lr,
          train_loader=trainLoader,
          val_loader=valLoader,
          save = save,
          save_path = save_path,
          device=device)

if __name__ == "__main__":
    batch_size = 32
    nlabels = 220
    nrepeat = 300
    nepochs = 1000
    lr = 1e-4
    SNR = 35
    seed = 0
    save = True
    save_path = "./checkpoint2/classifier/"
    main(batch_size=batch_size,
         nlabels=nlabels,
         nrepeat=nrepeat,
         nepochs=nepochs,
         lr = lr,
         SNR=SNR,
         seed=seed,
         save=save,
         save_path=save_path)