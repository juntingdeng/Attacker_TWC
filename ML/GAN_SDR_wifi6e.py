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
from VAE import myMemPolyVAE
from classifier_cnn import addNoise2Batch
from GAN_SDR import ClassifierSDR, Discriminator
# import matlab.engine
import __main__



def main(seed, nSDR, LR, n_epoch, batch_size, SNR, degLen, memLen, clf_path, save_ckp, save_time, save_path):
    print("+++++++++++++++++++Start main()+++++++++++++++++++") 
    samples_perSDR=300
    start = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("seed: {}".format(seed))
    print("start:{}, nSDR:{}, SNR:{}dB".format(start, nSDR, SNR))
    print("device: ", device)
    print("save_ckp: {}".format(save_ckp))
    print("save_time: {}".format(save_time))

    if not os.path.exists(save_path+"/ckp"):
        os.makedirs(save_path+"/ckp")
    if not os.path.exists(save_path+"/time_curve"):
        os.makedirs(save_path+"/time_curve")

    setattr(__main__, "ClassifierSDR", ClassifierSDR)
    classifier = ClassifierSDR(nSDR=nSDR)
    clf_load=torch.load(clf_path, map_location=device)
    # clf_load=torch.load("./checkpoint/ClassifierSDR/11SDR_normseperate_epoch2000.cnn", map_location=device)
    classifier=clf_load["classifier"]
    train_acc=clf_load["train_acc"]
    val_acc=clf_load["val_acc"]
    print("Load clf, train_acc:{:.2f}%, val_acc:{:.2f}%".format(train_acc.item()*100, val_acc.item()*100))

    preamble_path = "./data/wifi6e16qam/txSelectedPreambleDecimateDefault.mat"
    preamble_comp = scipy.io.loadmat(preamble_path)
    keys = list(preamble_comp.keys())
    preamble_comp = preamble_comp[keys[-1]].reshape(-1)
    preamble_real = torch.tensor(np.real(preamble_comp), dtype=torch.float).to(device)
    preamble_imag = torch.tensor(np.imag(preamble_comp), dtype=torch.float).to(device)
    print("preamble real: ", preamble_real.shape)
    print("preamble imag: ", preamble_imag.shape)

    datasetdict_load = np.load("./data/wifi6e16qam/wifi6e220RFFs.npy", allow_pickle=True).item()
    print(datasetdict_load.keys())
    dataset_load, target_load = datasetdict_load['dataset'], datasetdict_load['target']
    nlabels = nSDR
    nrepeat = 300
    datasets_all = np.zeros((nlabels*nrepeat, 2, 480))
    targets_all = np.zeros(nlabels*nrepeat)
    for labeli in range(nlabels):
        for repeati in range(nrepeat):
            datasets_all[labeli*nrepeat+repeati] = dataset_load[labeli]
            targets_all[labeli*nrepeat+repeati] = target_load[labeli]
    
    datasetall_magn = np.max((np.abs(datasets_all.min()), np.abs(datasets_all.max())))
    datasets_all /= datasetall_magn
    
    for SDRlabel in range(start, start+nSDR):
        print("\n-----------------SNR:{}, SDRlabel:{}-----------------".format(SNR, SDRlabel))
        timer_curve={}
        max_train_acc=0
        max_val_acc=0

        datasets = datasets_all[SDRlabel*nrepeat: (SDRlabel+1)*nrepeat]
        targets = targets_all[SDRlabel*nrepeat: (SDRlabel+1)*nrepeat]
        datasets = addNoise2Batch(datasets, SNR, shape=datasets.shape[1:], norm=False, device=device)        
        datasets=torch.tensor(datasets, dtype=float).to(device)
        targets=torch.tensor(targets, dtype=torch.int64).to(device)
        print("datasets: ", datasets.shape)
        print("targets: ", targets.shape)
        print("datasets range: ", datasets.max(), datasets.min())

        rffDataset=Data.TensorDataset(datasets, targets)
        len_rffdataset = len(rffDataset)
        r_train, r_val = 0.6, 0.4
        len_trainset = int(np.ceil(r_train*len_rffdataset))
        len_valset = len_rffdataset - len_trainset
        print('rffdataset len:{}, trainset len:{}, valset len:{}'.format(len_rffdataset, len_trainset, len_valset)) 
        rffTrain, rffVal= Data.random_split(rffDataset, [len_trainset, len_valset],generator=torch.manual_seed(seed))
        datasetLoader = Data.DataLoader(rffDataset, batch_size=1, shuffle=True, drop_last=True,generator=torch.manual_seed(seed))
        trainLoader = Data.DataLoader(rffTrain, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.manual_seed(seed))
        valLoader = Data.DataLoader(rffVal, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.manual_seed(seed))
        print("len trainLoader", len(trainLoader))
        print("len valLoader", len(valLoader))

        input_dim=1
        for i in range(1, len(datasets.shape)):
            input_dim=input_dim*datasets.shape[i]
        latent_dim=2
        poly_dim= degLen*memLen #15
        generator = myMemPolyVAE(txReal=preamble_real, 
                                 txImag=preamble_imag, 
                                 input_dim=input_dim, 
                                 latent_dim=latent_dim, 
                                 poly_dim=poly_dim, 
                                 batch_size=batch_size,  
                                 device=device,
                                 train=True).to(device)        
        discriminator = Discriminator(input_dim).to(device)
        
        criterion_disc = nn.BCELoss()
        criterion_cnn = nn.CrossEntropyLoss()

        optimizer_g = torch.optim.Adam(generator.parameters(), lr = LR)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = LR)

        real_label = Variable(torch.ones(batch_size, 1), requires_grad=False).to(device)
        fake_label = Variable(torch.zeros(batch_size, 1), requires_grad=False).to(device)
        real_label_bool = Variable(torch.ones(batch_size, 1)>0, requires_grad=False).to(device)
        fake_label_bool = Variable(torch.zeros(batch_size, 1)>0, requires_grad=False).to(device)

        train_loss_curve_d=[]
        train_loss_curve_g=[]
        train_acc_curve=[]
        train_acc_fake_curve=[]

        val_loss_curve_d=[]
        val_loss_curve_g=[]
        val_acc_curve=[]
        val_acc_fake_curve=[]

        train_time=[]
        train_time_step=[]
        test_time=[]
        test_time_step=[]
        #start training timer
        start, end = time.time(), None

        for epoch in range(n_epoch):
            # print("epoch: ", epoch)
            train_loss_d=0
            train_loss_g=0
            val_loss_d=0
            val_loss_g=0
            train_acc = []
            val_acc=[]
            train_acc_fake = []
            val_acc_fake=[]

            train_biacc = []
            val_biacc = []
            train_biacc_fake = []
            val_biacc_fake = []

            start_epoch_train = time.time()
            for batch_idx, (data, labels) in enumerate(trainLoader):
                # print("batch_idx: ", batch_idx)
                data, labels = data.to(device, dtype=torch.float), labels.to(device,  dtype=torch.int64)

                # Train discriminator
                optimizer_d.zero_grad()

                # Fake data_hat given by generator
                data_fake, mu, logvar = generator.forward(data, device=device)
                fake_out_d = discriminator.forward(data_fake)
                loss_fake_d = criterion_disc(fake_out_d, fake_label)
                fake_out_d_bool = (fake_out_d-0.5)>0
                train_biacc_fake.append((fake_out_d_bool==fake_label_bool).sum()/fake_out_d.shape[0])

                # Real data
                real_out_d = discriminator.forward(data)
                loss_real_d = criterion_disc(real_out_d, real_label)
                real_out_d_bool = (real_out_d-0.5)>0
                train_biacc.append((real_out_d_bool==real_label_bool).sum()/real_out_d.shape[0])

                # Update loss_d for discriminator
                if torch.max(fake_out_d)>1 or torch.min(fake_out_d)<0 or torch.max(real_out_d)>1 or torch.min(real_out_d)<0:
                    print("fake_out_d: ", torch.max(fake_out_d), torch.min(fake_out_d))
                    print("real_out_d: ", torch.max(real_out_d), torch.min(real_out_d))
                    print("loss_fake_d: ", torch.max(loss_fake_d), torch.min(loss_fake_d))
                    print("loss_real_d: ", torch.max(loss_real_d), torch.min(loss_real_d))
                    print("\n")
                    break
                loss_d = loss_fake_d+loss_real_d
                loss_d.backward()
                optimizer_d.step()
                train_loss_d+=loss_d.item()

                # Train generator
                optimizer_g.zero_grad()
                fake_out_g, mu, logvar = generator.forward(data, device=device)
                fake_out_d = discriminator.forward(fake_out_g)

                labels_ = classifier.forward(data)
                labels_arg = torch.argmax(labels_, dim=1)
                for i in range(batch_size):
                    train_acc.append(labels_arg[i].detach().cpu().numpy()==labels[i].detach().cpu().numpy())

                labels_fake = classifier.forward(fake_out_g)
                labels_fake_arg = torch.argmax(labels_fake, dim=1)
                for i in range(batch_size):
                    train_acc_fake.append(labels_fake_arg[i].detach().cpu().numpy()==labels[i].detach().cpu().numpy())

                loss_vae = generator.loss_function(fake_out_g, data, mu, logvar)
                loss_cnn = criterion_cnn(labels_fake, labels)
                loss_disc = criterion_disc(fake_out_d, real_label)
                loss_g = 1*loss_cnn + 0.01*loss_vae + 0.1*loss_disc
                loss_g.backward()
                optimizer_g.step()
                train_loss_g+=loss_g.item()
            end_epoch_train = time.time()

            start_epoch_test = time.time()
            for batch_idx, (data, labels) in enumerate(valLoader):
                data, labels = data.to(device, dtype=torch.float), labels.to(device,  dtype=torch.int64)

                # Fake data_hat given by generator
                data_fake, mu, logvar = generator.forward(data, device=device)
                fake_out_d = discriminator.forward(data_fake)
                loss_fake_d = criterion_disc(fake_out_d, fake_label)
                fake_out_d_bool = (fake_out_d-0.5)>0
                val_biacc_fake.append((fake_out_d_bool==fake_label_bool).sum()/fake_out_d.shape[0]) 

                # Real data
                real_out_d = discriminator.forward(data)
                loss_real_d = criterion_disc(real_out_d, real_label)
                real_out_d_bool = (real_out_d-0.5)>0
                val_biacc.append((real_out_d_bool==real_label_bool).sum()/real_out_d.shape[0]) 

                # Update loss_d for discriminator
                loss_d = loss_fake_d+loss_real_d
                val_loss_d+=loss_d.item()

                # Train generator
                fake_out_g, mu, logvar = generator.forward(data, device=device)
                fake_out_d = discriminator.forward(fake_out_g)

                labels_ = classifier.forward(data)
                labels_arg = torch.argmax(labels_, dim=1)
                for i in range(batch_size):
                    val_acc.append(labels_arg[i].detach().cpu().numpy()==labels[i].detach().cpu().numpy())

                labels_fake = classifier.forward(fake_out_g)
                labels_fake_arg = torch.argmax(labels_fake, dim=1)
                for i in range(batch_size):
                    val_acc_fake.append(labels_fake_arg[i].detach().cpu().numpy()==labels[i].detach().cpu().numpy())

                loss_vae = generator.loss_function(fake_out_g, data, mu, logvar)
                loss_cnn = criterion_cnn(labels_fake, labels)
                loss_disc = criterion_disc(fake_out_d, real_label)
                loss_g =  1*loss_cnn + 0.01*loss_vae + 0.1*loss_disc
                val_loss_g+=loss_g.item()
            end_epoch_test = time.time()
            
            train_time.append(end_epoch_train - start)
            train_time_step.append(end_epoch_train - start_epoch_train)

            test_time.append(end_epoch_test - start)
            test_time_step.append(end_epoch_test - start_epoch_train)

            train_loss_d = train_loss_d/len(trainLoader)
            train_loss_g = train_loss_g/len(trainLoader)
            train_acc = sum(train_acc)/(len(train_acc))
            train_acc_fake = sum(train_acc_fake)/(len(train_acc_fake))
            train_biacc = sum(train_biacc)/(len(train_biacc))
            train_biacc_fake = sum(train_biacc_fake)/(len(train_biacc_fake))

            val_loss_d = val_loss_d/len(valLoader)
            val_loss_g = val_loss_g/len(valLoader)
            val_acc = sum(val_acc)/(len(val_acc))
            val_acc_fake = sum(val_acc_fake)/(len(val_acc_fake))
            val_biacc = sum(val_biacc)/(len(val_biacc))
            val_biacc_fake = sum(val_biacc_fake)/(len(val_biacc_fake))

            train_loss_curve_d.append(train_loss_d)
            train_loss_curve_g.append(train_loss_g)
            train_acc_curve.append(train_acc)
            train_acc_fake_curve.append(train_acc_fake)

            val_loss_curve_d.append(val_loss_d)
            val_loss_curve_g.append(val_loss_g)
            val_acc_curve.append(val_acc)
            val_acc_fake_curve.append(val_acc_fake)

            if (epoch+1) % 100 ==0:
                print('Training Epoch: {} \t GLoss: {:.3f}, DLoss: {:.3f}, ClfAcc: {:.3f}, ClfAcc(Fake): {:.3f}, BiAcc: {:.3f}, BiAcc(Fake): {:.3f}'.format(epoch+1, train_loss_g, train_loss_d, train_acc, train_acc_fake, train_biacc, train_biacc_fake))
                print('Validation Epoch: {} \t GLoss: {:.3f}, DLoss: {:.3f}, ClfAcc: {:.3f}, ClfAcc(Fake): {:.3f}, BiAcc: {:.3f}, BiAcc(Fake): {:.3f}'.format(epoch+1, val_loss_g, val_loss_d, val_acc, val_acc_fake, val_biacc, val_biacc_fake))

            if epoch == 0:
                best_ckpt = {
                    'epoch': epoch,
                    'generator': generator,
                    'discriminator': discriminator,
                    'classifier': classifier,
                    }
                best_train_acc = train_acc_fake
                best_val_acc = val_acc_fake
                best_epoch = epoch
                best_train_biacc = train_biacc_fake
                best_val_biacc = val_biacc_fake
            elif val_acc_fake>=best_val_acc:
                best_ckpt = {
                    'epoch': epoch,
                    'generator': generator,
                    'discriminator': discriminator,
                    'classifier': classifier,
                    }
                best_train_acc = train_acc_fake
                best_val_acc = val_acc_fake
                best_epoch = epoch
                best_train_biacc = train_biacc_fake
                best_val_biacc = val_biacc_fake

            if train_acc_fake>=0.9 and val_acc_fake>=0.9:
                best_train_acc, best_val_acc = train_acc_fake, val_acc_fake
                assert best_train_acc == train_acc_fake_curve[-1]
                assert best_val_acc == val_acc_fake_curve[-1]
                checkpoint_path = save_path+"ckp/sdr"+str(SDRlabel)+"_SNR"+str(SNR)+"_epoch"+str(epoch+1)+"_trainAcc"+str(round(best_train_acc,4))+"_valAcc"+str(round(best_val_acc, 4))+"_trainBiAcc"+str(round(train_biacc_fake.item(),4))+"_valBiAcc"+str(round(val_biacc_fake.item(),4))+".gan"
                end = time.time()
                if save_ckp:
                    torch.save({
                        'epoch': epoch,
                        'generator': generator,
                        'discriminator': discriminator,
                        'classifier': classifier,
                        }, checkpoint_path)
                else:
                    print("!!max_acc>=0.9!! checkpoint:{}".format(checkpoint_path))
                break


            if (epoch+1) % 1000 ==0:
                checkpoint_path = save_path+"ckp/sdr"+str(SDRlabel)+"_SNR"+str(SNR)+"_epoch"+str(best_epoch+1)+"_trainAcc"+str(round(best_train_acc,4))+"_valAcc"+str(round(best_val_acc, 4))+"_trainBiAcc"+str(round(best_train_biacc.item(),4))+"_valBiAcc"+str(round(best_val_biacc.item(),4))+".gan"
                if save_ckp:
                    torch.save(best_ckpt, checkpoint_path)
                else:
                    print("!!(epoch+1) % 1000!! checkpoint:{}".format(checkpoint_path))

        if not end:
            end = time.time()
        assert len(train_time) == len(train_acc_fake_curve)
        assert len(test_time) == len(val_acc_fake_curve)
        timer_curve[SDRlabel] =[train_time_step, train_time, train_acc_fake_curve, test_time_step, test_time, val_acc_fake_curve]
    
        if save_time:
            path = save_path+"time_curve/"
            file_name='time_curve'+'_'+str(SDRlabel)
            np.save(path+file_name+".npy", timer_curve)
    print("+++++++++++++++++++End main()+++++++++++++++++++\n") 

if __name__ == "__main__":
    seed=345
    degLen=5
    memLen=3
    SNR = sys.argv[1:]
    SNR = [int(x) for x in SNR][0]
    nSDR = 220
    LR = 5e-3
    n_epoch = 5000
    batch_size = 32
    clf_path =  "./checkpoint2/classifier/wifi6e_220RFF_nrepeat300_epoch1000.cnn"
    save_path = "./checkpoint2/gan/gan_poly_ray_WIFI6E220_normAll_nosnrclf_bn_poly_deg"+str(degLen)+"_mem"+str(memLen)+"/snr"+str(SNR)+"/seed"+str(seed)+"/"
    main(seed=seed,
         nSDR=nSDR,
         LR=LR,
         n_epoch=n_epoch,
         batch_size=batch_size,
         SNR=SNR,
         degLen=degLen,
         memLen=memLen,
         clf_path=clf_path,
         save_ckp=True,
         save_time=True,
         save_path=save_path)