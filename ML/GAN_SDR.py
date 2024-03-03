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
from classifier_cnn import Classifier, Classifier2D, addNoise2Batch
# import matlab.engine
import __main__


class ClassifierSDR(nn.Module):
    def __init__(self, nSDR):
        super().__init__()

        #For 800 length 
        self.conv0 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5,stride=1, padding=2) # => 8*800
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,stride=1, padding=2) # => 8*800
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8,stride=2, padding=3) # => 16*400
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, stride=2, padding=2) # => 16*200

        self.linear1 = nn.Linear(in_features=32*200, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=nSDR)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        #For 800 length 
        x=F.relu(self.conv0(x))
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))

        # x=self.maxpool1(x)
        flatten = nn.Flatten()
        x=flatten(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        flatten = nn.Flatten()
        x=flatten(x)
        x=self.model(x)
        if torch.isnan(x).any():
            print("Discriminator output nan")
        return x

def corelation(x, y): #x,y are both n-d vector
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    cov_xy=0
    var_x=0
    var_y=0
    for i in range(x.shape[0]):
        cov_xy+=(x[i]-x_mean)*(y[i]-y_mean)
        var_x+=(x[i]-x_mean)*(x[i]-x_mean)
        var_y+=(y[i]-y_mean)*(y[i]-y_mean)
    corr = cov_xy/(np.sqrt(var_x*var_y))
    return corr

def classify_by_correlation(signals, datasets_220, targets_220):
    batch_size=signals.shape[0]
    targets=np.zeros((batch_size, 12))
    for bc in range(batch_size):
        r=np.zeros((220,))
        for i in range(datasets_220.shape[0]):
            r_real = corelation(signals[bc][0], datasets_220[i][0])
            r_imag = corelation(signals[bc][1], datasets_220[i][1])
            r[i] = (r_real+r_imag)/2
        rff_idx = np.argmax(r)
        targets[bc]=targets_220[rff_idx]
    return targets

def ReadSignalFromCsv(file, channel=2, num_samples=800):
    inf=[-1.7976931348623158e+308, 1.7976931348623158e+308]
    signals=[]
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        i=0
        while True:
            read_line = next(spamreader)
            if inf[0]<=float(read_line[0])<=inf[1]:
                break
        total_col = len(read_line)
        total_points = total_col//channel
        read_all_signs = np.zeros((total_points, channel, num_samples))

        row_idx=0
        for row in spamreader:
            for i in range(len(row)):
                point_idx = i//channel
                read_all_signs[point_idx, i%channel, row_idx]=float(row[i])
            row_idx+=1
            if row_idx>=num_samples:
                break
    return read_all_signs

def main(seed, SNR, degLen, memLen, save_ckp, save_time, save_path):
    print("+++++++++++++++++++Start main()+++++++++++++++++++") 
    samples_perSDR=300
    start = 0
    nSDR = 11
    LR = 1e-3
    n_epoch = 2000
    batch_size = 32
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
    clf_load=torch.load("./checkpoint2/classifier/11SDR_nosnr_normall_epoch200.cnn", map_location=device)
    # clf_load=torch.load("./checkpoint/ClassifierSDR/11SDR_normseperate_epoch2000.cnn", map_location=device)
    classifier=clf_load["classifier"]
    train_acc=clf_load["train_acc"]
    val_acc=clf_load["val_acc"]
    test_acc=clf_load["test_acc"]
    print("Load clf, train_acc:{:.2f}%, val_acc:{:.2f}%, test_acc:{:.2f}%".format(train_acc.item()*100, val_acc.item()*100, test_acc.item()*100))

    idata=ReadSignalFromCsv('./data/IDataTime.csv')[0][1]
    qdata=-ReadSignalFromCsv('./data/QDataTime.csv')[0][1]
    preamble_real = torch.tensor(idata.squeeze(), dtype=torch.float).to(device)
    preamble_imag = torch.tensor(-qdata.squeeze(), dtype=torch.float).to(device)
    print("preamble real: ", preamble_real.shape)
    print("preamble imag: ", preamble_imag.shape)
    
    matFiles=["pluto1_0.25meters_run2.mat",
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

    datasets_allSDR=None
    targets_allSDR=None
    samples_perSDR=300
    duplicate=4
    magn=17.032302068590234
    for matfilei in range(len(matFiles)):
        matfile = matFiles[matfilei]
        datasets_load=scipy.io.loadmat("./data/0.25meters/"+matfile)['packet_equalized_allrecord']
        datasets_load = datasets_load[:samples_perSDR]
        targets_load = matfilei*np.ones(datasets_load.shape[0])
        print("SDR: {}, range: {}~{}".format(matfilei, datasets_load.min(), datasets_load.max()))
        datasets_allSDR = np.concatenate([datasets_allSDR, datasets_load], axis=0) if matfilei!=0 else datasets_load
        targets_allSDR = np.concatenate([targets_allSDR, targets_load], axis=0) if matfilei!=0 else targets_load
    datasets_allSDR_magn = np.max((np.abs(datasets_allSDR.min()), np.abs(datasets_allSDR.max())))
    datasets_allSDR = datasets_allSDR/datasets_allSDR_magn
    print("datasets load all sdr shape: ", datasets_allSDR.shape)
    print("targets load all sdr shape: ", targets_allSDR.shape)
    print("datasets load all sdr and norm range: {}~{}".format(datasets_allSDR.min(), datasets_allSDR.max()))
    
    for SDRlabel in range(start, start+nSDR):
        print("\n-----------------SNR:{}, SDRlabel:{}-----------------".format(SNR, SDRlabel))
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
                                 degLen=degLen, 
                                 memLen=memLen, 
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
    seed=0
    degLen=9
    memLen=7
    SNR = sys.argv[1:]
    SNR = [int(x) for x in SNR][0]
    save_path = "./checkpoint2/gan/gan_poly_ray_SDR_normAll_nosnrclf_bn_dup_poly_deg"+str(degLen)+"_mem"+str(memLen)+"/snr"+str(SNR)+"/seed"+str(seed)+"/"
    main(seed=seed,
         SNR=SNR,
         degLen=degLen,
         memLen=memLen,
         save_ckp=True,
         save_time=True,
         save_path=save_path)