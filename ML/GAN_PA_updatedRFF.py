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
from VAE import myVAE
from classifier_cnn import Classifier, Classifier2D, addNoise2Batch

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Sigmoid(),

            nn.Linear(256, 32),
            nn.Sigmoid(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
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

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)

def plot_timer(timer_curve, rfflabels, save=False, path=None): # key->rfflabel, value->[train_time, train_acc_fake_curve, test_time, val_acc_fake_curve]
    file_name='time_curve'
    for x in rfflabels:
        file_name+='_'+str(x)

    if save:
        np.save(path+file_name+".npy", timer_curve)

    max_len = 0
    for rfflabel in timer_curve.keys():
        train_time_step, train_time, train_acc_fake_curve, test_time_step, test_time, val_acc_fake_curve = timer_curve[rfflabel]
        max_len = max(max_len, len(train_time), len(train_acc_fake_curve), len(test_time), len(val_acc_fake_curve))
    
    for rfflabel in timer_curve.keys():
        train_time_step, train_time, train_acc_fake_curve, test_time_step, test_time, val_acc_fake_curve = timer_curve[rfflabel]
        train_interval = np.mean(train_time_step[1:])
        test_interval = np.mean(test_time_step[1:])

        if len(train_time) < max_len:
            # train_time += [train_time[-1]+i*train_interval for i in range(1, max_len - len(train_time)+1)]
            for i in range(1, max_len - len(train_time)+1):
                train_time.append(train_time[-1]+train_interval)
            train_acc_fake_curve += [train_acc_fake_curve[-1] for i in range(1, max_len - len(train_acc_fake_curve)+1)]
        
        if len(test_time) < max_len:
            # test_time += [test_time[-1]+i*test_interval for i in range(1, max_len - len(test_time)+1)]
            for i in range(1, max_len - len(test_time)+1):
                test_time.append(test_time[-1]+test_interval)
            val_acc_fake_curve += [val_acc_fake_curve[-1] for i in range(1, max_len - len(val_acc_fake_curve)+1)]

        assert len(train_time) == max_len
        assert len(test_time) == max_len
        assert len(train_acc_fake_curve) == max_len
        assert len(val_acc_fake_curve) == max_len
        plt.plot(train_time, train_acc_fake_curve, label = 'Train-RFF{}'.format(rfflabel))
        plt.plot(test_time, val_acc_fake_curve, label = 'Validation-RFF{}'.format(rfflabel))

    plt.vlines(x=60, ymin=0, ymax=1)
    plt.ylim((0, 1.01))
    plt.xlim((0, 1000))
    plt.xlabel('Time/sec')
    plt.ylabel('Fake Positive Rate')
    plt.title('FPR vs Time')
    plt.legend()

def getACC(labels_pred, labels):
    batch_size = labels.shape[0]
    _, cfg_pred = torch.topk(labels_pred, 9, axis=1)
    cfg_pred = cfg_pred.cpu().detach().numpy()
    _, cfg = torch.topk(labels, 9, axis=1)
    cfg = cfg.cpu().detach().numpy()

    acc=[]
    for i in range(batch_size):
        y_ = list(cfg_pred[i])
        y = list(cfg[i])
        y_.sort()
        y.sort()
        acc.append(y_==y)
    return acc

def main(seed, sys_argv, SNR, LR=5e-4, n_epoch=2000, batch_size=128, loss_weights=[1, 0.1, 1], save_ckp=False, save_time=True, pretrained=False, rfflabel_pre=None, load_path=None, save_path=None):
    print("\n+++++++++++++++++++Start main()+++++++++++++++++++") 
    bold_start = "\033[1m"
    bold_end = "\033[0;0m"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(bold_start+"device: "+bold_end, device)
    print((bold_start+"pretrained: {}"+bold_end).format(pretrained))
    if pretrained:
        print((bold_start+"rfflabel_pre: {}"+bold_end).format(rfflabel_pre))
    print((bold_start+"save_time: {}"+bold_end).format(save_time))
    print((bold_start+"save_ckp: {}"+bold_end).format(save_ckp))

    print((bold_start+"seed: {}"+bold_end).format(seed))
    print((bold_start+"SNR: {}dB"+bold_end).format(SNR))
    print((bold_start+"loss_weights: {}"+bold_end).format(loss_weights))
    print((bold_start+"learning_rate: {}"+bold_end).format(LR))
    rfflabels = sys_argv[1:]
    rfflabels = [int(x) for x in rfflabels]
    print((bold_start+"rfflabels:{}"+bold_end).format(rfflabels))

    if not os.path.exists(save_path+"/ckp"):
        os.makedirs(save_path+"/ckp")
    if not os.path.exists(save_path+"/time_curve"):
        os.makedirs(save_path+"/time_curve")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load classE data
    preamble = scipy.io.loadmat('./preamble')['preamble']
    preamble_real = torch.tensor(np.real(preamble).squeeze(), dtype=torch.float).to(device)
    preamble_imag = torch.tensor(-np.imag(preamble).squeeze(), dtype=torch.float).to(device)
    
    PATH="./"
    dataFile="ChipCAIRprocessedType3BlockOS20len800comb220V1p2Quant16.npz"
    file = np.load(PATH+dataFile)
    items = file.files
    items = file.files
    datasets_all, targets_all = file[items[0]], file[items[1]]
    datasets_220=np.zeros((220, 2, 800))
    targets_220=np.zeros((220, 12))
    for i in range(220):
        datasets_220[i]= datasets_all[i*730]
        targets_220[i]=targets_all[i*730]
    
    timer_curve={} # key->rfflabel, value->[train_time, train_acc_fake_curve, test_time, val_acc_fake_curve]
    print("---------------SNR:{}dB-----------------".format(SNR))
    
    if pretrained:
        files = os.listdir(load_path)
        for file in files:
            if len(file.split("_"))>2 and int(file.split("_")[0][3:]) == rfflabel_pre and float(file.split("_")[4][6:])>=0.9:
                gan_load = torch.load(load_path+file)
                generator=gan_load['generator']
                discriminator=gan_load['discriminator']
                classifier=gan_load['classifier']
                trainAcc = float(file.split("_")[3][8:])
                valAcc = float(file.split("_")[4][6:])
                trainBiAcc = float(file.split("_")[5][10:])
                valBiAcc = float(file.split("_")[6].split(".")[0][8:])
                print("pretrained label:{}, trainAcc:{}, valAcc:{}, trainBiAcc:{}, valBiAcc:{}".format(rfflabel_pre, trainAcc, valAcc, trainBiAcc, valBiAcc))
                break          
    else:
        print((bold_start+"pretrained label:{}"+bold_end).format(None))


    for rfflabel in rfflabels:#range(n_class):
        if pretrained and rfflabel == rfflabel_pre:
            continue
        print((bold_start+"rfflabel:{}"+bold_end).format(rfflabel))
        datasets = torch.zeros((730, 2, 800))
        targets = torch.zeros((730, 12))
        datasets = datasets_all[rfflabel*730:(rfflabel+1)*730]
        datasets=addNoise2Batch(datasets, snr=SNR, norm=False, device=device)
        # magn = np.max((np.abs(datasets.min()), np.abs(datasets.max())))
        # datasets = datasets/magn
        targets = targets_all[rfflabel*730:(rfflabel+1)*730]
        for i in range(targets.shape[0]-1):
            if (targets[i]!=targets[i+1]).all():
                print("!!!Target error!!!")
        datasets, targets = torch.tensor(datasets), torch.tensor(targets)
        print("datasets: ", datasets.shape)
        print("targets: ", targets.shape)
        print("datasets range: ", datasets.max(), datasets.min())

        input_dim=1
        for i in range(1, len(datasets.shape)):
            input_dim=input_dim*datasets.shape[i]

        rffDataset=Data.TensorDataset(datasets, targets)
        len_rffdataset = len(rffDataset)
        r_train, r_val = 0.6, 0.4
        len_trainset = int(np.ceil(r_train*len_rffdataset))
        len_valset = len_rffdataset - len_trainset
        print('rffdataset len:{}, trainset len:{}, valset len:{}'.format(len_rffdataset, len_trainset, len_valset)) 
        rffTrain, rffVal = Data.random_split(rffDataset, [len_trainset, len_valset], generator=torch.Generator().manual_seed(seed))
        datasetLoader = Data.DataLoader(rffDataset, batch_size=1, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed))
        trainLoader = Data.DataLoader(rffTrain, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed))
        valLoader = Data.DataLoader(rffVal, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed))

        if not pretrained:
            torch.manual_seed(seed)
            latent_dim=12
            # generator = myMemPolyVAE(preamble_real, preamble_imag, input_dim, hidden_dim, latent_dim, batch_size, device).to(device)
            generator = myVAE(preamble_real, preamble_imag, input_dim, latent_dim, batch_size, device).to(device)
            discriminator = Discriminator(input_dim).to(device)
            classifier = Classifier().to(device)
            checkpoint_path = "./cnn1001.pth"
            checkpoint = torch.load(checkpoint_path, map_location=device)
            classifier = checkpoint['classifier'].to(device)
        

        criterion_disc = torch.nn.BCELoss()
        criterion_cnn = torch.nn.BCELoss()

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
                data, labels = data.to(device, dtype=torch.float), labels.to(device,  dtype=torch.float)

                # Train discriminator
                optimizer_d.zero_grad()

                # Fake data_hat given by generator
                data_fake, mu, logvar = generator.forward(data) #if pretrained else generator.forward(data, device=device)
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
                fake_out_g, mu, logvar = generator.forward(data) #if pretrained else generator.forward(data, device=device)
                fake_out_d = discriminator.forward(fake_out_g)

                labels_ = classifier.forward(data)
                train_acc+=getACC(labels_, labels)

                labels_fake = classifier.forward(fake_out_g)
                train_acc_fake+=getACC(labels_fake, labels)

                loss_vae = generator.loss_function(fake_out_g, data, mu, logvar)
                loss_cnn = criterion_cnn(labels_fake, labels)
                loss_disc = criterion_disc(fake_out_d, real_label)
                loss_g = loss_weights[0]*loss_cnn + loss_weights[1]*loss_vae + loss_weights[2]*loss_disc
                loss_g.backward()                
                optimizer_g.step()            
                train_loss_g+=loss_g.item()
            end_epoch_train = time.time()

            start_epoch_test = time.time()
            for batch_idx, (data_test, labels_test) in enumerate(valLoader):
                data_test, labels_test = data_test.to(device, dtype=torch.float), labels_test.to(device,  dtype=torch.float)

                # Fake data_hat given by generator
                data_fake_test, mu_test, logvar_test = generator.forward(data_test) #if pretrained else generator.forward(data_test, device=device)
                fake_out_d_test = discriminator.forward(data_fake_test)
                loss_fake_d_test = criterion_disc(fake_out_d_test, fake_label)
                fake_out_d_bool_test = (fake_out_d_test-0.5)>0
                val_biacc_fake.append((fake_out_d_bool_test==fake_label_bool).sum()/fake_out_d_test.shape[0]) 

                # Real data
                real_out_d_test = discriminator.forward(data_test)
                loss_real_d_test = criterion_disc(real_out_d_test, real_label)
                real_out_d_bool_test = (real_out_d_test-0.5)>0
                val_biacc.append((real_out_d_bool_test==real_label_bool).sum()/real_out_d_test.shape[0]) 

                # Update loss_d for discriminator
                loss_d_test = loss_fake_d_test+loss_real_d_test
                val_loss_d+=loss_d_test.item()

                # Train generator
                fake_out_g_test, mu_test, logvar_test = generator.forward(data_test) #if pretrained else generator.forward(data_test, device=device)
                fake_out_d_test = discriminator.forward(fake_out_g_test)

                labels_ = classifier.forward(data_test)

                labels_ = classifier.forward(data_test)
                val_acc+=getACC(labels_, labels_test)

                labels_fake_test = classifier.forward(fake_out_g_test)
                val_acc_fake+=getACC(labels_fake_test, labels_test)

                loss_vae_test = generator.loss_function(fake_out_g_test, data_test, mu_test, logvar_test)
                loss_cnn_test = criterion_cnn(labels_fake_test, labels_test)
                loss_disc_test =criterion_disc(fake_out_d_test, real_label)
                loss_g_test = loss_weights[0]*loss_cnn_test + loss_weights[1]*loss_vae_test + loss_weights[2]*loss_disc_test
                val_loss_g+=loss_g_test.item()

            end_epoch_test = time.time()
            
            train_time.append(end_epoch_train - start)
            train_time_step.append(end_epoch_train - start_epoch_train)
            # print("time: {} sec for epoch {}".format(time_epoch_train, epoch))

            test_time.append(end_epoch_test - start)
            test_time_step.append(end_epoch_test - start_epoch_train)
            # print("test time: {} sec for epoch {}".format(time_epoch_test, epoch))

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


            if (epoch+1) % 500 ==0:
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
                if not pretrained:
                    checkpoint_path = save_path+"ckp/rff"+str(rfflabel)+"_SNR"+str(SNR)+"_epoch"+str(epoch+1)+"_trainAcc"+str(round(best_train_acc,4))+"_valAcc"+str(round(best_val_acc, 4))+"_trainBiAcc"+str(round(train_biacc_fake.item(),4))+"_valBiAcc"+str(round(val_biacc_fake.item(),4))+".gan"
                else:
                    checkpoint_path = save_path+"ckp/rff"+str(rfflabel)+"_SNR"+str(SNR)+"_pretrained"+str(rfflabel_pre)+"_epoch"+str(epoch+1)+"_trainAcc"+str(round(best_train_acc,4))+"_valAcc"+str(round(best_val_acc, 4))+"_trainBiAcc"+str(round(train_biacc_fake.item(),4))+"_valBiAcc"+str(round(val_biacc_fake.item(),4))+".gan"
                end = time.time()
                if save_ckp:
                    torch.save({
                            'epoch': epoch,
                            'generator': generator,
                            'discriminator': discriminator,
                            'classifier': classifier,
                            }, checkpoint_path)
                # comment when fix n_epoch
                else:
                    print("!!max_acc>=0.9!! checkpoint:{}".format(checkpoint_path))
                break


            if (epoch+1) % 1000 ==0:
                if not pretrained:
                    checkpoint_path = save_path+"ckp/rff"+str(rfflabel)+"_SNR"+str(SNR)+"_epoch"+str(best_epoch+1)+"_trainAcc"+str(round(best_train_acc,4))+"_valAcc"+str(round(best_val_acc, 4))+"_trainBiAcc"+str(round(best_train_biacc.item(),4))+"_valBiAcc"+str(round(best_val_biacc.item(),4))+".gan"
                else:
                    checkpoint_path = save_path+"ckp/rff"+str(rfflabel)+"_SNR"+str(SNR)+"_pretrained"+str(rfflabel_pre)+"_epoch"+str(best_epoch+1)+"_trainAcc"+str(round(best_train_acc,4))+"_valAcc"+str(round(best_val_acc, 4))+"_trainBiAcc"+str(round(best_train_biacc.item(),4))+"_valBiAcc"+str(round(best_val_biacc.item(),4))+".gan"
                if save_ckp:
                    torch.save(best_ckpt, checkpoint_path)
                # comment when fix n_epoch
                else:
                    print("!!(epoch+1) % 1000!! checkpoint:{}".format(checkpoint_path))
        if not end:
            end = time.time()
        assert len(train_time) == len(train_acc_fake_curve)
        assert len(test_time) == len(val_acc_fake_curve)
        timer_curve[rfflabel] =[train_time_step, train_time, train_acc_fake_curve, test_time_step, test_time, val_acc_fake_curve]
    
    if save_time:
        if not pretrained:
            path = save_path+"time_curve/pre_"
        else:
            path = save_path+"time_curve/"
        plot_timer(timer_curve, rfflabels, save=save_time, path=path)

    print("+++++++++++++++++++End main()+++++++++++++++++++\n") 