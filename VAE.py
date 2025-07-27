import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.utils.data as Data
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io
# import matlab.engine

seed = 0
def applyPolyCoefComplex(xBatch, coefMatBatch):
    degLen=5
    memLen=3
    memLenM1=memLen-1
    yBatch=np.zeros_like(xBatch)
    for batchIdx in range(xBatch.shape[0]):
        x=xBatch[batchIdx]
        coefMat = coefMatBatch[batchIdx]
        coefMatTrans=coefMat.transpose()
        coeftMatReshaped = coefMatTrans.reshape((-1))
        xLen=x.shape[0]
        y=np.zeros_like(x)
        for timeIdx in range(memLen, xLen+1):
            xTerms=np.zeros((memLen*degLen), dtype=np.complex_)
            xTime=x[timeIdx-memLen: timeIdx]
            xTime=np.flipud(xTime)
            xTerms[:memLen] = xTime[:]
            for degIdx in range(1, degLen):
                startPos = degIdx*memLen
                endPos = startPos+memLen
                tmp=np.abs(xTime)**degIdx
                xTerms[startPos:endPos] = xTime[:]*tmp[:]
            y_tmp=coeftMatReshaped*xTerms
            y[timeIdx-1] = sum(y_tmp)
        yBatch[batchIdx]=y
    return yBatch

def fastPolyPrefix(xBatch): #(batch_size, 2, 800)
    degLen=5
    memLen=3
    xLen=xBatch.shape[-1]
    time_steps = xLen+1 - memLen
    batch_size = xBatch.shape[0]
    x_extented = np.zeros((batch_size, 2, time_steps, memLen))
    for t in range(time_steps):
        x_extented[:, :, t] = np.flip(xBatch[:, :, t: t+memLen], axis=-1)
    x_extented_abs = np.power(x_extented, 2) 
    x_extented_abs = np.sqrt(x_extented_abs[:, 0, :, :] + x_extented_abs[:, 1, :, :]) #(batch_size, time_steps, memLen)
    xTerms = np.stack([x_extented_abs for _ in range(degLen)], axis=0) # (degLen, batch_size, time_steps, memLen)
    degs = np.zeros_like(xTerms)
    for d in range(degLen):
        degs[d] = d*np.ones_like(xTerms[d])  # (degLen, batch_size, time_steps, memLen)
    xTerms = xTerms ** degs # (degLen, batch_size, time_steps, memLen)
    xTerms_real= xTerms[:]*x_extented[:, 0, :, :] # (degLen, batch_size, time_steps, memLen)
    xTerms_imag= xTerms[:]*x_extented[:, 1, :, :]
    xTerms = np.stack([xTerms_real, xTerms_imag], axis=2) # (degLen, batch_size, 2, time_steps, memLen)
    # (degLen, batch_size, 2, time_steps, memLen)->(batch_size, 2, time_steps, degLen*memLen)->(2, time_steps, batch_size, degLen*memLen)
    xTerms = xTerms.transpose(1, 2, 3, 0, 4).reshape(batch_size, 2, time_steps, -1).transpose(1, 2, 0, 3)     
    return xTerms

def fastPolyPrefixOut(xBatch, xTerms, coefMatBatch, memLen):
    _, time_steps, batch_size, dm = xTerms.shape #(2, time_steps, batch_size, degLen*memLen)
    coefmat = coefMatBatch.transpose(0, 2, 1).reshape(batch_size, -1)
    yBatch = np.zeros_like(xBatch)
    y_tmp = xTerms[:,:]*coefmat
    y_tmp = np.sum(y_tmp, axis=-1).transpose(2, 0, 1)#.reshape(batch_size, 2, time_steps)
    yBatch[:, :, memLen-1:] = y_tmp
    return yBatch

def fastPoly(xBatch, coefMatBatch): #(batch_size, 2, 800)
    degLen=5
    memLen=3
    xLen=xBatch.shape[-1]
    time_steps = xLen+1 - memLen
    batch_size = xBatch.shape[0]
    x_extented = np.zeros((batch_size, 2, time_steps, memLen))
    coefmat = coefMatBatch.transpose(0, 2, 1).reshape(batch_size, -1) #(batch_size, memLen, degLen)->(batch_size, degLen, memLen)->(batch_size, degLen*memLen)
    yBatch= np.zeros_like(xBatch) #(batch_size, 2, 800)
    for t in range(time_steps):
        x_extented[:, :, t] = np.flip(xBatch[:, :, t: t+memLen], axis=-1)
    x_extented_abs = np.power(x_extented, 2) 
    x_extented_abs = np.sqrt(x_extented_abs[:, 0, :, :] + x_extented_abs[:, 1, :, :]) #(batch_size, time_steps, memLen)
    xTerms = np.stack([x_extented_abs for _ in range(degLen)], axis=0) # (degLen, batch_size, time_steps, memLen)
    degs = np.zeros_like(xTerms)
    for d in range(degLen):
        degs[d] = d*np.ones_like(xTerms[d])  # (degLen, batch_size, time_steps, memLen)
    xTerms = xTerms ** degs # (degLen, batch_size, time_steps, memLen)
    xTerms_real= xTerms[:]*x_extented[:, 0, :, :] # (degLen, batch_size, time_steps, memLen)
    xTerms_imag= xTerms[:]*x_extented[:, 1, :, :]
    xTerms = np.stack([xTerms_real, xTerms_imag], axis=2)
    xTerms = xTerms.transpose(1, 2, 3, 0, 4).reshape(batch_size, 2, time_steps, -1).transpose(1, 2, 0, 3) # (degLen, batch_size, time_steps, memLen)->(batch_size, time_steps, degLen*memLen)->(time_steps, batch_size, degLen*memLen)
    y_tmp = xTerms[:,:]*coefmat
    y_tmp = np.sum(y_tmp, axis=-1).transpose(2, 0, 1)#.reshape(batch_size, 2, time_steps)
    yBatch[:, :, memLen-1:] = y_tmp
    return yBatch

def rayleighChannel(xBatch, h):
    yBatch=np.zeros_like(xBatch)
    for batchIdx in range(xBatch.shape[0]):
        x=xBatch[batchIdx]
        for t in range(x.shape[-1]):
            yBatch[batchIdx, t] = h[batchIdx, t]*x[t]
    return yBatch

class myVAE(nn.Module):
    def __init__(self, txReal, txImag, input_dim, latent_dim, batch_size, device='cpu', train=True):
        super(myVAE, self).__init__()
        if train:
            self.txReal=txReal.repeat(batch_size, 1)
            self.txImag=txImag.repeat(batch_size, 1)
        else:
            self.txReal=txReal
            self.txImag=txImag

        self.input_dim =  input_dim # H*W
        self.latent_dim =  latent_dim # Z
        self.device = device
        self.batch_size = batch_size
        
        # input: (N,1,H,W) -> output: (N,hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), #k=5 for 40 samples, k=5 for 800 samples
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            )
        
        # input: (N,hidden_dim) -> output: (N, Z)
        self.mean_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*800, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),

            nn.Linear(in_features=2048, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(in_features=128, out_features=latent_dim),
            )
        
        # input: (N,hidden_dim) -> output: (N, Z)
        self.logvar_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*800, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),

            nn.Linear(in_features=2048, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(in_features=128, out_features=latent_dim),
            )
    
        # input: (N, Z) -> output: (N,1,H,W)
        self.decoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=13, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 2, kernel_size=5, padding=2),
            )
        
    def forward(self, x, targets=None): 
        x = self.encoder(x)
        mu = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        z= self.reparametrize(mu, logvar)

        zReal= torch.cat((z, self.txReal), dim=-1)
        zImag= torch.cat((z, self.txImag), dim=-1)
        z_tx = torch.stack((zReal, zImag)).permute((1,0,2))
        x_hat = self.decoder(z_tx)
        # x_hat_magn=torch.max(torch.abs(x_hat.min()), torch.abs(x_hat.max()))
        # x_hat = x_hat/x_hat_magn
        return x_hat, mu, logvar

    #@staticmethod
    def reparametrize(self, mu, logvar, sigma=None):
        # TODO:
        epsilon=torch.randn(logvar.shape, generator=torch.manual_seed(seed)).to(self.device) # N~(0,1)
        if sigma:
            torch.manual_seed(seed)
            # while not((-sigma<epsilon.detach().cpu().numpy()<=-(sigma-1)).all() or ((sigma-1)<epsilon.detach().cpu().numpy()<=sigma).all()):
            epsilon_num = epsilon.detach().cpu().numpy()
            for bi in range(epsilon_num.shape[0]):
                for i in range(epsilon_num[bi].shape[0]):
                    while not(((-sigma<epsilon_num[bi][i]) and (epsilon_num[bi][i]<=-(sigma-1))) or (((sigma-1)<epsilon_num[bi][i]) and (epsilon_num[bi][i]<=sigma))):
                        epsilon[bi][i] = torch.randn(1).to(self.device)
                        epsilon_num[bi][i] = epsilon[bi][i].detach().cpu().numpy()
            # print("simga:{}, epsilon range:{}~{}".format(sigma, epsilon_num.min(), epsilon_num.max()))
        z = mu+torch.exp(logvar/2)*epsilon
        return z

    def loss_function(self, x_hat, x, mu, logvar):
        KL_loss = torch.mean(-0.5*torch.sum(1+logvar-torch.square(mu)-torch.exp(logvar), axis=1), axis=0)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')/self.batch_size
        loss=reconstruction_loss+KL_loss
        return loss  

class myMemPolyVAE(myVAE):
    def __init__(self, txReal, txImag, input_dim, latent_dim, poly_dim, batch_size, device='cpu', train=True):
        super().__init__(txReal, txImag, input_dim, latent_dim, batch_size, device, train=True)
        if train:
            self.txComp=txReal+1j*txImag
            self.txComp=self.txComp.repeat(batch_size, 1)
            self.tx = torch.stack([self.txComp.real, self.txComp.imag], axis=1)
        else:
            self.txComp=txReal+1j*txImag

        self.txTerms = fastPolyPrefix(self.tx.detach().cpu().numpy())
        
        # First submission version
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, stride=1, padding=2), #k=5 for 40 samples, k=5 for 800 samples
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 16, kernel_size=5, stride=1, padding=2), #k=5 for 40 samples, k=5 for 800 samples
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 16, kernel_size=5, stride=1, padding=2), #k=5 for 40 samples, k=5 for 800 samples
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            # nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2), #k=5 for 40 samples, k=5 for 800 samples
            # nn.BatchNorm1d(32),
            # nn.LeakyReLU(),

            nn.Conv1d(16, 2, kernel_size=5, stride=1, padding=2),
            )
        
        # input: (N,hidden_dim) -> output: (N, Z)
        self.mean_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=960, out_features=64), #sdr->2*800, wifi->2*480
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(in_features=64, out_features=latent_dim),
            )
        
        # input: (N,hidden_dim) -> output: (N, Z)
        self.logvar_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=960, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(in_features=64, out_features=latent_dim),
            )
        
        # input: (N,hidden_dim) -> output: (N, Z)
        self.poly_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=960, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(in_features=64, out_features=poly_dim),
            )
        
        self.decoderForPoly = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=5, padding=2), #k=5 for 40 samples, k=5 for 800 samples
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),

            nn.Conv1d(4, 2, kernel_size=5, padding=2),
            )
        
    def forward(self, x, device='cpu'):
        x_encoder = self.encoder(x)

        poly_coef = self.poly_layer(x_encoder)
        mu = self.mean_layer(x_encoder)
        logvar = self.logvar_layer(x_encoder)
        
        poly_coef=poly_coef.detach().cpu().numpy().reshape(x.shape[0], 3, 5)
        ycoef2 = fastPolyPrefixOut(self.tx.detach().cpu().numpy(), self.txTerms, poly_coef, memLen=3)
        ycoef = torch.tensor(ycoef2).to(device)
        # assert (ycoef1==ycoef2).all()

        h=self.reparametrizeH(mu, logvar)
        h1 = (h[:,0,...]**2+h[:,1,...]**2)[:, None, ...]
        h1=torch.sqrt(h1)
        y=h1*ycoef
        
        x_hat = self.decoderForPoly(y)
        return x_hat, mu, logvar
    
    def reparametrizeH(self, mu, logvar):
        h_real=torch.randn((self.batch_size, 800), generator=torch.manual_seed(seed)).to(self.device)
        h_imag=torch.randn((self.batch_size, 800), generator=torch.manual_seed(seed+1)).to(self.device)
        h_real = mu[:,0].reshape(-1,1) + torch.exp(logvar[:,0]/2).reshape(-1,1)*h_real
        h_imag = mu[:,1].reshape(-1,1) + torch.exp(logvar[:,1]/2).reshape(-1,1)*h_imag
        # h=h_real+1j*h_imag
        h = torch.stack([h_real, h_imag], axis=1)
        
        return h
    
    def loss_function(self, x_hat, x, mu, logvar):
        KL_loss = torch.mean(-0.5*torch.sum(1+logvar-torch.square(mu)-torch.exp(logvar), axis=1), axis=0)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')/self.batch_size
        loss=reconstruction_loss+KL_loss
        return loss 
  

def train(vae, trainLoader, testLoader, valLoader, device):
    #TODO
    n_epochs=1000
    optimizer = optim.Adam(vae.parameters(), lr=5e-4)
    trainLoss_curve=[]
    valLoss_curve=[]
    for epoch in range(n_epochs):
        vae.train()
        train_loss = 0
        val_loss = 0
        loss = None
        for batch_idx, (data, labels) in enumerate(trainLoader):
        #TODO
            data, labels = data.to(device, dtype=torch.float), labels.to(device,  dtype=torch.float)
            optimizer.zero_grad()
            data_hat, mu, logvar = vae.forward(data)
            loss = vae.loss_function(data_hat, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

        for batch_idx, (data, labels) in enumerate(testLoader):
            data, labels = data.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
            data_hat, mu, logvar = vae.forward(data)
            loss = vae.loss_function(data_hat, data, mu, logvar)
            val_loss+=loss.item()

        train_loss = train_loss/len(trainLoader)
        trainLoss_curve.append(train_loss)

        val_loss = val_loss/len(valLoader)
        valLoss_curve.append(val_loss)

        if (epoch+1) % 10 ==0:
            print('Epoch: {} \tTraining Loss: {:.3f}, Validation Loss: {:.3f}'.format(epoch+1, train_loss, val_loss))

    plt.figure(2)
    plt.plot(trainLoss_curve, label = "Train Loss")
    plt.plot(valLoss_curve, label = "Val Loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("VAE Training Loss")

    return trainLoss_curve, valLoss_curve

def vaeTest(vae, harmonics, testLoader, device):
    vae.eval()
    rffs = next(iter(testLoader))[0].to(device, dtype=torch.float) # assume batch size >10
    # print("rffs shape:", rffs.shape)

    rffsPred, mu, logvar = vae.forward(rffs)
    sigma = torch.exp(logvar/2)
    sigma2 = 2*sigma
    sigma3 = 3*sigma

    logvar_ = torch.log(sigma)*2
    logvar2 = torch.log(sigma2)*2
    logvar3 = torch.log(sigma3)*2
    logvars = [logvar, logvar2, logvar3]

    deltas = torch.arange(0, 11, 10)
    rffsPred=[]
    for var in logvars:
        z=vae.reparametrize(mu, var)
        zReal= torch.cat((z, vae.txReal), dim=-1)
        zImag= torch.cat((z, vae.txImag), dim=-1)
        z_tx = torch.stack((zReal, zImag)).permute((1,0,2))

        predTmp=vae.decoder(z_tx)
        predTmp = predTmp[0]
        rffsPred.append(predTmp.cpu().detach().numpy())

    rffsPred=np.array(rffsPred)#.reshape((len(logvars),-1,2,800))
    rffs=rffs.cpu().detach().numpy()    
    fig2, ax2 = plt.subplots() # MSE loss between generated and original signals, one curve for each harmonis, one figure for all harmonics
    for h in range(min(harmonics, 5)):
        rffI=rffs[0, 0, :].reshape(800,)
        rffQ=rffs[0, 1, :].reshape(800,)
        loss=[]
        fig1, ax1 = plt.subplots() #generated & original signals, one figure for each harmonics
        for i in range(len(logvars)):
            rffPredI=rffsPred[i, 0, :].reshape(800,)
            rffPredQ=rffsPred[i, 1, :].reshape(800,)
            ax1.plot(rffPredI[:800], label=f'VAE Generated: {i+1}$\sigma$')
            loss.append(np.sqrt(np.sum((rffI-rffPredI)**2+(rffQ-rffPredQ)**2)))

        ax1.plot(rffI[:800],linestyle=':', label='Original Output of PA')
        ax1.legend()
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Amplitude')
        ax1.set_title("Generated Signals, Harmonic{}".format(h+1))
        ax2.plot(loss, label='loss-var-h'+str(h+1))
    ax2.legend()
    ax2.set_xticks(np.arange(1, 4))
    ax2.set_xlabel(r'$\sigma$')
    ax2.set_ylabel('Test Loss(MSE)')
    ax2.set_title("Test Loss(MSE), Harmonic{}".format(h+1))

def vaeGenerateSigma(vae, harmonics, dataset, preambleReal, premableImag, device):
    vae.eval()
    harmonics = torch.sum(dataset[:, 1:, :, :], axis=1)
    dataset = torch.stack([dataset[:, 0, :, :], harmonics], dim=1)
    dataset = dataset.to(device, dtype=torch.float)

    dataset_vae, mu, logvar = vae.forward(dataset)

    sigma = torch.exp(logvar/2)

    thds_high = [mu+sigma, mu+2*sigma, mu+3*sigma]
    thds_low = [mu-sigma, mu-2*sigma, mu-3*sigma]

    rffsPred=[]

    for t in range(len(thds_high)):
        torch.manual_seed(seed)
        thd_high = thds_high[t]
        thd_low = thds_low[t]

        epsilon=torch.zeros_like(logvar).to(device)
        z=torch.zeros_like(logvar).to(device)
        for i in range(epsilon.shape[0]):
            for j in range(epsilon.shape[1]):
                while True:
                    epsilon[i, j] = torch.randn(1)
                    z[i, j] = mu[i, j]+sigma[i, j]*epsilon[i, j]
                    if thd_low[i, j] <=z[i, j] <= thd_high[i, j]:
                        break

        zReal= torch.cat((z, vae.txReal), dim=-1)
        zImag= torch.cat((z, vae.txImag), dim=-1)
        z_tx = torch.stack((zReal, zImag)).permute((1,0,2))
        z_tx  =z_tx.unsqueeze(1)

        predTmp=vae.decoder(z_tx)
        rffsPred.append(predTmp.cpu())

    return dataset_vae, thds_high, thds_low, rffsPred