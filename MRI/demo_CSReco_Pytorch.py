#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reco. demo in pytorch 
test on ISTA-RED, (pre)ISTA-PnP, (pre)FISTA-PnP, and QNP-PnP.
Using Normalization-equivanent DRUNET as the denoiser.
@author: hongtao
"""
import torch
import torchkbnufft as tkbn
import numpy as np
import optalg_Tao_Pytorch as opt # package for optimization algorithms
import scipy.io
import matplotlib.pyplot as plt
import os
from models.network_dncnn import *
from models.fdncnn import *
from models.drunet import *
#=========================================================================
# add arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-im_type', dest='im_type', type=str,default = 'Brain')
parser.add_argument('-trj_type', dest='trj_type', type=str,default = 'Spiral')
parser.add_argument('-sigma_noise', dest='sigma_noise', type=float,default = 0.1)
parser.add_argument('-cuda_index', dest='cuda_index', type=str,default = '0')
parser.add_argument('-im_index', dest='im_index', type=str,default = '1')
parser.add_argument('-MaxIter', dest='MaxIter', type=int,default = 200)
parser.add_argument('-noise_level', dest='noise_level', type=float,default = 1e-3)
parser.add_argument('-gamma_QNP_PnP', dest='gamma_QNP_PnP', type=float,default = 1.7)
parser.add_argument('-Hessian_Type', dest='Hessian_Type', type=str,default = 'Modified-SR1')
parser.add_argument('-MaxCG_Iter', dest='MaxCG_Iter', type=int,default = 20)
parser.add_argument('-lamda', dest='lamda', type=float,default = 0.01)
parser.add_argument('-mu_corr', dest='mu_corr', type=float,default = 0)
parser.add_argument('-NetworkType', dest='NetworkType', type=str,default = 'DRUNet')
parser.add_argument('-model_type', dest='model_type', type=str,default = 'norm-equiv')
parser.add_argument('-verbose', dest='verbose', default = True)
parser.add_argument('-isSave', dest='isSave', default = False)
parser.add_argument('-iSmkdir', dest='iSmkdir', default = True)
parser.add_argument('-isPlot', dest='isPlot', default = False)
args = parser.parse_args()
#=========================================================================
# Radial - noise 6e-4
# Spiral - noise 1e-3 
# input SNR ~ 14.5dB 
#=========================================================================
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
#     )
cuda_index = args.cuda_index
trj_type = args.trj_type #'Radial'#'Spiral' #
im_type = args.im_type #'Knee' # 
im_index = args.im_index# choose which test image
sigma_noise = args.sigma_noise # the noise level for trainning the denoiser
MaxIter = args.MaxIter # maximal number of iterations to recover the image
if trj_type == 'Spiral':
    noise_level = args.noise_level  # input SNR 17dB the additive noise level for the measurements
elif trj_type == 'Radial':
    noise_level = args.noise_level
gamma_QNP_PnP = args.gamma_QNP_PnP
Hessian_Type = args.Hessian_Type #'SR1'# #'SR1' #'Modified-SR1' #
MaxCG_Iter = args.MaxCG_Iter # the number of iteration for CG
lamda = args.lamda    # parameter for RED
mu_corr = args.mu_corr     # 1e-3
NetworkType = args.NetworkType #'DnCNN' #'FDnCNN'
model_type = args.model_type#'scale-equiv' #'ordinary' #
verbose = args.verbose
isSave = args.isSave # whether save the reconstructed images
iSmkdir = args.iSmkdir
isPlot = args.isPlot
#------------------------------------------------------------------
device = torch.device('cuda:'+cuda_index)
# load the denoising network
savedmodelfolder = '/faststorage/tahong/TrainModel/'   
if im_type == 'Brain':
    modelName = 'norm-equivMSECh1_20240320_SE_DRUCNNBrainDenoise_sigma{}.pth'.format(sigma_noise)
elif im_type == 'Knee':
    modelName = 'norm-equivMSECh1_20240320_SE_DRUCNNKneeDenoise_sigma{}.pth'.format(sigma_noise)

if NetworkType == 'DnCNN':
    model = DnCNN(in_nc=1, out_nc=1).to(device)
elif NetworkType == 'FDnCNN':
    model = FDnCNN(blind=True,mode = model_type,in_nc=1, out_nc=1).to(device)
elif NetworkType == 'DRUNet':
    model = DRUNet(blind=True,mode = model_type,in_nc=1, out_nc=1).to(device)   

scale = 1/255
model_path = (savedmodelfolder+modelName)
model.load_state_dict(torch.load(model_path))
model.eval()

# pick the test image
if im_type == 'Brain':
    filename = 'BrainDeepGT_' + im_index + '.mat'
    data = scipy.io.loadmat('/export/data/tahong/MRIData/DeepBrainChosen/' + filename)
elif im_type == 'Knee':
    filename = 'KneeGT_' + im_index + '.mat'
    data = scipy.io.loadmat('/export/data/tahong/MRIData/fastMRIKneeChosen/' + filename)

# choose image
im_real = data['im_real']
im_imag = data['im_imag']
im = im_real+1j*im_imag

im = im/np.max(np.abs(im))
im_original = im

# save the GT image.
if trj_type == 'Spiral':
    if im_type == 'Brain':
        folderName = '/faststorage/tahong/QNPPnP/results/Spiral/DeepBrain'+im_index
    elif im_type == 'Knee':
        folderName = '/faststorage/tahong/QNPPnP/results/Spiral/Knee'+im_index
    if iSmkdir:
        if not os.path.exists(folderName):
            os.mkdir(folderName)
    trj_file = "data/spiral_brain/trj.npy" # trajectory
    mps_file = "data/spiral_brain/mpsSim32.npy" # sensitivity maps
    mps = np.load(mps_file)
    trj = np.load(trj_file)
    trj = trj[0:-1:6,:,:]
elif trj_type == 'Radial':
    if im_type == 'Brain':
        folderName = '/faststorage/tahong/QNPPnP/results/Radial/DeepBrain'+im_index
    elif im_type == 'Knee':
        folderName = '/faststorage/tahong/QNPPnP/results/Radial/Knee'+im_index
    if iSmkdir:
        if not os.path.exists(folderName):
            os.mkdir(folderName)
    #trj_file = "data/radial/trj.npy" # trajectory
    mps_file = "data/radial/mpsSim32.npy" # sensitivity maps
    mps = np.load(mps_file)
    #trj = np.load(trj_file)
    #trj = trj[0:-1:2,:,:]
    #1 1 2 3 5 8 13 21 34 55 89
    nspokes = 34
    spokelength = 1024
    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(nspokes,spokelength,1))
    ky = np.zeros(shape=(nspokes,spokelength,1))
    ky[0,:,0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[i,:,0] = np.cos(ga) * kx[i - 1,:,0] - np.sin(ga) * ky[i - 1,:,0]
        ky[i,:,0] = np.sin(ga) * kx[i - 1,:,0] + np.cos(ga) * ky[i - 1,:,0]
    trj = np.concatenate((kx,ky),axis=2)
#os.mkdir(folderName)
np.save(folderName + '/Real.npy', np.real(im_original))
np.save(folderName + '/Imag.npy', np.imag(im_original))
np.save(folderName + '/Trj.npy',trj)

im_size = im.shape
im = torch.tensor(im).unsqueeze(0).unsqueeze(0).to(torch.complex64)
im = im.to(device)

# trj saved in [(batch) dim NumPoints]
trj_reshape = trj.reshape(1,trj.shape[0]*trj.shape[1],trj.shape[2])
ktraj = torch.tensor(trj_reshape, dtype=torch.float).permute(0,2,1)
ktraj = ktraj.to(device)
# define the NuFFT operator
nufft_ob = tkbn.KbNufft(im_size=im_size,device=device)
adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size,device=device)
# set the sensitivity mapping
smaps = torch.tensor(mps).unsqueeze(0) 
smaps = smaps.to(device)

# define the forward model
Ax = lambda x: nufft_ob(x, ktraj, smaps=smaps.to(x))
ATx = lambda x: adjnufft_ob(x,ktraj,smaps=smaps.to(x))
# normalize the forward model
L = opt.Power_Iter(Ax,ATx,im_size,tol = 1e-6,device=device)
L_sr = torch.sqrt(L)
Ax = lambda x: nufft_ob(x, ktraj, smaps=smaps.to(x))/L_sr
ATx = lambda x: adjnufft_ob(x,ktraj,smaps=smaps.to(x))/L_sr
# formulate the measurements
b = Ax(im)
b_m,b_n,b_k = b.shape
torch.manual_seed(2)
noise_real = torch.randn(b_m,b_n,b_k).to(device)
torch.manual_seed(5)
noise_imag = torch.randn(b_m,b_n,b_k).to(device)
b_noise  = b+noise_level*(noise_real+1j*noise_imag)
snr = 10*torch.log10(torch.norm(b)/torch.norm(b_noise-b))
print('The measurements SNR is {0}'.format(snr.cpu().numpy()))

# density compensation reco
#dcf = (coord[..., 0]**2 + coord[..., 1]**2)**0.5
#dcomp = tkbn.calc_density_compensation_function(ktraj, im_size)
#x_density = ATx(dcomp*b_noise)

'''
algName = '/RED_ISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_RED_ISTA,psnr_set_RED_ISTA,ssim_set_RED_ISTA,CPUTime_set_RED_ISTA = \
opt.ISTA_RED(MaxIter,Ax,ATx,b_noise,denoiser = model,scale=scale,\
MaxCG_Iter=MaxCG_Iter,lamda=lamda,\
verbose = verbose,save=loc,original=im_original,SaveIter=isSave,device=device)
'''


'''
algName = '/PnP_FISTA_RESTART'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_FISTARESTART,psnr_set_PnP_FISTARESTART,ssim_set_PnP_FISTARESTART,CPUTime_set_PnP_FISTARESTART = \
opt.FISTA_PnP(MaxIter,Ax,ATx,b_noise,isRestart=True,ReStartIter=Restart_Iter,denoiser = model,mu_corr=mu_corr,\
scale=scale,save=loc,isPred=False,original=im_original,L=1,SaveIter=isSave,\
verbose = verbose,device=device)
'''


algName = '/PnP_ISTA_34Spok'#'/PnP_ISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_ISTA,psnr_set_PnP_ISTA,ssim_set_PnP_ISTA,CPUTime_set_PnP_ISTA,fixed_PnP_ISTA = \
opt.ISTA_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,mu_corr=mu_corr,\
scale=scale,save=loc,isPred=False,original=im_original,SaveIter=isSave,verbose = verbose,device=device)

w_pred = lambda x: 2*x-ATx(Ax(x))
#w_pred = lambda x: 8*x-ATx(Ax(8*x))

'''
algName = '/Pred1_PnP_FISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_FISTA_Pre1,psnr_set_PnP_FISTA_Pre1,ssim_set_PnP_FISTA_Pre1,CPUTime_set_PnP_FISTA_Pre1,fixed_PnP_FISTA_Pre1  = \
opt.FISTA_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,scale=scale,\
save=loc,isPred=True,w_pred=w_pred,original=im_original,SaveIter=isSave,\
verbose = verbose,device=device)

algName = '/RED_FISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_RED_FISTA,psnr_set_RED_FISTA,ssim_set_RED_FISTA,CPUTime_set_RED_FISTA = \
opt.FISTA_RED(MaxIter,Ax,ATx,b_noise,denoiser = model,scale=scale,\
MaxCG_Iter=MaxCG_Iter,lamda=lamda,\
verbose = verbose,save=loc,original=im_original,SaveIter=isSave,device=device)    

algName = '/PnP_FISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_FISTA,psnr_set_PnP_FISTA,ssim_set_PnP_FISTA,CPUTime_set_PnP_FISTA,fixed_PnP_FISTA = \
opt.FISTA_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,mu_corr=mu_corr,\
scale=scale,save=loc,isPred=False,original=im_original,SaveIter=isSave,\
verbose = verbose,device=device)
'''

algName = '/Pre1_PnP_ISTA_34Spok'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_ISTA_Pre1,psnr_set_PnP_ISTA_Pre1,ssim_set_PnP_ISTA_Pre1,CPUTime_set_PnP_ISTA_Pre1, fixed_PnP_ISTA_Pre1= \
opt.ISTA_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,mu_corr=mu_corr,\
scale=scale,save=loc,isPred=True,\
w_pred=w_pred,original=im_original,SaveIter=isSave,verbose = verbose,device=device)


w_pred = lambda x: 4*x-ATx(Ax((10/3)*x))
'''
algName = '/Pred2_PnP_FISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_FISTA_Pre2,psnr_set_PnP_FISTA_Pre2,ssim_set_PnP_FISTA_Pre2,\
CPUTime_set_PnP_FISTA_Pre2,fixed_PnP_FISTA_Pre2  = \
opt.FISTA_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,scale=scale,\
save=loc,isPred=True,w_pred=w_pred,original=im_original,SaveIter=isSave,\
verbose = verbose,device=device)
'''

algName = '/Pre2_PnP_ISTA_34Spok'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_ISTA_Pre2,psnr_set_PnP_ISTA_Pre2,ssim_set_PnP_ISTA_Pre2,\
CPUTime_set_PnP_ISTA_Pre2,fixed_PnP_ISTA_Pre2 = \
opt.ISTA_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,mu_corr=mu_corr,\
scale=scale,save=loc,isPred=True,\
w_pred=w_pred,original=im_original,SaveIter=isSave,verbose = verbose,device=device)

algName = '/PnP_QNP_34Spok'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_QNP,psnr_set_PnP_QNP,ssim_set_PnP_QNP,CPUTime_set_PnP_QNP,fixed_PnP_QNP,gradnorm_PnP_QNP = \
opt.QNP_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,scale=scale,\
Hessian_Type=Hessian_Type,gamma=gamma_QNP_PnP,verbose = verbose,\
save=loc,original=im_original,SaveIter=isSave,device=device)

algName = '/ADMM_34Spok'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_PnP_ADMM,psnr_set_PnP_ADMM,ssim_set_PnP_ADMM,CPUTime_set_PnP_ADMM = \
opt.ADMM_PnP(MaxIter,Ax,ATx,b_noise,denoiser = model,mu_corr=mu_corr,\
scale=scale,sigma=1,eta=1,MaxCG_Iter=MaxCG_Iter,save=loc,\
original=im_original,SaveIter=isSave,verbose = verbose,device=device)


# -------------------------------------------------------
# plot results
# -------------------------------------------------------
if isPlot:
    plt.plot(psnr_set_PnP_ISTA)
    plt.plot(psnr_set_PnP_ISTA_Pre1)
    plt.plot(psnr_set_PnP_ISTA_Pre2)
    #plt.plot(psnr_set_PnP_FISTA)
    #plt.plot(psnr_set_PnP_FISTA_Pre1)
    #plt.plot(psnr_set_PnP_FISTA_Pre2)
    plt.legend(['PnP-ISTA','PnP-ISTAPre1','PnP-ISTAPre2'])#,'PnP-FISTA','PnP-FISTAPre1','PnP-FISTAPre2'])
    plt.grid()
    plt.title(trj_type+im_type)
    #plt.savefig(trj_type+im_type+'PredPSNR.pdf', format='pdf')
    plt.show()

    plt.plot(CPUTime_set_PnP_ISTA,psnr_set_PnP_ISTA)
    plt.plot(CPUTime_set_PnP_ISTA_Pre1,psnr_set_PnP_ISTA_Pre1)
    plt.plot(CPUTime_set_PnP_ISTA_Pre2,psnr_set_PnP_ISTA_Pre2)
    #plt.plot(CPUTime_set_PnP_FISTA,psnr_set_PnP_FISTA)
    #plt.plot(CPUTime_set_PnP_FISTA_Pre1,psnr_set_PnP_FISTA_Pre1)
    #plt.plot(CPUTime_set_PnP_FISTA_Pre2,psnr_set_PnP_FISTA_Pre2)
    plt.legend(['PnP-ISTA','PnP-ISTAPre1','PnP-ISTAPre2'])#,'PnP-FISTA','PnP-FISTAPre1','PnP-FISTAPre2'])
    plt.grid()
    plt.title(trj_type+im_type)
    #plt.savefig(trj_type+im_type+'CPUPredPSNR.pdf', format='pdf')
    plt.show()

    plt.plot(psnr_set_PnP_ISTA)
    plt.plot(psnr_set_PnP_ADMM)
    #plt.plot(psnr_set_PnP_FISTA)
    #plt.plot(psnr_set_RED_ISTA)
    #plt.plot(psnr_set_RED_FISTA)
    plt.plot(psnr_set_PnP_QNP)
    plt.legend(['PnP-ISTA','PnP-ADMM','PnP-QNP'])#,'RED-QNP','PnP-FISTA','RED-FISTA','RED-ISTA',
    plt.grid()
    plt.title(trj_type+im_type)
    #plt.savefig(trj_type+im_type+'PSNR.pdf', format='pdf')
    plt.show()

    plt.plot(CPUTime_set_PnP_ISTA,psnr_set_PnP_ISTA)
    plt.plot(CPUTime_set_PnP_ADMM,psnr_set_PnP_ADMM)
    #plt.plot(CPUTime_set_PnP_FISTA,psnr_set_PnP_FISTA)
    #plt.plot(CPUTime_set_RED_FISTA,psnr_set_RED_FISTA)
    plt.plot(CPUTime_set_PnP_QNP,psnr_set_PnP_QNP)
    plt.legend(['PnP-ISTA','PnP-ADMM','PnP-QNP'])#'PnP-FISTA','RED-FISTA',
    plt.title(trj_type+im_type)
    plt.grid()
    #plt.savefig(trj_type+im_type+'CPUPSNR.pdf', format='pdf')
    plt.show()

    plt.plot(CPUTime_set_PnP_ISTA,psnr_set_PnP_ISTA)
    plt.plot(CPUTime_set_PnP_ISTA_Pre1,psnr_set_PnP_ISTA_Pre1)
    plt.plot(CPUTime_set_PnP_ISTA_Pre2,psnr_set_PnP_ISTA_Pre2)
    plt.plot(CPUTime_set_PnP_QNP,psnr_set_PnP_QNP)
    plt.legend(['PnP-ISTA','PnP-ISTAPre1','PnP-ISTAPre2','PnP-QNP'])
    plt.title(trj_type+im_type)
    plt.grid()
    #plt.savefig(trj_type+im_type+'NoREDADMMCPUPSNR.pdf', format='pdf')
    plt.show()

    plt.plot(psnr_set_PnP_ISTA)
    #plt.plot(psnr_set_RED_ISTA)
    plt.plot(psnr_set_PnP_QNP)
    plt.legend(['PnP-ISTA','PnP-QNP'])#RED-FISTA''RED-ISTA',
    plt.title(trj_type+im_type)
    plt.grid()
    #plt.savefig(trj_type+im_type+'NoREDADMMCPUPSNR.pdf', format='pdf')
    plt.show()

    plt.plot(CPUTime_set_PnP_ISTA,psnr_set_PnP_ISTA)
    #plt.plot(CPUTime_set_RED_ISTA,psnr_set_RED_ISTA)
    plt.plot(CPUTime_set_PnP_QNP,psnr_set_PnP_QNP)
    plt.legend(['PnP-ISTA','PnP-QNP'])#RED-FISTA''RED-ISTA',
    plt.title(trj_type+im_type)
    plt.grid()
    #plt.savefig(trj_type+im_type+'NoREDADMMCPUPSNR.pdf', format='pdf')
    plt.show()

'''
plt.plot(psnr_set_PnP_FISTA[0:50])
plt.plot(psnr_set_PnP_FISTA_Pre1[0:50])
plt.plot(psnr_set_PnP_FISTA_Pre2[0:50])
plt.plot(psnr_set_PnP_QNP[0:50])
plt.legend(['PnP-FISTA','PnP-FISTA-Pre1','PnP-FISTA-Pre2','PnP-QNP'])#RED-FISTA'
plt.title(trj_type+im_type)
plt.grid()
#plt.savefig(trj_type+im_type+'NoREDADMMCPUPSNR.pdf', format='pdf')
plt.show()
'''
