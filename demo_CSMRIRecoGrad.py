#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS MRI reco. demo in pytorch 
test on GD, AGD, and CQNPM.
Using Gradient-Driven based denoiser.
D_sigma = x-\nabla g_theta(x) where g_theta(x) is a scalar function, theta is the parmeters in the NN.
@author: hongtao
Date: 28 Nov. 2024
Tao Hong, Zhaoyi Xu, Se Young Chun, Luis Hernandez-Garcia, and Jeffrey A. Fessler, 
``Convergent Complex Quasi-Newton Proximal Methods for Gradient-Driven Denoisers in Compressed Sensing MRI Reconstruction'',
To appear in IEEE Transactions on Computational Imaging, arXiv:2505.04820, 2025.
"""

import torch
import torchkbnufft as tkbn
import numpy as np
import OptAlgMRIGD as opt # package with optimization algorithms
import scipy.io
import os
import json
import modelsutilities as modutl # contain the NN structure
#=========================================================================
# add arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-im_type', dest='im_type',help='Image type: Brain or Knee', type=str,default = 'Brain')
parser.add_argument('-trj_type', dest='trj_type',help='Trj type: Spiral or Radial', type=str,default = 'Spiral')
parser.add_argument('-sigma_min', dest='sigma_min', help='Lower bound of sigma for training denoiser',type=float,default = 1.0)
parser.add_argument('-sigma_max', dest='sigma_max',help='Upper bound of sigma for training denoiser',type=float,default = 1.0)
parser.add_argument('-modelIterName', dest='modelIterName', help='Name for the chosen trained NN',type=str,default = 'PotentialDenoiser_iter17499sigma')
parser.add_argument('-im_index', dest='im_index', help='Choose which image for testing',type=str,default = '1')
parser.add_argument('-isCoilCompress', dest='isCoilCompress', help='True, we use the coil compression',type=bool,default = True)
parser.add_argument('-compression_coils', dest='compression_coils',help='the number of coils after coil compression',type=int,default = 12)
parser.add_argument('-MaxIter', dest='MaxIter', help='Maximal number of iterations',type=int,default = 10)
parser.add_argument('-MaxIterGD', dest='MaxIterGD', help='Maximal number of iterations for GD and AGD',type=int,default = 10)
parser.add_argument('-MaxInnerIter', dest='MaxInnerIter', help='Maximal number of iteration for the inner solver',type=int,default = 15)
parser.add_argument('-noise_level', dest='noise_level', help='Specify the noise level',type=float,default = 1e-4)
parser.add_argument('-lamda', dest='lamda',help='trade-off parameter', type=float,default = 1e-4)    
parser.add_argument('-isIni', dest='isIni',help='isIni: true we use the DCT as the initialization otherwise use A^HKspace', type=bool,default = False)
parser.add_argument('-isCPU', dest='isCPU',help='isCPU: true we use the CPU otherwise we use GPU', type=bool,default = False)
parser.add_argument('-cuda_index', dest='cuda_index',help='Choose the used GPU index', type=str,default = '0')
parser.add_argument('-ub', dest='ub', type=float,help='upper bound for each component, if ub=torch.inf, we assume no bound',default = 1)# np.inf)#
parser.add_argument('-ScaleFac', dest='ScaleFac', type=float,help='scale factor for the image that the maximal magnitude is ScaleFac',default = 1)# np.inf)#
parser.add_argument('-numLayers ', dest='numLayers', help='the number of layers of the NN',type=int,default = 6)
parser.add_argument('-gamma_QNP', dest='gamma_QNP', help='gamma parameter to scale the diagonal matrix in QNP if use SR1 estimation',type=float,default = 1.6)
parser.add_argument('-gamma_QN', dest='gamma_QN', help='gamma parameter to scale the diagonal matrix in QN if use SR1 estimation',type=float,default = 1.6)
parser.add_argument('-Hessian_Type', dest='Hessian_Type', help='Specify the strategy to estimate the Hessian matrix, SR1 or Modified-SR1',type=str,default = 'Modified-SR1')
parser.add_argument('-beta', dest='beta', help='ratio to reduce the stepsize',type=float,default = 1.2)
parser.add_argument('-alphaGD', dest='alphaGD', help='stepsize for GD method',type=float,default = 0.5) # brain: 0.1
parser.add_argument('-alphaISTA', dest='alphaISTA',help='stepsize for ISTA', type=float,default = 1)  #2 brain: 1
parser.add_argument('-alphaFISTA_LL', dest='alphaFISTA_LL', help='stepsize for FISTA_LL',type=float,default = 1)
parser.add_argument('-a_kQN', dest='a_kQN', help='stepsize for QN',type=float,default = 1)
parser.add_argument('-a_kQNP', dest='a_kQNP', help='stepsize for QNP',type=float,default = 1)
parser.add_argument('-isLinearSearchGD', dest='isLinearSearchGD',help='True: use line search for GD method', type=bool,default = True) #False
parser.add_argument('-isLinearSearchISTA', dest='isLinearSearchISTA', help='True: use line search for ISTA',type=bool,default = True)
parser.add_argument('-isLinearSearchQNP', dest='isLinearSearchQNP',help='True: use line search for QNP',type=bool, default = False)
parser.add_argument('-isLinearSearchQN', dest='isLinearSearchQN',help='True: use line search for QN',type=bool, default = False)
parser.add_argument('-verbose', dest='verbose',help='True: print the output information', type=bool,default = True)
parser.add_argument('-isSave', dest='isSave',type=bool,help='True: save the iter at each iteration', default = False)
parser.add_argument('-iSmkdir', dest='iSmkdir', help='True: create the folder', type=bool,default = True)
args = parser.parse_args()

cuda_index = args.cuda_index
trj_type = args.trj_type            # 'Radial'#'Spiral' #
im_type = args.im_type              # 'Knee' # 
im_index = args.im_index            # choose which test image
sigma_min = args.sigma_min
sigma_max = args.sigma_max
MaxIter = args.MaxIter              # maximal number of iterations to recover the image
MaxIterGD = args.MaxIterGD

if trj_type == 'Spiral':
    noise_level = args.noise_level  # input SNR 17dB the additive noise level for the measurements
elif trj_type == 'Radial':
    noise_level = args.noise_level
elif trj_type == 'Cartesian':
    noise_level = args.noise_level

gamma_QNP = args.gamma_QNP
gamma_QN = args.gamma_QN
Hessian_Type = args.Hessian_Type    #'SR1' # 'SR1' # 'Modified-SR1' #
MaxInnerIter = args.MaxInnerIter    # the number of iteration for CG
lamda = args.lamda                  # parameter for RED

beta = args.beta
alphaGD = args.alphaGD
isLinearSearchGD = args.isLinearSearchGD
alphaISTA = args.alphaISTA
isLinearSearchISTA = args.isLinearSearchISTA
alphaFISTA_LL = args.alphaFISTA_LL
isLinearSearchQNP = args.isLinearSearchQNP
a_kQN = args.a_kQN
isLinearSearchQN = args.isLinearSearchQN
a_kQNP = args.a_kQNP
L_gtheta_QNP = 1/args.alphaISTA  
ub = args.ub
verbose = args.verbose
isSave = args.isSave
iSmkdir = args.iSmkdir
numLayers = args.numLayers
#--------------------------------------------------------------------
if args.isCPU:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:'+cuda_index)

# Instantiate the model
model = modutl.Network(num_blocks=numLayers)
ModelNameindex = im_type+args.modelIterName+ str(sigma_min) + str(sigma_max)  + '.pth'
ModelPath = '/workspace/taohong/Grad/'+ModelNameindex
model.load_state_dict(torch.load(ModelPath,weights_only=True))
model = model.to(device)
model.eval()

# pick the test image
if im_type == 'Brain':
    filename = 'BrainDeepGT_' + im_index + '.mat'
    data = scipy.io.loadmat('./DeepBrain/' + filename)
elif im_type == 'Knee':
    filename = 'KneeGT_' + im_index + '.mat'
    data = scipy.io.loadmat('./fastMRIKnee/' + filename)
    
# choose image
im_real = data['im_real']
im_imag = data['im_imag']
im = im_real+1j*im_imag

im = im/np.max(np.abs(im))*args.ScaleFac
im_original = im

# save the GT image.
if trj_type == 'Spiral':
    if im_type == 'Brain':
        folderName = '/YourOwnPath/Spiral/Brain'+im_index
    elif im_type == 'Knee':
        folderName = '/YourOwnPath/Spiral/Knee'+im_index
    if iSmkdir:
        if not os.path.exists(folderName):
            os.mkdir(folderName)
    trj_file = "MRI/data/spiral_brain/trj.npy" # trajectory
    mps_file = "MRI/data/mpsSim32.npy" # sensitivity maps
    mps = np.load(mps_file)
    trj = np.load(trj_file)
    trj = trj[::6,:,:]
elif trj_type == 'Radial':
    if im_type == 'Brain':
        folderName = '/YourOwnPath/Radial/DeepBrain'+im_index
    elif im_type == 'Knee':
        folderName = '/YourOwnPath/Radial/Knee'+im_index
    if iSmkdir:
        if not os.path.exists(folderName):
            os.mkdir(folderName)
    mps_file = "./MRI/data/mpsSim32.npy" # sensitivity maps
    mps = np.load(mps_file)
    nspokes = 55
    spokelength = 1024
    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(nspokes,spokelength,1))
    ky = np.zeros(shape=(nspokes,spokelength,1))
    ky[0,:,0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[i,:,0] = np.cos(ga) * kx[i - 1,:,0] - np.sin(ga) * ky[i - 1,:,0]
        ky[i,:,0] = np.sin(ga) * kx[i - 1,:,0] + np.cos(ga) * ky[i - 1,:,0]
    trj = np.concatenate((kx,ky),axis=2)
elif trj_type == 'Cartesian':
    if im_type == 'Brain':
        folderName = '/YourOwnPath/Cartesian/DeepBrain'+im_index
    elif im_type == 'Knee':
        folderName = '/YourOwnPath/Cartesian/Knee'+im_index
    if iSmkdir:
        if not os.path.exists(folderName):
            os.mkdir(folderName)
    mask = scipy.io.loadmat('./MRI/data/Linemask0.1_256cent15.mat')
    mask = mask['mask']
    mask = np.float32(mask)
    mps_file = "./MRI/data/mpsSim32.npy" 
    mps = np.load(mps_file)
    smaps = torch.from_numpy(mps).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

if trj_type != 'Cartesian':
    Dict = {'im_original':im_original,'Trj':trj}
    scipy.io.savemat(folderName+'/GT_Trj.mat',Dict)
else:
    Dict = {'im_original':im_original,'mask':mask.cpu().numpy()}
    scipy.io.savemat(folderName+'/GT_Trj.mat',Dict)

im_size = im.shape
im = torch.tensor(im).unsqueeze(0).unsqueeze(0).to(torch.complex64)
im = im.to(device)

if  trj_type != 'Cartesian':
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
else:
    # Cartesian sampling
    def Ax_operator(x,smaps,mask):
        _,size_coil,_,_ = smaps.shape
        y = torch.zeros_like(smaps)
        x_temp = smaps*x
        for iter in range(size_coil):
            y[0,iter,:,:] = mask*torch.fft.fftshift(torch.fft.fft2(x_temp[0,iter,:,:],norm='ortho'))
        return y
    def ATx_operator(y,smaps,mask):
        _,size_coil,_,_ = smaps.shape
        y_temp = mask*y
        x_temp = torch.zeros_like(smaps)
        for iter in range(size_coil):
            x_temp[0,iter,:,:] = smaps[0,iter,:,:].conj()*torch.fft.ifft2(torch.fft.ifftshift(y_temp[0,iter,:,:]),norm='ortho')
        x = torch.sum(x_temp,1).unsqueeze(1)
        return x
    Ax = lambda x: Ax_operator(x,smaps,mask)
    ATx = lambda x: ATx_operator(x,smaps,mask)

# normalize the forward model
L = opt.Power_Iter(Ax,ATx,im_size,tol = 1e-6,device=device)
L_sr = torch.sqrt(L)
if  trj_type != 'Cartesian':
    Ax = lambda x: nufft_ob(x, ktraj, smaps=smaps.to(x))/L_sr
    ATx = lambda x: adjnufft_ob(x,ktraj,smaps=smaps.to(x))/L_sr
else:
    Ax = lambda x: Ax_operator(x,smaps,mask)/L_sr
    ATx = lambda x: ATx_operator(x,smaps,mask)/L_sr

# formulate the measurements
b = Ax(im)

torch.manual_seed(2)
noise_real = torch.randn(b.shape).to(device)
torch.manual_seed(5)
noise_imag = torch.randn(b.shape).to(device)
b_noise  = b+noise_level*(noise_real+1j*noise_imag)


snr = 10*torch.log10(torch.norm(b)/torch.norm(b_noise-b))
print('The measurements SNR is {0}'.format(snr.cpu().numpy()))

if args.isCoilCompress:
    if  trj_type != 'Cartesian':
        kspace_com,maps_comp=opt.Coil_Compression(b_noise,smaps,args.compression_coils)
        b_noise = kspace_com
        # define the forward model
        Ax = lambda x: nufft_ob(x, ktraj, smaps=maps_comp.to(x))/L_sr
        ATx = lambda x: adjnufft_ob(x,ktraj,smaps=maps_comp.to(x))/L_sr
    else:
        kspace_com,maps_comp=opt.Coil_Compression(b_noise,smaps,args.compression_coils,True)
        b_noise = kspace_com
        Ax = lambda x: Ax_operator(x,maps_comp,mask)/L_sr
        ATx = lambda x: ATx_operator(x,maps_comp,mask)/L_sr

if args.isIni:
    # density compensation reco
    # dcf = (coord[..., 0]**2 + coord[..., 1]**2)**0.5
    if  trj_type != 'Cartesian':
        dcomp = tkbn.calc_density_compensation_function(ktraj, im_size)
        x_density = ATx(dcomp*b_noise)
        x_ini = x_density/torch.max(torch.abs(x_density))
        x_ini = None
    else:
        x_ini = None
else:
    x_ini = None

with open(folderName+'/argsMRIIm' + im_index + im_type + trj_type +'.json', 'w') as f:
    json.dump(vars(args), f)

algName = '/GD_GD'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_RED_Grad,psnr_set_RED_Grad,ssim_set_RED_Grad,CPUTime_set_RED_Grad,Cost_set_RED_Grad,iter_diff_RED_Grad,iter_diff_RED_Grad_norm = \
    opt.GDEnergyGrad(MaxIterGD,Ax,ATx,b_noise,ub=ub,x_ini=x_ini,\
                    g_theta=model,scale=args.ScaleFac,lamda=lamda,alpha=alphaGD,beta = beta,isLinearSearch = isLinearSearchGD,\
                        save=loc,original=im_original,SaveIter=isSave,verbose = verbose,device=device)


algName = '/GD_ISTA'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_RED_ISTA,psnr_set_RED_ISTA,ssim_set_RED_ISTA,CPUTime_set_RED_ISTA,Cost_set_RED_ISTA,iter_diff_RED_ISTA,iter_diff_RED_ISTA_norm = \
    opt.ISTAEnergyGrad(MaxIter,Ax,ATx,b_noise,ub=ub,x_ini=x_ini,\
                      g_theta=model,scale=args.ScaleFac,lamda=lamda,alpha=alphaISTA,beta=beta,isLinearSearch = isLinearSearchISTA,\
                        MaxInnerIter = MaxInnerIter,save=loc,original=im_original,SaveIter=isSave,verbose = verbose,device=device)

algName = '/GD_FISTA_LL'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_RED_FISTA_LL,psnr_set_RED_FISTA_LL,ssim_set_RED_FISTA_LL,CPUTime_set_RED_FISTA_LL,Cost_set_RED_FISTA_LL,iter_diff_RED_FISTA_LL,iter_diff_RED_FISTA_LL_norm = \
    opt.AccISTAEnergyGrad_LL(MaxIter,Ax,ATx,b_noise,ub=ub,x_ini=x_ini,\
                      g_theta=model,scale=args.ScaleFac,lamda=lamda,alpha=alphaFISTA_LL,beta=beta,\
                        MaxInnerIter = MaxInnerIter,save=loc,original=im_original,SaveIter=isSave,verbose = verbose,device=device)


algName = '/GD_QNP'
loc = folderName+algName
if iSmkdir:
    if not os.path.exists(loc):
        os.mkdir(loc)
x_RED_QNP,psnr_set_RED_QNP,ssim_set_RED_QNP,CPUTime_set_RED_QNP,Cost_set_RED_QNP,iter_diff_RED_QNP,iter_diff_RED_QNP_norm = \
    opt.QNPEnergyGrad(MaxIter,Ax,ATx,b_noise,ub=ub,x_ini=x_ini,g_theta = model,scale=args.ScaleFac,L=L_gtheta_QNP,lamda=lamda,beta=beta,a_k=a_kQNP,isLinearSearch = isLinearSearchQNP,\
                     Hessian_Type = Hessian_Type,gamma=gamma_QNP, MaxInnerIter = MaxInnerIter,save=loc,original=im_original,SaveIter=isSave,\
                        verbose = verbose,device=device)
