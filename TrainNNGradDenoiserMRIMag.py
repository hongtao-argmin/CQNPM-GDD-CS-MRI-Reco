#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hongtao
1. train as in Cohen's paper
2. for MRI data denoiser
"""
import torch
import numpy as np
import scipy.io
import h5py
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import modelsutilities as modutl
from torch.utils.tensorboard import SummaryWriter
from os import listdir
from os.path import isfile
import OptAlgMRIGD  as opt # package for optimization algorithms
writer = SummaryWriter('runs/Denoiser_Loss')
#=========================================================================
# add arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-LossType', dest='LossType', type=str,default = 'MSE')
parser.add_argument('-numLayers', dest='numLayers', type=int,default = 6)
parser.add_argument('-MaxIter', dest='MaxIter', type=int,default = 50000)
parser.add_argument('-sigma_min', type=float, dest='sigma_min', help="Model is trained for all noise levels between sigma_min and sigma_max.", default=15/255.0)
parser.add_argument('-sigma_max', type=float, dest='sigma_max', help="Model is trained for all noise levels between sigma_min aand sigma_max.", default=15/255.0)
parser.add_argument('-cuda_index', dest='cuda_index', type=str,default = '1')
parser.add_argument('-verbose', dest='verbose', default = True)
parser.add_argument('-batch_size', type=int, dest='batch_size', default=32)
parser.add_argument('-halve_lr_every', dest='halve_lr_every',type=int, default = 2000)
parser.add_argument('-lr', type=float, dest='lr', help="Adam learning rate.", default=1e-4)
parser.add_argument('-seed', type=int, dest="seed", help="Seed for reproductibility.", default=1234)
parser.add_argument('-save_every', type=int, dest="save_every", default=10000)
parser.add_argument('-in_folder_train',dest='in_folder_train',type=str,default='/export/data/tahong/MRIData/Brain_data_MoDL/TrDataScaleMag.mat')#'/export/data/tahong/MRIData/knee_multicoil_train_Ims/TrDataMag.mat'
parser.add_argument('-out_folder',dest='out_folder',type=str,default='/faststorage/tahong/PotentialDenoisers/MRIGradDenoiser')
args = parser.parse_args()
#=========================================================================

cuda_index = args.cuda_index
MaxIter = args.MaxIter
numLayers = args.numLayers 
LossType = args.LossType
#------------------------------------------------------------------
device = torch.device('cuda:'+cuda_index)

def EvalImCriterion(img_set,model,device,sigma_min,sigma_max):
    PSNR_temp = 0
    SSIM_temp = 0
    model.eval()
    for iter_im in range(img_set.size(0)):
        im = img_set[iter_im,:,:,:]
        im_torch = im.to(device).unsqueeze(0)
        sigma = np.random.uniform(low=sigma_min, high=sigma_max)
        im_torch_noisy = im_torch+sigma*torch.randn_like(im_torch)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()  # In-place operation to zero out the gradient
        im_torch_noisy.requires_grad = True
        outputs = 0.5 * torch.sum((im_torch_noisy - model(im_torch_noisy)).reshape((im_torch_noisy.shape[0], -1)) ** 2,dim=1).unsqueeze(1)
        grad_f = torch.autograd.grad(outputs=outputs, inputs=im_torch_noisy, grad_outputs=torch.ones_like(outputs), create_graph=True, only_inputs=True)[0]
        im_torch_noisy.requires_grad = False
        img_clear = im_torch_noisy-grad_f.detach()
        PSNR = opt.PSNR(np.abs(torch.squeeze(im_torch).cpu().numpy()),np.abs(torch.squeeze(img_clear).cpu().numpy()))
        SSIM = opt.SSIM(np.abs(torch.squeeze(im_torch).cpu().numpy()),np.abs(torch.squeeze(img_clear).cpu().numpy()))
        PSNR_temp = PSNR_temp+PSNR
        SSIM_temp = SSIM_temp+SSIM
    psnr_ave = PSNR_temp/img_set.size(0)
    ssim_ave = SSIM_temp/img_set.size(0)
    model.train(True)
    return psnr_ave,ssim_ave


def load_images_train(in_folder):
    scale = 1/255.0
    trainData = scipy.io.loadmat(in_folder)
    trainData = trainData['trainData_temp']/scale
    trainData = torch.from_numpy(trainData)
    trainData = trainData.to(torch.float32)
    return trainData


def load_images_test(in_folder):
    testData_temp = h5py.File(in_folder, 'r')
    testData_temp = testData_temp['testdata']
    testData_temp = np.transpose(testData_temp, (3,2,1,0))
    testData = torch.from_numpy(testData_temp).to(torch.float32)/255.0
    return testData


class TrainingDataset(Dataset):
    def __init__(self, in_folder, sigma_min=1, sigma_max=55, batch_size=128,device=torch.device('cpu')):
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.batch_size = batch_size

        self.images_train = load_images_train(in_folder)
        self.number_of_images = len(self.images_train)
        self.device = device
    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):

        sigma = np.random.uniform(low=self.sigma_min, high=self.sigma_max)

        img_torch = self.images_train[idx,:,:,:].to(self.device)
        img_noisy_torch = img_torch+sigma*torch.randn_like(img_torch)
        sigma_torch = sigma * torch.ones(1, 1, 1, device=self.device)

        return img_torch, img_noisy_torch, sigma_torch

# Instantiate the model
model = modutl.Network(num_blocks=numLayers)
model = model.to(device)
model.train()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of trainable parameters: {total_params}')


if LossType == 'MSE':
    criterion = nn.MSELoss().to(device)
elif LossType == 'L1':
    criterion = nn.L1Loss().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
scheduler = StepLR(optimizer, step_size=args.halve_lr_every, gamma=0.5)
datasetTrain = TrainingDataset(in_folder=args.in_folder_train, sigma_min=args.sigma_min, sigma_max=args.sigma_max, batch_size=args.batch_size,device=device)
count_first = True
PSNR_anchor = 0
SSIM_anchor = 0
for k in range(MaxIter):
    img_torch, img_noisy_torch, sigma_torch = next(iter(DataLoader(datasetTrain, batch_size=args.batch_size, shuffle=True)))
    trainLabels = img_noisy_torch-img_torch
    img_noisy_torch.requires_grad = True
    outputs = model(img_noisy_torch)
    grad_f = torch.autograd.grad(outputs=outputs, inputs=img_noisy_torch, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    loss = criterion(grad_f/sigma_torch,trainLabels/sigma_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print('Loss is {} at {}th iteration'.format(loss.item(),k+1))
    writer.add_scalar('loss', loss.item(), k)
    if (k+1) % args.save_every == 0 and k!=0:
        torch.save(model.state_dict(), args.out_folder + "/PotentialDenoiser" +  "_iter" + str(k) +'sigma' + str(args.sigma_min) + str(args.sigma_max) + ".pth")
writer.close()
print('Done.\n')
