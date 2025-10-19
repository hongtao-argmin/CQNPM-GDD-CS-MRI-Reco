#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 00:04:30 2023
implement the optimization algorithms in pytorch 
for classical model-based reconstruction
Update: 11/18/2023 also include 3D case.
@author: hongtao
"""

import torch
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
#from models.network_dncnn import *
#from models.fdncnn import *
import matplotlib.pyplot as plt
def Power_Iter(A,AT,im_size,tol = 1e-6,device='cpu'):
    ''' 
    Power iteration to estimate the maximal eigenvalue of AHA.
    '''
    b_k = (torch.randn(im_size)).unsqueeze(0).unsqueeze(0).to(device)
    if AT == None:
        Ab_k = A(b_k+1j*b_k)
    else:
        Ab_k = AT(A(b_k+1j*b_k))
    norm_b_k = torch.norm(Ab_k)
    while True:
        b_k = Ab_k/norm_b_k
        if AT==None:
            Ab_k = A(b_k)
        else:
            Ab_k = AT(A(b_k))
        norm_b_k_1 = torch.norm(Ab_k)
        if torch.abs(norm_b_k_1-norm_b_k)<=tol:
            break
        else:
            norm_b_k = norm_b_k_1
    #b = b_k
    L = torch.vdot(b_k.flatten(),Ab_k.flatten()/torch.vdot(b_k.flatten(),b_k.flatten()))
    return torch.real(L)

def CG_Alg_Handle(x_k,RHS,A,MaxCG_Iter,tol=1e-6):
    r_k = RHS - A(x_k)
    p_k = r_k
    for iter in range(MaxCG_Iter):
        Ap_k = A(p_k)
        alpha_k = torch.vdot(r_k.flatten(),r_k.flatten())/torch.vdot(p_k.flatten(),Ap_k.flatten())
        x_k_1 = x_k+alpha_k*p_k
        if iter<MaxCG_Iter:
            r_k_1 = r_k - alpha_k*A(p_k)
            if torch.norm(r_k_1)<tol:
                break
            beta_k = torch.vdot(r_k_1.flatten(),r_k_1.flatten())/torch.vdot(r_k.flatten(),r_k.flatten())
            p_k_1 = r_k_1+beta_k*p_k
            p_k = p_k_1
            r_k = r_k_1
            x_k = x_k_1
    return x_k_1

def SSIM(original,compressed):
    return ssim(original,compressed,\
                data_range=compressed.max() - compressed.min())

def PSNR(original, compressed):
    mse = np.mean((np.abs(original - compressed)) ** 2)
    if(mse == 0):  
        return 100
    # decide the scale of the image
    if np.max(np.abs(original))<1.01:
        max_pixel = 1
    else:
        max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return psnr
def RecurrentPred(x,Ax,ATx,coef):
    if len(coef)==2:
        y = coef[0]*x+coef[1]*ATx(Ax(x))
        return y
    else:
        x_temp = RecurrentPred(x,Ax,ATx,coef[1:])
        y = coef[0]*x+ATx(Ax(x_temp))
    return y
#def PSNR(original, compressed):
#    return psnr(original,compressed,\
#                data_range=compressed.max() - compressed.min())
        
'''
    compressed = compressed/np.max(np.abs(compressed))
    mse = np.mean((np.abs(original - compressed)) ** 2)
    if(mse == 0):  
        return 100
    # decide the scale of the image
    if np.max(np.abs(original))<1.01:
        max_pixel = 1
    else:
        max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return psnr
'''

def denoiseCNN(y_noise,scale,denoiser):
    torch_tempR = torch.real(y_noise)
    torch_tempI = torch.imag(y_noise)
    temp = torch.max(torch.max(torch.abs(torch_tempR)),torch.max(torch.abs(torch_tempI)))
    torch_tempR = torch.real(y_noise)/(temp*scale)
    torch_tempI = torch.imag(y_noise)/(temp*scale)
    torch_temp = torch.cat((torch_tempR,torch_tempI),dim=1)
    with torch.no_grad():
        denoise_out = denoiser(torch_temp)
    temp_R = denoise_out[:,0,:,:].unsqueeze(0)*(temp*scale)
    temp_I = denoise_out[:,1,:,:].unsqueeze(0)*(temp*scale)
    x = (temp_R+1j*temp_I)
    return x

def denoiseCNNMag(y_noise,scale,denoiser):
    torch_mag = torch.abs(y_noise)
    torch_phase = torch.angle(y_noise)
    torch_mag = torch_mag/scale
    with torch.no_grad():
        denoise_out = denoiser(torch_mag)
    temp_mag = denoise_out*scale
    x = temp_mag*torch.exp(1j*torch_phase)
    return x

def ISTA_PnP(num_iters,Ax,ATx,b,Ch = 1, denoiser=None,scale=None,mu_corr=0,L=1,\
isPred = False,w_pred=lambda x: x,save=None,original=None,SaveIter=False,\
verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with ISTA PnP:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + R(x) 
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
    Ch: 1(default)-one challen denoising only mag; 2(or others)-real and imaginary parts denoising
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time,lst_fixed
    """

    AHb = ATx(b)
    x = AHb.to(device)
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_fixed = []
    if verbose:
        if isPred:
            pbar = tqdm(total=num_iters, desc="ISTA Pre - PnP",\
            leave=True)
        else:
            pbar = tqdm(total=num_iters, desc="ISTA PnP",\
            leave=True)
    
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    #if isPred:
    #    w_pred = lambda x: 2*x-ATx(Ax(x))#-mu_corr*x
    #else:
    #    w_pred = lambda x: x
    for k in range(num_iters):
        start_time = time.perf_counter()
        gr = ATx(Ax(x))-AHb#+mu_corr*x
        # temp = z-gr
        temp = x-w_pred(gr)/L
        '''
        torch_mag = torch.abs(temp)
        torch_phase = torch.angle(temp)
        temp_max = torch.max(torch_mag)
        torch_mag = torch_mag/temp_max*255
        with torch.no_grad():
            denoise_out = denoiser(torch_mag)
        temp_mag = denoise_out/255*temp_max
        x = temp_mag*torch.exp(1j*torch_phase)
        '''
        if Ch==1:
            x = denoiseCNNMag(temp,scale,denoiser)
        else:#if ch==2:
            x = denoiseCNN(temp,scale,denoiser)
        
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))

        if Ch == 1:
            fixed_temp = x - denoiseCNNMag(x-w_pred(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x
        else:
            fixed_temp = x - denoiseCNN(x-w_pred(ATx(Ax(x))-AHb)/L,scale,denoiser)
        lst_fixed.append(torch.norm(fixed_temp).cpu().numpy())
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())                
            np.save("%s/fixed.npy" % save, lst_fixed)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_fixed

def FISTA_PnP(num_iters,Ax,ATx,b,Ch = 1,isRestart=False,ReStartIter=20,denoiser=None,\
scale=None,mu_corr=0,L=1,save=None,isPred = False,w_pred=lambda x: x,original=None,\
SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with FISTA PnP:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + R(x) 
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """
    AHb = ATx(b)
    x = AHb.to(device)
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_fixed = []
    if verbose:
        if isRestart:
            if isPred:
                pbar = tqdm(total=num_iters, desc="Restart-FISTA Pre-PnP", \
                leave=True)
            else:
                pbar = tqdm(total=num_iters, desc="Restart-FISTA PnP", \
                leave=True)
        else:
            if isPred:
                pbar = tqdm(total=num_iters, desc="FISTA Pre-PnP", \
                leave=True)
            else:
                pbar = tqdm(total=num_iters, desc="FISTA PnP", \
                leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    #model_path = ('/faststorage/tahong/TrainModel/modelDnCNNMRIBrainDenoise_sigma{}.pth'.format(0.1))
    #scale = 0.00012998041861197527
    #model_path = ('/faststorage/tahong/TrainModel/ResizeImagesScale255_02222024_DnCNNMRIBrainDenoise_sigma{}.pth'.format(0.1))
    #scale = 0.00013353747187876233
    t_k_1 = 1
    #if isPred:
    #    w_pred = lambda x: 2*x-ATx(Ax(x))#-mu_corr*x
    #else:
    #    w_pred = lambda x: x
    z = x.clone().to(device)
    x_old = z
    for k in range(num_iters):
        start_time = time.perf_counter()
        gr = ATx(Ax(z))-AHb#+mu_corr*z
        #temp = z-gr
        temp = z-w_pred(gr)/L
        if Ch == 1:
            x = denoiseCNNMag(temp,scale,denoiser)
        else:
            x = denoiseCNN(temp,scale,denoiser)
        if isRestart:
            if np.mod(k,ReStartIter)==0:
                t_k_1 = 1
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        #t_k_1 = (k+1+3-1)/3
        z = x + ((t_k-1)/t_k_1)*(x - x_old)
        #z = (1+0.999999999)*x -0.999999999*x_old
        x_old = x
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy())) 
        if Ch == 1:
            fixed_temp = x - denoiseCNNMag(x-w_pred(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x
        else:
            fixed_temp = x- denoiseCNN(x-w_pred(ATx(Ax(x))-AHb)/L,scale,denoiser)
        lst_fixed.append(torch.norm(fixed_temp).cpu().numpy())
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
            np.save("%s/fixed.npy" % save, lst_fixed)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_fixed

def QNP_PnP(num_iters,Ax,ATx,b,Ch = 1,denoiser = None,scale=None,mu_corr=0,L=1,a_k=1,\
Hessian_Type = 'SR1',gamma=1.6,save=None,original=None,SaveIter=False,\
verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with QNP PnP:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + R(x) 
    
  Inputs:
    num_iters: Maximum number of iterations.
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """
    
    AHb = ATx(b)
    x = AHb.to(device)#torch.zeros_like(AHb)#
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_fixed = []
    lst_gradnorm = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="QNP PnP", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    x_old = x
    epsi = 1e-8
    mag_par = 1.414
    if Hessian_Type == 'Modified-SR1':
        # parameters for MOdified-SR1 to bounded the Hessian matrix
        eta = 1
        theta_1 = 2e-6
        theta_2 = 2e2
        beta_stepsize = 0.01
        MaxIterBeta = 100
    for k in range(num_iters):
        start_time = time.perf_counter()
        gr = ATx(Ax(x))-AHb#+mu_corr*x
        if k==0:
            temp_grad = x-gr/L
            sigma_max = 1
            gr_old = gr
        else:
            y_k = gr-gr_old
            s_k = x-x_old
            x_old = x
            gr_old = gr
            if Hessian_Type == 'SR1':
                y_k_dot = torch.real(torch.vdot(y_k.flatten(),y_k.flatten()))
                tau_BB = y_k_dot/torch.real(torch.vdot(s_k.flatten(),y_k.flatten()))
                if tau_BB<0:
                    temp_grad = x_old-gr
                    D = 1
                    sigma_max = a_k
                    Bx_inv = lambda xx: xx
                else:
                    H_0 = gamma*tau_BB
                    D = H_0
                    temp_1 = y_k-D*s_k
                    temp_2 = torch.real(torch.vdot(temp_1.flatten(),s_k.flatten())) 
                    if torch.abs(temp_2)<=epsi*torch.sqrt(y_k_dot)*torch.norm(temp_1):
                        u = 0
                        u_sign = 0
                        sigma_max = a_k
                        Bx_inv = lambda xx: (a_k/D)*xx
                    else:
                        u = temp_1/torch.sqrt(torch.abs(temp_2))
                        u_inv = u/D
                        u_sign = torch.sign(temp_2)
                        u_u_dot = torch.real(torch.vdot(u_inv.flatten(),u.flatten()))
                        u_sign_scale = a_k*u_sign/(1+u_sign*u_u_dot)
                        Bx_inv = lambda xx: ((a_k/D)*xx-(u_sign_scale*torch.vdot(u_inv.flatten(),xx.flatten()))*u_inv)
                        if u_sign>0:
                            #sigma_max = a_k/D
                            sigma_max = torch.real(a_k*D+torch.vdot(u.flatten(),u.flatten()))
                            sigma_min = torch.real(a_k*D)
                        else:
                            #sigma_max = a_k/torch.real(D-torch.vdot(u.flatten(),u.flatten()))
                            sigma_max =  torch.real(a_k*D)
                            sigma_min = torch.real(a_k*D-torch.vdot(u.flatten(),u.flatten()))
                        sigma_max = mag_par*sigma_max
            elif Hessian_Type == 'Modified-SR1':
                beta = 0
                for iter_beta in range(MaxIterBeta):
                    v_k = beta*s_k+((1-beta)*eta)*y_k
                    v_k_s_k = torch.real(torch.vdot(v_k.flatten(),s_k.flatten()))
                    s_k_s_k = torch.real(torch.vdot(s_k.flatten(),s_k.flatten()))
                    v_k_v_k = torch.real(torch.vdot(v_k.flatten(),v_k.flatten()))
                    if v_k_s_k/s_k_s_k>=theta_1 and v_k_v_k/v_k_s_k<=theta_2:
                        break
                    else:
                        beta = beta+beta_stepsize
                tau_temp = (s_k_s_k/v_k_s_k)#/gamma
                tau = tau_temp-torch.sqrt(tau_temp**2-s_k_s_k/v_k_v_k)# looks + will be better
                if tau<0:
                    temp_grad = x_old-gr
                    D = 1
                    sigma_max = a_k
                    Bx_inv = lambda xx: xx
                else:
                    H_0 = tau
                    D_inv = H_0
                    rho = v_k_s_k-H_0*v_k_v_k
                    if torch.abs(rho)<=epsi*(s_k_s_k-2*H_0*v_k_s_k+H_0**2*v_k_v_k)*torch.norm(v_k):
                        u = 0
                        u_sign = 0
                        sigma_max = a_k
                        Bx_inv = lambda xx: (a_k*H_0)*xx
                    else:
                        temp_1 = s_k-H_0*v_k
                        u_sign = torch.sign(rho)
                        u_inv =  temp_1/torch.sqrt(torch.abs(rho))
                        Bx_inv = lambda xx: ((a_k*H_0)*xx+(u_sign*torch.vdot(u_inv.flatten(),xx.flatten()))*u_inv)
                        if u_sign>0:
                            sigma_max = torch.real(1/(a_k*H_0))
                            sigma_min = 1/torch.real(a_k*H_0+torch.vdot(u_inv.flatten(),u_inv.flatten()))
                        else:
                            sigma_max = 1/torch.real(a_k*H_0-torch.vdot(u_inv.flatten(),u_inv.flatten()))
                            sigma_min = torch.real(1/(a_k*H_0))
                        sigma_max = mag_par*sigma_max
            #print(sigma_max.cpu()/sigma_min.cpu())
            temp_grad = x_old-Bx_inv(gr_old)
        if Ch == 1:
            x = denoiseCNNMag(temp_grad,scale/sigma_max,denoiser)
        else:
            x = denoiseCNN(temp_grad,scale/sigma_max,denoiser)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        if Ch == 1:
            fixed_temp = x - denoiseCNNMag(x-(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x
        else:
            fixed_temp = x - denoiseCNN(x-(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x 
        lst_fixed.append(torch.norm(fixed_temp).cpu().numpy()) 
        lst_gradnorm.append(torch.norm(gr).cpu().numpy())   
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
            np.save("%s/fixed.npy" % save, lst_fixed)
            np.save("%s/gradnorm.npy" % save, lst_gradnorm)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_fixed,lst_gradnorm


def QNP_PnP_BB(num_iters,Ax,ATx,b,Ch = 1,denoiser = None,scale=None,L=1,save=None,original=None,SaveIter=False,\
verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with QNP PnP:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + R(x) 
    
  Inputs:
    num_iters: Maximum number of iterations.
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """
    
    AHb = ATx(b)
    x = AHb.to(device)#torch.zeros_like(AHb)#
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_fixed = []
    lst_gradnorm = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="BB PnP", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    x_old = x
    for k in range(num_iters):
        start_time = time.perf_counter()
        gr = ATx(Ax(x))-AHb#+mu_corr*x
        if k==0:
            temp_grad = x-gr/L
            sigma_max = 1
            gr_old = gr
        else:
            y_k = gr-gr_old
            s_k = x-x_old
            x_old = x
            gr_old = gr
            v_k_s_k = torch.real(torch.vdot(y_k.flatten(),s_k.flatten()))
            s_k_s_k = torch.real(torch.vdot(s_k.flatten(),s_k.flatten()))
            tau = (s_k_s_k/v_k_s_k)
            temp_grad = x_old-gr_old/(tau*1.7)
        if Ch == 1:
            x = denoiseCNNMag(temp_grad,1,denoiser)
        else:
            x = denoiseCNN(temp_grad,1,denoiser)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        if Ch == 1:
            fixed_temp = x - denoiseCNNMag(x-(ATx(Ax(x))-AHb)/L,1,denoiser)#+mu_corr*x
        else:
            fixed_temp = x - denoiseCNN(x-(ATx(Ax(x))-AHb)/L,1,denoiser)#+mu_corr*x 
        lst_fixed.append(torch.norm(fixed_temp).cpu().numpy()) 
        lst_gradnorm.append(torch.norm(gr).cpu().numpy())   
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
            np.save("%s/fixed.npy" % save, lst_fixed)
            np.save("%s/gradnorm.npy" % save, lst_gradnorm)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_fixed,lst_gradnorm


def FISTA_RED(num_iters,Ax,ATx,b,Ch = 1,denoiser=None,scale=None,lamda=0.1,MaxCG_Iter = 4,save=None,original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with FISTA RED:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 +lamda 1/2x^T(x-f(x))
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """

    AHb = ATx(b)
    x = AHb.to(device)
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="FISTA RED", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    #model_path = ('/faststorage/tahong/TrainModel/modelDnCNNMRIBrainDenoise_sigma{}.pth'.format(0.1))
    #scale = 0.00012998041861197527
    #model_path = ('/faststorage/tahong/TrainModel/ResizeImagesScale255_02222024_DnCNNMRIBrainDenoise_sigma{}.pth'.format(0.1))
    #scale = 0.00013353747187876233
    z = x
    x_old = z
    t_k_1 = 1
    Ax_red = lambda xx: ATx(Ax(xx))+lamda*xx
    for k in range(num_iters):
        start_time = time.perf_counter()
        # call denoiser
        if Ch == 1:
            z = denoiseCNNMag(z,scale,denoiser)
        else:
            z = denoiseCNN(z,scale,denoiser)
        RHS = AHb+lamda*z
        x = CG_Alg_Handle(x_old,RHS,Ax_red,MaxCG_Iter,tol=1e-6)
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        z = x + ((t_k-1)/t_k_1)*(x - x_old)
        x_old = x
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time)

def ISTA_RED(num_iters,Ax,ATx,b,Ch = 1,denoiser=None,scale=None,lamda=0.1,MaxCG_Iter = 4,save=None,original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with ISTA RED:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 +lamda 1/2x^T(x-f(x))
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """

    AHb = ATx(b)
    x = AHb.to(device)
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="ISTA RED", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    #model_path = ('/faststorage/tahong/TrainModel/modelDnCNNMRIBrainDenoise_sigma{}.pth'.format(0.1))
    #scale = 0.00012998041861197527
    #model_path = ('/faststorage/tahong/TrainModel/ResizeImagesScale255_02222024_DnCNNMRIBrainDenoise_sigma{}.pth'.format(0.1))
    #scale = 0.00013353747187876233
    x_old = x
    t_k_1 = 1
    Ax_red = lambda xx: ATx(Ax(xx))+lamda*xx
    for k in range(num_iters):
        start_time = time.perf_counter()
        # call denoiser
        if Ch == 1:
            z = denoiseCNNMag(x_old,scale,denoiser)
        else:
            z = denoiseCNN(x_old,scale,denoiser)
        RHS = AHb+lamda*z
        x = CG_Alg_Handle(x_old,RHS,Ax_red,MaxCG_Iter,tol=1e-6)
        x_old = x
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time)

def QNP_RED(num_iters,Ax,ATx,b,Ch = 1,denoiser=None,scale=None,gamma=1.6,lamda=0.1,MaxCG_Iter = 4,save=None,original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with QNP RED:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 +lamda 1/2x^T(x-f(x))
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """

    AHb = ATx(b)
    x = AHb.to(device)
   
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="QNP RED", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    #model_path = ('/faststorage/tahong/TrainModel/modelDnCNNMRIBrainDenoise_sigma{}.pth'.format(0.1))
    #scale = 0.00012998041861197527
    #model_path = ('/faststorage/tahong/TrainModel/ResizeImagesScale255_02222024_DnCNNMRIBrainDenoise_sigma{}.pth'.format(0.1))
    #scale = 0.00013353747187876233
    x_old = x
    for k in range(num_iters):
        start_time = time.perf_counter()
        # call denoiser
        if Ch == 1:
            x_out = denoiseCNNMag(x,scale,denoiser)
        else:
            x_out = denoiseCNN(x,scale,denoiser)
        if k==0:
            g_k = lamda*(x-x_out)
        else:
            g_k_1 = lamda*(x-x_out)
        if k==0:
            Ax_red = lambda xx: ATx(Ax(xx))+lamda*xx
            RHS = AHb+lamda*x_out
        else:
            m_k = g_k_1-g_k
            s_k = x-x_old
            x_old = x
            g_k = g_k_1
            tau = gamma*torch.vdot(m_k.flatten(),m_k.flatten())/torch.vdot(m_k.flatten(),s_k.flatten())
            #tau = torch.abs(tau)
            if torch.abs(tau)<0:
                Bx = lambda xx:lamda*xx
            else:
                H_0 = tau
                temp_1 = m_k-H_0*s_k
                temp_2 = torch.vdot(s_k.flatten(),temp_1.flatten())
                if torch.abs(temp_2)<=1e-8*torch.norm(s_k)*torch.norm(temp_1):
                    u_k = 0
                else:
                    u_k = temp_1/torch.sqrt(temp_2)
                Bx = lambda xx: H_0*xx+u_k*torch.vdot(u_k.flatten(),xx.flatten())
            Ax_red = lambda xx:ATx(Ax(xx))+Bx(xx)
            RHS = AHb+Bx(x_old)-g_k
        # update x
        x = CG_Alg_Handle(x_old,RHS,Ax_red,MaxCG_Iter,tol=1e-6)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time)

def ADMM_PnP(num_iters,Ax,ATx,b,Ch = 1,denoiser = None,scale=None,mu_corr=0,sigma=1,eta=1,MaxCG_Iter = 4,save=None,original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with ADMM RED:
  .. math:
    \min_x \frac{1}{2sigma^2} \| A x - b \|_2^2 +\phi(x)
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
    sigma: represents the noise level, but one can set it to be one and apply it in the denoiser itself.
    eta: the parameters inside ADMM, has many different ways to choose it but we set it to be 1 following
    Rizwan Ahmad et al. Plug-and-Play Methods for Magnetic Resonance Imaging Using denoisers for image recovery. IEEE SPM.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """

    AHb = ATx(b)
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    x = AHb
    if verbose:
        pbar = tqdm(total=num_iters, desc="ADMM PnP", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    v_k = AHb
    u_k = torch.zeros_like(v_k)
    Ax_red = lambda xx: ATx(Ax(xx))+(sigma/eta)*xx#+mu_corr*xx
    for k in range(num_iters):
        start_time = time.perf_counter()
        if k==0:
            RHS = AHb
        else:
            RHS = AHb+(sigma/eta)*(v_k-u_k)
        x = CG_Alg_Handle(x,RHS,Ax_red,MaxCG_Iter,tol=1e-6)
        z = x+u_k
        # call denoiser
        if Ch == 1:
            v_k = denoiseCNNMag(z,scale,denoiser)
        else:
            v_k = denoiseCNN(z,scale,denoiser)
        u_k = u_k+(x-v_k)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time)

# -------------------------------------------------------------------
# stochastic version
def Scalable_QNP_PnP(num_iters,Ax,ATx,Ax_set,ATx_set,b,b_set,AHb_set,L_set,k_s= None,Ch = 1,denoiser = None,scale=None,mu_corr=0,L=1,a_k=1,\
Hessian_Type = 'SR1',gamma=1.6,save=None,original=None,SaveIter=False,\
verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with QNP PnP:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + R(x) 
    
  Inputs:
    num_iters: Maximum number of iterations.
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    Ax_set: contains Ax operator for each batch
    ATx_set: contains ATx operator for each batch
    b_set: contains k-space assciated to each batch
    AHb_set: contains AHb_set assciated to each batch
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """
    if k_s is None:
        k_s = len(Ax_set) 
    AHb = ATx(b)
    x = AHb.to(device)#torch.zeros_like(AHb)#
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_fixed = []
    lst_gradnorm = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="QNP PnP", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    size_1,size_2,size_3,size_4 = x.size()
    im_size = size_1*size_2*size_3*size_4
    # save the histories
    # for estimating the Hessian matrics
    x_old_set = torch.zeros(k_s,size_1,size_2,size_3,size_4,device=device,dtype=torch.complex64)
    g_old_set = torch.zeros(k_s,size_1,size_2,size_3,size_4,device=device,dtype=torch.complex64)
    # information for the Hessian matrix
    scale_set = torch.zeros(k_s,device=device)
    v_set = torch.zeros(im_size,k_s,device=device,dtype=torch.complex64)
    Bx_set = []
    
    epsi = 1e-8
    if Hessian_Type == 'Modified-SR1':
        # parameters for Modified-SR1 to bound the Hessian matrix
        eta = 1
        theta_1 = 2e-6
        theta_2 = 2e2
        beta_stepsize = 0.01
        MaxIterBeta = 100
    # Hessian structure, 'SR1': D+UU^H, 'Modified-SR1':D-UU^H.
    # in complex version.
    for k in range(num_iters):
        start_time = time.perf_counter()
        count_index = np.mod(k,k_s)
        gr = ATx_set[count_index](Ax_set[count_index](x))-AHb_set[count_index]
        if k<k_s:
            temp_grad = x-gr#/L_set[count_index]
            x_old_set[count_index,:,:,:,:] = x
            g_old_set[count_index,:,:,:,:] = gr
            Bx = lambda xx: L_set[count_index]*xx
            Bx_inv = lambda xx: 1/L_set[count_index]*xx
            Bx_set.append(Bx)
            v_set[:,count_index] = torch.zeros(im_size,device=device,dtype=torch.complex64)
            scale_set[count_index] = L_set[count_index]
        else:

            y_k = gr-g_old_set[count_index,:,:,:,:]
            s_k = x-x_old_set[count_index,:,:,:,:]
            x_old_set[count_index,:,:,:,:] = x
            g_old_set[count_index,:,:,:,:] = gr

            if Hessian_Type == 'SR1':
                y_k_dot = torch.real(torch.vdot(y_k.flatten(),y_k.flatten()))
                tau_BB = y_k_dot/torch.real(torch.vdot(s_k.flatten(),y_k.flatten()))
                if tau_BB<0:
                    temp_grad = x-gr#/L_set[count_index]
                    D = L_set[count_index]
                    Bx = lambda xx: L_set[count_index]*xx
                    Bx_set[count_index] = Bx
                    v_set[:,count_index] = torch.zeros(im_size,device=device,dtype=torch.complex64)
                    scale_set[count_index] = D
                    Bx_inv = lambda xx: 1/L_set[count_index]*xx
                else:
                    H_0 = gamma*tau_BB
                    temp_1 = y_k-H_0*s_k
                    temp_2 = torch.real(torch.vdot(temp_1.flatten(),s_k.flatten()))#.to(torch.complex64)
                    if torch.abs(temp_2)<=epsi*torch.sqrt(y_k_dot)*torch.norm(temp_1):
                        Bx = lambda xx: H_0*xx
                        v_set[:,count_index] = torch.zeros(im_size,device=device,dtype=torch.complex64)
                        scale_set[count_index] = H_0
                        Bx_set[count_index] = Bx
                        Bx_inv = lambda xx: 1/H_0*xx
                    else:
                        u = temp_1/torch.sqrt(temp_2)
                        Bx = lambda xx: (H_0*xx+(torch.vdot(u.flatten(),xx.flatten()))*u)
                        Bx_set[count_index] = Bx
                        v_set[:,count_index] = u.flatten()
                        scale_set[count_index] = H_0
                        u = temp_1 / torch.sqrt(torch.abs(temp_2))
                        u_inv = u/H_0
                        u_sign = torch.sign(temp_2)
                        u_u_dot = torch.real(torch.vdot(u_inv.flatten(),u.flatten()))
                        u_sign_scale = a_k*u_sign/(1+u_sign*u_u_dot)
                        Bx_inv = lambda xx: ((1/H_0)*xx-(u_sign_scale*torch.vdot(u_inv.flatten(),xx.flatten()))*u_inv)
            elif Hessian_Type == 'Modified-SR1':
                beta = 0
                for iter_beta in range(MaxIterBeta):
                    v_k = beta*s_k+((1-beta)*eta)*y_k
                    v_k_s_k = torch.real(torch.vdot(v_k.flatten(),s_k.flatten()))
                    s_k_s_k = torch.real(torch.vdot(s_k.flatten(),s_k.flatten()))
                    v_k_v_k = torch.real(torch.vdot(v_k.flatten(),v_k.flatten()))
                    if v_k_s_k/s_k_s_k>=theta_1 and v_k_v_k/v_k_s_k<=theta_2:
                        break
                    else:
                        beta = beta+beta_stepsize
                tau_temp = (s_k_s_k/v_k_s_k)#/gamma
                tau = tau_temp-torch.sqrt(tau_temp**2-s_k_s_k/v_k_v_k)# looks + will be better
                if tau<0:
                    temp_grad = x-gr#/L_set[count_index]
                    D = L_set[count_index]
                    Bx = lambda xx: L_set[count_index]*xx
                    Bx_inv = lambda xx: 1/L_set[count_index]*xx
                    Bx_set[count_index] = Bx
                    v_set[:,count_index] = torch.zeros(im_size,device=device,dtype=torch.complex64)
                    scale_set[count_index] = D
                else:
                    H_0 = tau/gamma
                    rho = torch.real(v_k_s_k-H_0*v_k_v_k)#.to(torch.complex64)
                    if torch.abs(rho)<=epsi*(s_k_s_k-2*H_0*v_k_s_k+H_0**2*v_k_v_k)*torch.norm(v_k):
                        Bx = lambda xx: (1/H_0)*xx
                        Bx_inv = lambda xx: H_0*xx
                        Bx_set[count_index] = Bx
                        v_set[:,count_index] = torch.zeros(im_size,device=device,dtype=torch.complex64)
                        scale_set[count_index] = 1/H_0
                    else:
                        temp_1 = s_k-H_0*v_k
                        u_sign = torch.sign(rho)
                        u_inv =  temp_1/torch.sqrt(torch.abs(rho))
                        Bx_inv = lambda xx: ((a_k*H_0)*xx+(u_sign*torch.vdot(u_inv.flatten(),xx.flatten()))*u_inv)            
                        u = u_inv/H_0
                        u_u_dot = torch.real(torch.vdot(u_inv.flatten(),u.flatten()))
                        u_sign_scale = 1/(1+u_u_dot)
                        Bx = lambda xx: ((1/H_0)*xx-(u_sign_scale*torch.vdot(u.flatten(),xx.flatten()))*u)
                        Bx_set[count_index] = Bx
                        v_set[:,count_index] = u.flatten()*torch.sqrt(u_sign_scale)
                        scale_set[count_index] = 1/H_0
            temp_grad = x-Bx_inv(gr)
            '''
            if k<2*k_s:
                for iter_k in range(count_index):
                    if iter_k == 0:
                        temp_grad = Bx_set[iter_k](x_old_set[iter_k,:,:,:,:])-a_k*(g_old_set[iter_k,:,:,:,:])
                    else:
                        temp_grad = temp_grad+Bx_set[iter_k](x_old_set[iter_k,:,:,:,:])-a_k*(g_old_set[iter_k,:,:,:,:])
                D_temp = torch.sum(scale_set[0:count_index+1])
                v_set_temp = v_set[:,0:count_index+1]
                k_s_temp = count_index+1
            else:
                for iter_k in range(k_s):
                    if iter_k == 0:
                        temp_grad = Bx_set[iter_k](x_old_set[iter_k,:,:,:,:])-a_k*(g_old_set[iter_k,:,:,:,:])
                    else:
                        temp_grad = temp_grad+Bx_set[iter_k](x_old_set[iter_k,:,:,:,:])-a_k*(g_old_set[iter_k,:,:,:,:])
                D_temp = torch.sum(scale_set)
                v_set_temp = v_set
                k_s_temp = k_s
            if Hessian_Type == 'SR1':
                # D+UU^H
                Id = torch.eye(k_s_temp,device=device)
            elif Hessian_Type == 'Modified-SR1':
                # D-UU^H
                Id = -torch.eye(k_s_temp,device=device)
            temp_mid = torch.linalg.inv(Id+(v_set_temp.conj().t()@v_set_temp)/D_temp)@(v_set_temp.conj().t()@temp_grad.flatten())
            temp_grad = temp_grad/D_temp-\
            torch.reshape((1/D_temp**2)*(v_set_temp@torch.reshape(temp_mid,(k_s_temp,1))),(size_1,size_2,size_3,size_4))
            '''
        if Ch == 1:
            x = denoiseCNNMag(temp_grad,scale,denoiser)
        else:
            x = denoiseCNN(temp_grad,scale,denoiser)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        if Ch == 1:
            fixed_temp = x - denoiseCNNMag(x-(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x
        else:
            fixed_temp = x - denoiseCNN(x-(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x 
        lst_fixed.append(torch.norm(fixed_temp).cpu().numpy()) 
        lst_gradnorm.append(torch.norm(gr).cpu().numpy())   
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
            np.save("%s/fixed.npy" % save, lst_fixed)
            np.save("%s/gradnorm.npy" % save, lst_gradnorm)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_fixed,lst_gradnorm

def Scalable_QNP_PnP_VRG(num_iters,Ax,ATx,Ax_set,ATx_set,b,b_set,AHb_set,L_set,k_s= None,Ch = 1,denoiser = None,scale=None,mu_corr=0,L=1,a_k=1,\
Hessian_Type = 'SR1',gamma=1.6,save=None,original=None,SaveIter=False,\
verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with QNP PnP:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + R(x) 
    
  Inputs:
    num_iters: Maximum number of iterations.
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    Ax_set: contains Ax operator for each batch
    ATx_set: contains ATx operator for each batch
    b_set: contains k-space assciated to each batch
    AHb_set: contains AHb_set assciated to each batch
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """
    if k_s is None:
        k_s = len(Ax_set) 
    AHb = ATx(b)
    x = AHb.to(device)#torch.zeros_like(AHb)#
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_fixed = []
    lst_gradnorm = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="Scalable QNP PnP", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    size_1,size_2,size_3,size_4 = x.size()
    im_size = size_1*size_2*size_3*size_4
    # information for the Hessian matrix
    scale_set = torch.zeros(k_s,device=device)
    gr_set = torch.zeros(im_size,k_s,device=device,dtype=torch.complex64)
    
    epsi = 1e-8
    if Hessian_Type == 'Modified-SR1':
        # parameters for Modified-SR1 to bounded the Hessian matrix
        eta = 1
        theta_1 = 2e-6
        theta_2 = 2e2
        beta_stepsize = 0.01
        MaxIterBeta = 100
    # Hessian structure, 'SR1': D+UU^H, 'Modified-SR1':D-UU^H.
    # in complex version.
    for k in range(num_iters):
        start_time = time.perf_counter()
        count_index = np.mod(k,k_s+1)
        if count_index == 0:
            for iter_k in range(k_s):
                temp_gr = ATx_set[iter_k](Ax_set[iter_k](x))-AHb_set[iter_k]
                gr_set[:,iter_k] = temp_gr.flatten()
            gr_tot = torch.reshape(torch.sum(gr_set,1),(size_1,size_2,size_3,size_4))
            gr = gr_tot
            temp_grad = x-gr
            x_old = x
            gr_old = gr
        else:
            gr = ATx_set[count_index-1](Ax_set[count_index-1](x))-AHb_set[count_index-1]
            y_k = gr-gr_old
            s_k = x-x_old
            gr = gr-torch.reshape(gr_set[:,count_index-1],(size_1,size_2,size_3,size_4))+gr_tot
            x_old = x
            gr_old = gr
            if Hessian_Type == 'SR1':
                y_k_dot = torch.real(torch.vdot(y_k.flatten(),y_k.flatten()))
                tau_BB = y_k_dot/torch.real(torch.vdot(s_k.flatten(),y_k.flatten()))
                if tau_BB<0:
                    temp_grad = x-gr
                    Bx_inv = lambda xx: a_k*xx
                else:
                    H_0 = gamma*tau_BB
                    temp_1 = y_k-H_0*s_k
                    temp_2 = torch.real(torch.vdot(temp_1.flatten(),s_k.flatten()))
                    if torch.abs(temp_2)<=epsi*torch.sqrt(y_k_dot)*torch.norm(temp_1):
                        Bx_inv = lambda xx: (a_k/H_0)*xx
                    else:
                        u = temp_1/torch.sqrt(torch.abs(temp_2))
                        u_inv = u/H_0
                        u_sign = torch.sign(temp_2)
                        u_u_dot = torch.real(torch.vdot(u_inv.flatten(),u.flatten()))
                        u_sign_scale = a_k*u_sign/(1+u_sign*u_u_dot)
                        Bx_inv = lambda xx: ((a_k/H_0)*xx-(u_sign_scale*torch.vdot(u_inv.flatten(),xx.flatten()))*u_inv)
            elif Hessian_Type == 'Modified-SR1':
                beta = 0
                for iter_beta in range(MaxIterBeta):
                    v_k = beta*s_k+((1-beta)*eta)*y_k
                    v_k_s_k = torch.real(torch.vdot(v_k.flatten(),s_k.flatten()))
                    s_k_s_k = torch.real(torch.vdot(s_k.flatten(),s_k.flatten()))
                    v_k_v_k = torch.real(torch.vdot(v_k.flatten(),v_k.flatten()))
                    if v_k_s_k/s_k_s_k>=theta_1 and v_k_v_k/v_k_s_k<=theta_2:
                        break
                    else:
                        beta = beta+beta_stepsize
                tau_temp = (s_k_s_k/v_k_s_k)#/gamma
                tau = tau_temp-torch.sqrt(tau_temp**2-s_k_s_k/v_k_v_k)# looks + will be better
                if tau<0:
                    temp_grad = x-gr
                    Bx_inv = lambda xx: xx
                else:
                    H_0 = tau#/gamma
                    rho = torch.real(v_k_s_k-H_0*v_k_v_k)
                    if torch.abs(rho)<=epsi*(s_k_s_k-2*H_0*v_k_s_k+H_0**2*v_k_v_k)*torch.norm(v_k):
                        Bx_inv = lambda xx: (a_k*H_0)*xx
                    else:
                        temp_1 = s_k-H_0*v_k
                        u_sign = torch.sign(rho)
                        u_inv =  temp_1/torch.sqrt(torch.abs(rho))
                        Bx_inv = lambda xx: ((a_k*H_0)*xx+(u_sign*torch.vdot(u_inv.flatten(),xx.flatten()))*u_inv)
            temp_grad = x-Bx_inv(gr)
        if Ch == 1:
            x = denoiseCNNMag(temp_grad,scale,denoiser)
        else:
            x = denoiseCNN(temp_grad,scale,denoiser)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        if Ch == 1:
            fixed_temp = x - denoiseCNNMag(x-(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x
        else:
            fixed_temp = x - denoiseCNN(x-(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x 
        lst_fixed.append(torch.norm(fixed_temp).cpu().numpy()) 
        lst_gradnorm.append(torch.norm(gr).cpu().numpy())   
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
            np.save("%s/fixed.npy" % save, lst_fixed)
            np.save("%s/gradnorm.npy" % save, lst_gradnorm)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_fixed,lst_gradnorm

def Scalable_ISTA_PnP(num_iters,Ax,ATx,Ax_set,ATx_set,b,b_set,AHb_set,L_set,k_s= None,Ch = 1,denoiser = None,scale=None,mu_corr=0,L=1,\
save=None,original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with QNP PnP:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + R(x) 
    
  Inputs:
    num_iters: Maximum number of iterations.
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    Ax_set: contains Ax operator for each batch
    ATx_set: contains ATx operator for each batch
    b_set: contains k-space assciated to each batch
    AHb_set: contains AHb_set assciated to each batch
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """
    if k_s is None:
        k_s = len(Ax_set) 
    AHb = ATx(b)
    x = AHb.to(device)#torch.zeros_like(AHb)#
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_fixed = []
    lst_gradnorm = []
    # w_pred = lambda x: 4*x-ATx(Ax((10/3)*x))
    if verbose:
        pbar = tqdm(total=num_iters, desc="Scalable ISTA PnP", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    
    for k in range(num_iters):
        start_time = time.perf_counter()
        count_index = np.mod(k,k_s)
        gr = ATx_set[count_index](Ax_set[count_index](x))-AHb_set[count_index]
        temp_grad = x-(4*gr-ATx_set[count_index](Ax_set[count_index]((10/3)*gr)))#/L_set[count_index]
        #temp_grad = x - gr/L_set[count_index]
        if Ch == 1:
            x = denoiseCNNMag(temp_grad,scale,denoiser)
        else:
            x = denoiseCNN(temp_grad,scale,denoiser)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        if Ch == 1:
            fixed_temp = x - denoiseCNNMag(x-(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x
        else:
            fixed_temp = x - denoiseCNN(x-(ATx(Ax(x))-AHb)/L,scale,denoiser)#+mu_corr*x 
        lst_fixed.append(torch.norm(fixed_temp).cpu().numpy()) 
        lst_gradnorm.append(torch.norm(gr).cpu().numpy())   
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
            np.save("%s/fixed.npy" % save, lst_fixed)
            np.save("%s/gradnorm.npy" % save, lst_gradnorm)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_fixed,lst_gradnorm


def Scalable_ADMM_PnP(num_iters,Ax,ATx,Ax_set,ATx_set,b,b_set,AHb_set,L_set,k_s= None,Ch = 1,denoiser = None,scale=None,mu_corr=0,sigma=1,eta=1,MaxCG_Iter = 4,save=None,original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with ADMM PnP:
  .. math:
    \min_x \frac{1}{2sigma^2} \| A x - b \|_2^2 +\phi(x)
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
    L: the maximal eiganvalue for A'A.
    sigma: represents the noise level, but one can set it to be one and apply it in the denoiser itself.
    eta: the parameters inside ADMM, has many different ways to choose it but we set it to be 1 following
    Rizwan Ahmad et al. Plug-and-Play Methods for Magnetic Resonance Imaging Using denoisers for image recovery. IEEE SPM.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """

    AHb = ATx(b)
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    x = AHb
    if verbose:
        pbar = tqdm(total=num_iters, desc="Scalable ADMM PnP", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    v_k = AHb
    u_k = torch.zeros_like(v_k)
    Ax_red = lambda xx: ATx(Ax(xx))+(sigma/eta)*xx#+mu_corr*xx
    for k in range(num_iters):
        start_time = time.perf_counter()
        count_index = np.mod(k,k_s)
        Ax_sub_pred = lambda xx: ATx_set[count_index](Ax_set[count_index](xx))+(sigma/eta)*xx
        AHb_sub = AHb_set[count_index]
        if k==0:
            RHS = AHb_sub
        else:
            RHS = AHb_sub+(sigma/eta)*(v_k-u_k)
        x = CG_Alg_Handle(x,RHS,Ax_sub_pred,MaxCG_Iter,tol=1e-6)
        z = x+u_k
        # call denoiser
        if Ch == 1:
            v_k = denoiseCNNMag(z,scale,denoiser)
        else:
            v_k = denoiseCNN(z,scale,denoiser)
        u_k = u_k+(x-v_k)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            
        #lst_cost.append(0.5*(torch.norm(Ax(x)-b)**2).cpu().numpy()/L.cpu().numpy()+TR_off*torch.sum(torch.abs(torch.squeeze(Wx(x)))).cpu().numpy())         
        if save != None:
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                np.save("%s/iter_%03d.npy" % (save, k), torch.squeeze(x).cpu().numpy())
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return x,lst_psnr,lst_ssim,np.cumsum(lst_time)
