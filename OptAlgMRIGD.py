#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Complex plane.

Implement the optimization algorithms in pytorch 
for deep gradient denoiser based regularizer

for clarity, we only implememt the gradient version algorithm here.
i.e., we work on the model proposed in 
Regev Cohen et al. It Has Potential: Gradient-Driven Denoisers for Convergent Solutions to Inverse Problems, NIPS.

This is for MRI, so it supports complex number.
@author: hongtao

# Tao Hong, Zhaoyi Xu, Se Young Chun, Luis Hernandez-Garcia, and Jeffrey A. Fessler, 
# ``Convergent Complex Quasi-Newton Proximal Methods for Gradient-Driven Denoisers in Compressed Sensing MRI Reconstruction'',
# To appear in IEEE Transactions on Computational Imaging, arXiv:2505.04820, 2025.

"""

import torch
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import scipy.io
import cvxpy as cp

# formulate forward model and its adjoint with Cartesian or non-Cartesian sampling
def ForwardandAdjoint(smaps,L_sr,ktraj=None,trj_type='Cartesian',nufft_ob=None,adjnufft_ob=None):
    if trj_type != 'Cartesian':
        Ax = lambda x: nufft_ob(x, ktraj, smaps=smaps.to(x))/L_sr
        ATx = lambda x: adjnufft_ob(x,ktraj,smaps=smaps.to(x))/L_sr
    else:
        Ax = lambda x: Ax_operator(x,smaps,ktraj)/L_sr
        ATx = lambda x: ATx_operator(x,smaps,ktraj)/L_sr
    return Ax,ATx

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

def project(x,ub):
    # project for |x_i|\leq ub
    return x/torch.maximum(torch.tensor(1),torch.max(torch.abs(x))/ub)

def Coil_Compression(kspace,maps,coils_out,isCartesian=False):
    # we only implement the version to use numCoils for coils compression
    # in pytorch
    _,coils_in,im_m,im_n = maps.size()
    if isCartesian:
        kspace = torch.reshape(torch.squeeze(kspace),(coils_in,im_m*im_n))
    kspace = torch.squeeze(kspace)
    _,S,Vh = torch.linalg.svd(kspace.t(),full_matrices=False)

    Vh = Vh[:,0:coils_out]
    kspace = (kspace.t()@Vh).t()
    maps_out = (torch.reshape(torch.squeeze(maps),(coils_in,im_m*im_n)).t()@Vh).t()
    maps_out = torch.reshape(maps_out,(coils_out,im_m,im_n)).unsqueeze(0)
    if isCartesian:
        kspace = torch.reshape(kspace,(coils_out,im_m,im_n)).unsqueeze(0)
    else: 
        kspace = kspace.unsqueeze(0)
    return kspace,maps_out

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
        if torch.abs(norm_b_k_1-norm_b_k)<=tol:#/norm_b_k_1
            break
        else:
            norm_b_k = norm_b_k_1
    L = torch.vdot(b_k.flatten(),Ab_k.flatten()/torch.vdot(b_k.flatten(),b_k.flatten()))
    return torch.real(L)

def CG_Alg_Handle(x_k,RHS,A,MaxCG_Iter,tol=1e-6):
    iter_fbw = 0
    r_k = RHS - A(x_k)
    iter_fbw = iter_fbw+1
    p_k = r_k
    for iter in range(MaxCG_Iter):
        Ap_k = A(p_k)
        iter_fbw = iter_fbw+1
        alpha_k = torch.vdot(r_k.flatten(),r_k.flatten())/torch.vdot(p_k.flatten(),Ap_k.flatten())
        x_k_1 = x_k+alpha_k*p_k
        if iter<MaxCG_Iter:
            r_k_1 = r_k - alpha_k*Ap_k
            if torch.norm(r_k_1)<tol:
                break
            beta_k = torch.vdot(r_k_1.flatten(),r_k_1.flatten())/torch.vdot(r_k.flatten(),r_k.flatten())
            p_k_1 = r_k_1+beta_k*p_k
            p_k = p_k_1
            r_k = r_k_1
            x_k = x_k_1
    return x_k_1,iter_fbw


def PGD_Alg_Handle(x_old,RHS,A,Max_Iter,alpha=1,mu=None,isCost = False,isStrong=False,beta=1.1,cost_fun=None,ub=1,tol=1e-6):
    # solve 0.5 x^H A x- b^H x, s.t., |x|<=1
    # 1/alpha is the largest eigenvalue of A
    v_int = x_old
    iter_fbw = 0
    if alpha==None:
        alpha = 1
        beta = 1.2
        isLinesearch = True
    else:
        isLinesearch = False
 
    cost_set = []
    if isCost:
        cost_set.append(cost_fun(x_old).cpu().numpy())
        iter_fbw = iter_fbw+0.5
        
    if isStrong:
        if torch.is_tensor(alpha):
            acc_par = (torch.sqrt(1/alpha)-torch.sqrt(mu))/(torch.sqrt(1/alpha)+torch.sqrt(mu))
        else:
            acc_par = (np.sqrt(1/alpha)-np.sqrt(mu))/(np.sqrt(1/alpha)+np.sqrt(mu))
        alpha = 4/3*alpha

    for iter in range(Max_Iter):
        grad = A(v_int)-RHS
        iter_fbw = iter_fbw+1
        if isLinesearch:
            cost_pre = cost_fun(v_int)
            iter_fbw = iter_fbw+0.5
            while True:
                x = v_int-alpha*grad
                x = project(x,ub)
                cost_curr = cost_fun(x)
                iter_fbw = iter_fbw+0.5
                if cost_curr > cost_pre+torch.real(torch.vdot(grad.flatten(),(x-v_int).flatten()))+1/(2*alpha)*torch.norm(x-v_int)**2:
                    if alpha<1e-8:
                        break
                    alpha = alpha/beta
                else:
                    break
            if isCost:
                cost_set.append(cost_curr.cpu().numpy())
        else:
            x = v_int-alpha*grad
            x = project(x,ub)
            if isCost:
                cost_set.append(cost_fun(x).cpu().numpy())
                iter_fbw = iter_fbw+0.5
        x_diff = x-x_old
        x_old = x
        if torch.norm(x_diff)<tol:
            break
        else:
            if isStrong:
                v_int = x+acc_par*x_diff
            else:
                v_int = x+iter/(iter+3)*x_diff
    return x,cost_set,iter_fbw

def WPM_Solver(y,D_temp,u_temp,u_sign,ub,alpha_star=1):
    '''
    solver for min_x |x-y|_W^2, s.t. forall |x_j| leq ub
    W = D_temp+u_sign u_temp u_temp^H
    '''
    if torch.norm(u_temp)==0:
        x = project(y,ub)
        alpha_star = None
    else:
        if alpha_star==None:
            alpha_star_old = 1
        else:
            alpha_star_old = alpha_star
        if u_sign:
            while True:
                temp = y-u_temp/D_temp*alpha_star_old
                grad = torch.vdot(u_temp.flatten(),(y-project(temp,ub)).flatten())+alpha_star_old
                index = torch.abs(temp)>ub
                temp[index] = 0
                Jacob = 1+torch.vdot(u_temp.flatten(),temp)
                alpha_star = alpha_star_old - grad/Jacob
                if torch.abs(grad)<1e-6:
                    break
                else:
                    alpha_star_old = alpha_star
            x = project(y-u_temp/D_temp*alpha_star,ub)
        else:
            while True:
                temp = y+u_temp/D_temp*alpha_star_old
                grad = torch.vdot(u_temp.flatten(),(y-project(temp,ub)).flatten())+alpha_star_old
                index = torch.abs(temp)>ub
                temp[index] = 0
                Jacob = 1+torch.vdot(u_temp.flatten(),temp.flatten())
                alpha_star = alpha_star_old - grad/Jacob
                if torch.abs(grad)<1e-6:
                    break
                else:
                    alpha_star_old = alpha_star
            x = project(y+u_temp/D_temp*alpha_star,ub)
    return x,alpha_star


def SSIM(original,compressed):
    if np.max(original)<1.1:
        data_range = 1.0
    else:
        data_range = 255.0
    return ssim(original,compressed,\
                data_range=data_range)

def PSNR(original, compressed):
    if np.max(original)<1.1:
        data_range = 1.0
    else:
        data_range = 255.0
    return psnr(original,compressed,\
                data_range=data_range)

def EnergyCNNMagGrad(y_noise,scale,model):
    # compute nabla g_theta (x)
    # the equivalent denoiser is x - nabla g_theta (x)
    torch_mag = torch.abs(y_noise)
    torch_phase = torch.angle(y_noise)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()  # In-place operation to zero out the gradient
    torch_mag.requires_grad = True
    output_test = model(torch_mag)
    grad_f_test = torch.autograd.grad(outputs=output_test, inputs=torch_mag, grad_outputs=torch.ones_like(output_test), create_graph=True)[0]
    torch_mag.requires_grad = False
    temp_mag = grad_f_test.detach()
    x = temp_mag*torch.exp(1j*torch_phase)
    return x

def CostCNNMagGrad(y_noise,scale,model):
    # compute g_theta (x)
    torch_mag = torch.abs(y_noise)
    with torch.no_grad():
        output_test = model(torch_mag)
    return torch.squeeze(output_test)

def GDEnergyGrad(num_iters,Ax,ATx,b,ub=np.inf,x_ini=None,g_theta=None,scale=1,lamda=0.1,alpha=0.1,\
                 beta = 2,isLinearSearch = False,save=None,original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with gradient-driven denoiser descent RED:

  min_x frac{1}{2} || A x - b ||^2 +lamda g_theta(x)
    
  Inputs:
    num_iters: Maximum number of iterations.
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    Ch: 1, only Mag; 2, real and imaginary
    g_theta: energy function
    scale: scale the image for denoising
    lamda: trade-off parmeter
    alpha: initial stepsize
    beta: half the stepsize if we use line search
    isLinearSearch: true: use line search, false (default): fixed stepsize
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
  
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """

    AHb = ATx(b)
    if x_ini is None:
        x = AHb.to(device)
    else:
        x = x_ini
    if not (ub==np.inf):
        x = project(x,ub)
    x_ini_norm = torch.norm(x)**2
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_cost = []
    lst_iter_diff = []
    lst_iter_diff_normXini = []
    lst_fwcount = []
    lst_bwcount = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="GD - GD Denoiser", \
                    leave=True)
        
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    cost_pre = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
    lst_cost.append(cost_pre.cpu().numpy())

    x_old = x
    for k in range(num_iters):
        count_fw = 0
        count_bw = 0
        start_time = time.perf_counter()
        if alpha<1e-8:
            break
        grad_1 = ATx(Ax(x_old)-b)
        count_fw = count_fw+1
        count_bw = count_bw+1
        # call denoiser for grad_2
        grad_2 = lamda*EnergyCNNMagGrad(x_old,scale,g_theta)
        grad = grad_1+grad_2
        if isLinearSearch:
            cost_pre = 0.5*torch.norm(Ax(x_old)-b)**2+lamda*CostCNNMagGrad(x_old,scale,g_theta)
            count_fw = count_fw+1
            if ub==np.inf:
                x = x_old - alpha*grad
            else:
                x = x_old - alpha*grad
                x = project(x,ub)
            while True:

                cost_temp = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
                count_fw = count_fw+1
                if cost_temp>cost_pre - 0.5*alpha*torch.norm(grad)**2:
                    alpha = alpha/beta
                    if ub==np.inf:
                        x = x_old - alpha*grad
                    else:
                        x = project(x_old - alpha*grad,ub)
                    if alpha<1e-8:
                        break
                else:
                    break
        else:
            if ub==np.inf:
                x = x_old - alpha*grad
            else:
                x = project(x_old - alpha*grad,ub)
        x_old = x
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        lst_iter_diff_normXini.append((torch.norm(x-x_old)**2/x_ini_norm).cpu().numpy())
        lst_iter_diff.append((torch.norm(x-x_old)**2).cpu().numpy())
        lst_fwcount.append(count_fw)
        lst_bwcount.append(count_bw)

        if not isLinearSearch:
            cost_temp = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
        lst_cost.append(cost_temp.cpu().numpy())

        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        
        if save != None:
            if SaveIter:
                Dict_iter = {'im':torch.squeeze(x).cpu().numpy()} 
                scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_iter)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if save != None:
            lst_time_cumsum = np.cumsum(lst_time)
            if original is not None:
                Dict = {'lst_time':lst_time_cumsum,'lst_cost':lst_cost,'lst_iter_diff':lst_iter_diff,\
                        'lst_iter_diff_norm':lst_iter_diff_normXini,'lst_psnr':lst_psnr,'lst_ssim':lst_ssim,\
                        'lst_fwcount':lst_fwcount,'lst_bwcount':lst_bwcount}
                scipy.io.savemat("%s/Results.mat" % save,Dict)
                np.savetxt("%s/lst_time.txt" % save,lst_time_cumsum)
                np.savetxt("%s/lst_cost.txt" % save,lst_cost)
                np.savetxt("%s/lst_iter_diff.txt" % save,lst_iter_diff)
                np.savetxt("%s/lst_iter_diff_norm.txt" % save,lst_iter_diff_normXini)
                np.savetxt("%s/lst_psnr.txt" % save,lst_psnr)
                np.savetxt("%s/lst_ssim.txt" % save,lst_ssim)
                np.savetxt("%s/lst_fwcount.txt" % save,lst_fwcount)
                np.savetxt("%s/lst_bwcount.txt" % save,lst_bwcount)
                data = np.column_stack((lst_time_cumsum, lst_cost))  # combine into 2D array
                np.savetxt("%s/lst_timecost.txt" % save,data)
                data = np.column_stack((lst_time_cumsum, lst_psnr))  # combine into 2D array
                np.savetxt("%s/lst_timepsnr.txt" % save,data)
            else:
                Dict = {'lst_time':lst_time_cumsum,'lst_cost':lst_cost,'lst_iter_diff':lst_iter_diff,'lst_iter_diff_norm':lst_iter_diff_normXini,\
                        'lst_fwcount':lst_fwcount,'lst_bwcount':lst_bwcount}
                scipy.io.savemat("%s/Results.mat" % save,Dict)
                np.savetxt("%s/lst_time.txt" % save,lst_time_cumsum)
                np.savetxt("%s/lst_cost.txt" % save,lst_cost)
                np.savetxt("%s/lst_iter_diff.txt" % save,lst_iter_diff)
                np.savetxt("%s/lst_fwcount.txt" % save,lst_fwcount)
                np.savetxt("%s/lst_bwcount.txt" % save,lst_bwcount)
                np.savetxt("%s/lst_iter_diff_norm.txt" % save,lst_iter_diff_normXini)
                data = np.column_stack((lst_time_cumsum, lst_cost))  # combine into 2D array
                np.savetxt("%s/lst_timecost.txt" % save,data)
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return torch.squeeze(x).cpu().numpy(),lst_psnr,lst_ssim,np.cumsum(lst_time),lst_cost,lst_iter_diff,lst_iter_diff_normXini


def ISTAEnergyGrad(num_iters,Ax,ATx,b,ub=np.inf,x_ini=None,g_theta=None,scale=1,lamda=0.1,\
                   alpha=0.1,beta=2,isLinearSearch = False,MaxInnerIter = 4,save=None,\
                    original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with ISTA gradient driven denoiser:
  .. math:
    min_x 0.5*|| A x - b ||^2 +lamda g_theta(x)
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    Ax: forward model.
    ATx: adjoint of forward model
    b (Array): Measurement.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
    """
    AHb = ATx(b)
    if x_ini is None:
        x = AHb.to(device)
    else:
        x = x_ini
    if not (ub==np.inf):
        x = project(x,ub)
    x_ini_norm = torch.norm(x)**2
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_cost = []
    lst_iter_diff = []
    lst_iter_diff_normXini = []
    lst_fwcount = []
    lst_bwcount = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="ISTA - GD Denoiser", \
                    leave=True)
     
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))

    cost_pre = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
    lst_cost.append(cost_pre.cpu().numpy())

    x_old = x
    gamma = 0.45
    for k in range(num_iters):
        count_fw = 0
        count_bw = 0
        start_time = time.perf_counter()
        if alpha<1e-8:
            break
        grad_1 = lamda*EnergyCNNMagGrad(x_old,scale,g_theta)
        if isLinearSearch:
            cost_pre = 0.5*torch.norm(Ax(x_old)-b)**2+lamda*CostCNNMagGrad(x_old,scale,g_theta)
            count_fw = count_fw+1
            v_k = x_old - alpha*grad_1
            while True:
                Ax_red = lambda xx: alpha*ATx(Ax(xx))+xx
                RHS = alpha*AHb+v_k
                if ub==np.inf:
                    x,iter_CG = CG_Alg_Handle(x_old,RHS,Ax_red,MaxInnerIter,tol=1e-6)  
                    count_fw = count_fw+iter_CG/2
                    count_bw = count_bw+iter_CG/2
                else:
                    x,_,iter_PGD = PGD_Alg_Handle(x_old,RHS,Ax_red,MaxInnerIter,alpha=1/(alpha+1),ub=ub,tol=1e-6)
                    count_fw = count_fw+iter_PGD/2
                    count_bw = count_bw+iter_PGD/2

                cost_temp = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
                count_fw = count_fw+1

                if cost_pre-cost_temp< (gamma/alpha)*torch.norm(x-x_old)**2:
                    alpha = alpha/beta
                    if alpha<1e-8:
                        break
                    v_k = x_old - alpha*grad_1
                else:
                    break
        else:
            Ax_red = lambda xx: alpha*ATx(Ax(xx))+xx
            v_k = x_old - alpha*grad_1
            RHS = alpha*AHb+v_k
            if ub==np.inf:
                x,iter_CG = CG_Alg_Handle(x_old,RHS,Ax_red,MaxInnerIter,tol=1e-6)  
                count_fw = count_fw+iter_CG/2
                count_bw = count_bw+iter_CG/2
            else:
                x,_,iter_PGD = PGD_Alg_Handle(x_old,RHS,Ax_red,MaxInnerIter,alpha=1/(alpha+1),ub=ub,tol=1e-6)
                count_fw = count_fw+iter_PGD/2
                count_bw = count_bw+iter_PGD/2
        temp_diff = x-x_old
        x_old = x
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        lst_iter_diff_normXini.append((torch.norm(temp_diff)**2/x_ini_norm).cpu().numpy())
        lst_iter_diff.append((torch.norm(temp_diff)**2).cpu().numpy())
        lst_fwcount.append(count_fw)
        lst_bwcount.append(count_bw)

        if not isLinearSearch:
            cost_temp = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
        lst_cost.append(cost_temp.cpu().numpy())


        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))

        if save != None:
            if SaveIter:
                Dict_iter = {'im':torch.squeeze(x).cpu().numpy()} 
                scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_iter)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()

    if save != None:
        lst_time_cumsum = np.cumsum(lst_time)
        if original is not None:
            Dict = {'lst_time':lst_time_cumsum,'lst_cost':lst_cost,'lst_iter_diff':lst_iter_diff,\
                    'lst_iter_diff_norm':lst_iter_diff_normXini,'lst_psnr':lst_psnr,'lst_ssim':lst_ssim,\
                        'lst_fwcount':lst_fwcount,'lst_bwcount':lst_bwcount}
            scipy.io.savemat("%s/Results.mat" % save,Dict)
            np.savetxt("%s/lst_time.txt" % save,lst_time_cumsum)
            np.savetxt("%s/lst_cost.txt" % save,lst_cost)
            np.savetxt("%s/lst_iter_diff.txt" % save,lst_iter_diff)
            np.savetxt("%s/lst_iter_diff_norm.txt" % save,lst_iter_diff_normXini)
            np.savetxt("%s/lst_psnr.txt" % save,lst_psnr)
            np.savetxt("%s/lst_ssim.txt" % save,lst_ssim)
            np.savetxt("%s/lst_fwcount.txt" % save,lst_fwcount)
            np.savetxt("%s/lst_bwcount.txt" % save,lst_bwcount)
            data = np.column_stack((lst_time_cumsum, lst_cost))  # combine into 2D array
            np.savetxt("%s/lst_timecost.txt" % save,data)
            data = np.column_stack((lst_time_cumsum, lst_psnr))  # combine into 2D array
            np.savetxt("%s/lst_timepsnr.txt" % save,data)
        else:
            Dict = {'lst_time':lst_time_cumsum,'lst_cost':lst_cost,'lst_iter_diff':lst_iter_diff,'lst_iter_diff_norm':lst_iter_diff_normXini,\
                    'lst_fwcount':lst_fwcount,'lst_bwcount':lst_bwcount}
            scipy.io.savemat("%s/Results.mat" % save,Dict)
            np.savetxt("%s/lst_time.txt" % save,lst_time_cumsum)
            np.savetxt("%s/lst_fwcount.txt" % save,lst_fwcount)
            np.savetxt("%s/lst_bwcount.txt" % save,lst_bwcount)
            np.savetxt("%s/lst_cost.txt" % save,lst_cost)
            np.savetxt("%s/lst_iter_diff.txt" % save,lst_iter_diff)
            np.savetxt("%s/lst_iter_diff_norm.txt" % save,lst_iter_diff_normXini)
            data = np.column_stack((lst_time_cumsum, lst_cost))  # combine into 2D array
            np.savetxt("%s/lst_timecost.txt" % save,data)
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return torch.squeeze(x).cpu().numpy(),lst_psnr,lst_ssim,np.cumsum(lst_time),lst_cost,lst_iter_diff,lst_iter_diff_normXini


def AccISTAEnergyGrad_LL(num_iters,Ax,ATx,b,ub=np.inf,x_ini=None,g_theta=None,scale=1,lamda=0.1,alpha=0.1,beta=2,\
                        MaxInnerIter = 4,save=None,original=None,SaveIter=False,verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with ISTA gradient-driven denoiser: 
  with convergence guarantee
  .. math:
    min_x 0.5*|| A x - b ||^2 +lamda g_theta(x)
    
    Implement:

    Huan Li et al. Accelerated Proximal Gradient Methods for Nonconvex Programming, NIPS.

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
    if x_ini is None:
        x = AHb.to(device)
    else:
        x = x_ini
    if not (ub==np.inf):
        x = project(x,ub)
    x_ini_norm = torch.norm(x)**2
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_cost = []
    lst_iter_diff = []
    lst_iter_diff_normXini = []
    lst_fwcount = []
    lst_bwcount = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="FISTA LL - GD Denoiser", \
                    leave=True)
     
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))

    cost_pre = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
    lst_cost.append(cost_pre.cpu().numpy())

    
    x_old = x
    z = x
    t_k_pre = 0
    t_k_post = 1
    eta_acc = 0.5
    delta_cond = 0.1
    q_k = 1
    c_k = cost_pre
    temp_diff = x-x_old
    for k in range(num_iters):
        count_fw = 0
        count_bw = 0
        start_time = time.perf_counter()
        y = x+(t_k_pre/t_k_post)*(z-x)+((t_k_pre-1)/t_k_post)*temp_diff
        x_old = x
        t_k_pre = t_k_post
        # call denoiser for grad_2
        grad_1 = lamda*EnergyCNNMagGrad(y,scale,g_theta)

        v_k = y-alpha*grad_1
        Ax_red = lambda xx:alpha*ATx(Ax(xx))+xx
        RHS = alpha*AHb+v_k
        if ub==np.inf:
            z,iter_CG = CG_Alg_Handle(y,RHS,Ax_red,MaxInnerIter,tol=1e-6)
            count_fw = count_fw+iter_CG/2
            count_bw = count_bw+iter_CG/2
        else:
            z,_,iter_PGD = PGD_Alg_Handle(y,RHS,Ax_red,MaxInnerIter,alpha=1/(alpha+1),beta=beta,ub=ub,tol=1e-6)
            count_fw = count_fw+iter_PGD/2
            count_bw = count_bw+iter_PGD/2
        

        cost_temp_z = 0.5*torch.norm(Ax(z)-b)**2+lamda*CostCNNMagGrad(z,scale,g_theta)
        count_fw = count_fw+1

        if cost_temp_z<=c_k-delta_cond*torch.norm(z-y)**2:
            x = z
        else:
            grad_1 = lamda*EnergyCNNMagGrad(x,scale,g_theta)
            v_k = x-alpha*grad_1
            RHS = alpha*AHb+v_k
            if ub==np.inf:
                temp_v,iter_CG = CG_Alg_Handle(x,RHS,Ax_red,MaxInnerIter,tol=1e-6)
                count_fw = count_fw+iter_CG/2
                count_bw = count_bw+iter_CG/2
            else:
                temp_v,_,iter_PGD = PGD_Alg_Handle(x,RHS,Ax_red,MaxInnerIter,alpha=1/(alpha+1),beta=beta,ub=ub,tol=1e-6)
                count_fw = count_fw+iter_PGD/2
                count_bw = count_bw+iter_PGD/2

            cost_temp_v = 0.5*torch.norm(Ax(temp_v)-b)**2+lamda*CostCNNMagGrad(temp_v,scale,g_theta)
            count_fw = count_fw+1

            if cost_temp_z<=cost_temp_v:
                x = z
            else:
                x = temp_v
        temp_diff = x-x_old
        t_k_post = (np.sqrt(4*t_k_post**2+1)+1)/2
        q_k_1 = eta_acc*q_k+1

        cost_temp_x = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
        count_fw = count_fw+1

        c_k = (eta_acc*q_k*c_k+cost_temp_x)/q_k_1
        q_k = q_k_1
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        lst_iter_diff_normXini.append((torch.norm(temp_diff)**2/x_ini_norm).cpu().numpy())
        lst_iter_diff.append((torch.norm(temp_diff)**2).cpu().numpy())
        lst_fwcount.append(count_fw)
        lst_bwcount.append(count_bw)

        cost_pre = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
        lst_cost.append(cost_pre.cpu().numpy())

        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
        
        if save != None:
            if SaveIter:
                Dict_iter = {'im':torch.squeeze(x).cpu().numpy()} 
                scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_iter)
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()
    if save != None:
        lst_time_cumsum = np.cumsum(lst_time)
        if original is not None:
            Dict = {'lst_time':lst_time_cumsum,'lst_cost':lst_cost,'lst_iter_diff':lst_iter_diff,\
                    'lst_iter_diff_norm':lst_iter_diff_normXini,'lst_psnr':lst_psnr,'lst_ssim':lst_ssim,\
                   'lst_fwcount':lst_fwcount,'lst_bwcount':lst_bwcount}
            scipy.io.savemat("%s/Results.mat" % save,Dict)
            np.savetxt("%s/lst_time.txt" % save,lst_time_cumsum)
            np.savetxt("%s/lst_cost.txt" % save,lst_cost)
            np.savetxt("%s/lst_iter_diff.txt" % save,lst_iter_diff)
            np.savetxt("%s/lst_iter_diff_norm.txt" % save,lst_iter_diff_normXini)
            np.savetxt("%s/lst_psnr.txt" % save,lst_psnr)
            np.savetxt("%s/lst_ssim.txt" % save,lst_ssim)
            np.savetxt("%s/lst_fwcount.txt" % save,lst_fwcount)
            np.savetxt("%s/lst_bwcount.txt" % save,lst_bwcount)
            data = np.column_stack((lst_time_cumsum, lst_cost))  # combine into 2D array
            np.savetxt("%s/lst_timecost.txt" % save,data)
            data = np.column_stack((lst_time_cumsum, lst_psnr))  # combine into 2D array
            np.savetxt("%s/lst_timepsnr.txt" % save,data)
        else:
            Dict = {'lst_time':lst_time_cumsum,'lst_cost':lst_cost,'lst_iter_diff':lst_iter_diff,'lst_iter_diff_norm':lst_iter_diff_normXini,\
                   'lst_fwcount':lst_fwcount,'lst_bwcount':lst_bwcount}
            scipy.io.savemat("%s/Results.mat" % save,Dict)
            np.savetxt("%s/lst_time.txt" % save,lst_time_cumsum)
            np.savetxt("%s/lst_cost.txt" % save,lst_cost)
            np.savetxt("%s/lst_iter_diff.txt" % save,lst_iter_diff)
            np.savetxt("%s/lst_fwcount.txt" % save,lst_fwcount)
            np.savetxt("%s/lst_bwcount.txt" % save,lst_bwcount)
            np.savetxt("%s/lst_iter_diff_norm.txt" % save,lst_iter_diff_normXini)
            data = np.column_stack((lst_time_cumsum, lst_cost))  # combine into 2D array
            np.savetxt("%s/lst_timecost.txt" % save,data)
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return torch.squeeze(x).cpu().numpy(),lst_psnr,lst_ssim,np.cumsum(lst_time),lst_cost,lst_iter_diff,lst_iter_diff_normXini


def QNPEnergyGrad(num_iters,Ax,ATx,b,ub=np.inf,x_ini=None,g_theta = None,scale=None,L=10,lamda=0.1,beta=2,a_k=1,isLinearSearch = False,\
                     Hessian_Type = 'SR1',gamma=1.6, MaxInnerIter = 4,save=None,original=None,SaveIter=False,\
                        verbose = True,device='cpu'):
    """
  Solve the MRI Reco. with QNP gradient-driven denoiser:
  .. math:
    min_x 0.5*|| A x - b ||^2 +lamda g_theta(x)
    
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
    if x_ini is None:
        x = AHb.to(device)#torch.zeros_like(AHb)#
    else:
        x = x_ini
    if not (ub==np.inf):
        x = project(x,ub)
    x_ini_norm = torch.norm(x)**2
    lst_time  = []
    lst_psnr = []
    lst_ssim = []
    lst_cost = []
    lst_iter_diff = []
    lst_iter_diff_normXini = []
    lst_fwcount = []
    lst_bwcount = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="QNP - GD Denoiser", \
                    leave=True)
    lst_time.append(0)
    lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
    lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))

    cost_1_pre = lamda*CostCNNMagGrad(x,scale,g_theta)
    cost_2_pre = 0.5*torch.norm(Ax(x)-b)**2
    cost = cost_1_pre+cost_2_pre
    lst_cost.append(cost.cpu().numpy())

    epsi = 1e-8
    #mag_par = 1.414
    if Hessian_Type == 'Modified-SR1':
        # parameters for MOdified-SR1 to bounded the Hessian matrix
        eta = 1
        theta_1 = 2e-6
        theta_2 = 2e2
        beta_stepsize = 0.01
        MaxIterBeta = 100
    #x_old = x
    for k in range(num_iters):
        count_fw = 0
        count_bw = 0
        start_time = time.perf_counter()

        gr = lamda*EnergyCNNMagGrad(x,scale,g_theta)
        # formulate Bx and Binvx
        if k==0:
            #Binvx = lambda xx: xx/L
            Bx = lambda xx: L*xx
            sigma_max = L
            sigma_min = L
            gr_old = gr
            x_old = x
        else:
            y_k = gr-gr_old
            s_k = x-x_old
            x_old = x
            gr_old = gr
            if Hessian_Type == 'SR1':
                # formulate B
                # tauBB is B^{-1} of BB step
                y_k_dot = torch.real(torch.vdot(y_k.flatten(),y_k.flatten()))
                tau_BB = y_k_dot/torch.real(torch.vdot(y_k.flatten(),s_k.flatten()))
                if tau_BB<=0: 
                    Bx = lambda xx: L*xx
                    sigma_max = L
                    sigma_min = L
                else:
                    D = gamma*tau_BB
                    temp_1 = y_k-D*s_k
                    temp_2 = torch.real(torch.vdot(s_k.flatten(),temp_1.flatten()))
                    if temp_2<=epsi*torch.norm(s_k)*torch.norm(temp_1):
                        #Binvx = lambda xx: (1/D)*xx
                        Bx = lambda xx: D*xx
                        sigma_max = D
                        sigma_min = D
                    else:
                        u = temp_1
                        Bx = lambda xx: D*xx+(torch.vdot(u.flatten(),xx.flatten())/temp_2)*u 
                        sigma_max = D+torch.vdot(u.flatten(),u.flatten())/temp_2
                        sigma_min = D

            elif Hessian_Type == 'Modified-SR1':
                beta_hess = 0
                for _ in range(MaxIterBeta):
                    v_k = beta_hess*s_k+((1-beta_hess)*eta)*y_k
                    v_k_s_k = torch.real(torch.vdot(v_k.flatten(),s_k.flatten()))
                    s_k_s_k = torch.real(torch.vdot(s_k.flatten(),s_k.flatten()))
                    v_k_v_k = torch.real(torch.vdot(v_k.flatten(),v_k.flatten()))
                    if (v_k_s_k/s_k_s_k>=theta_1) and (v_k_v_k/v_k_s_k<=theta_2):
                        break
                    else:
                        beta_hess = beta_hess+beta_stepsize
                tau_temp = s_k_s_k/v_k_s_k

                tau = tau_temp-torch.sqrt(tau_temp**2-s_k_s_k/v_k_v_k)# looks + will be better
                if tau<=0:
                    Bx = lambda xx: L*xx
                    sigma_max = L
                    sigma_min = L
                else:
                    H_0 = tau
                    rho = v_k_s_k-H_0*v_k_v_k
                    u = s_k-H_0*v_k
                    if rho<=epsi*torch.norm(u)*torch.norm(v_k):
                        Bx = lambda xx: 1/H_0*xx
                        sigma_max = 1/H_0
                        sigma_min = 1/H_0
                    else:
                        u_u = torch.real(torch.vdot(u.flatten(),u.flatten()))
                        temp_deno = 1/(H_0**2*rho+H_0*u_u)
                        Bx = lambda xx: 1/H_0*xx-(temp_deno*torch.vdot(u.flatten(),xx.flatten()))*u
                        sigma_max = 1/H_0
                        sigma_min = 1/H_0-temp_deno*u_u
        alpha = a_k
        if isLinearSearch:
            cost_1_pre = lamda*CostCNNMagGrad(x_old,scale,g_theta)
            while True:
                Ax_red = lambda xx: alpha*ATx(Ax(xx))+Bx(xx)
                RHS = Bx(x_old)+alpha*(AHb-gr)
                if ub==np.inf:
                    x,iter_CG = CG_Alg_Handle(x_old,RHS,Ax_red,MaxInnerIter,tol=1e-6)
                    count_fw = count_fw + iter_CG/2
                    count_bw = count_bw + iter_CG/2
                else:
                    #cost_fun = lambda xx:torch.real(alpha*torch.vdot(gr.flatten(),(xx-x_old).flatten()))+0.5*torch.real(torch.vdot((xx-x_old).flatten(),Bx(xx-x_old).flatten()))+(0.5*alpha)*torch.norm(Ax(xx)-b)**2
                    x,_,iter_PGD = PGD_Alg_Handle(x_old,RHS,Ax_red,MaxInnerIter,alpha=1/(sigma_max+alpha),mu = sigma_min,isStrong=True,beta=beta,ub=ub,tol=1e-6)
                    count_fw = count_fw + iter_PGD/2
                    count_bw = count_bw + iter_PGD/2
                cost_1_curr = lamda*CostCNNMagGrad(x,scale,g_theta)
                x_diff = x-x_old
                if cost_1_curr<=cost_1_pre+torch.real(torch.vdot(x_diff.flatten(),gr.flatten()))+(0.5/alpha)*torch.real(torch.vdot(x_diff.flatten(),Bx(x_diff).flatten())):
                    break
                else:
                    if alpha<1e-8:
                        break
                    else:
                        alpha = alpha/beta
        else:
            Ax_red = lambda xx: alpha*ATx(Ax(xx))+Bx(xx)
            RHS = Bx(x_old)+alpha*(AHb-gr)
            if ub==np.inf:
                x,iter_CG = CG_Alg_Handle(x_old,RHS,Ax_red,MaxInnerIter,tol=1e-6)
                count_fw = count_fw + iter_CG/2
                count_bw = count_bw + iter_CG/2
            else:
                #cost_fun = lambda xx:torch.real(alpha*torch.vdot(gr.flatten(),(xx-x_old).flatten()))+0.5*torch.real(torch.vdot((xx-x_old).flatten(),Bx(xx-x_old).flatten()))+(0.5*alpha)*torch.norm(Ax(xx)-b)**2
                x,_,iter_PGD = PGD_Alg_Handle(x_old,RHS,Ax_red,MaxInnerIter,alpha=1/(sigma_max+alpha),mu = sigma_min,isStrong=True,beta=beta,ub=ub,tol=1e-6)
                count_fw = count_fw + iter_PGD/2
                count_bw = count_bw + iter_PGD/2
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        lst_fwcount.append(count_fw)
        lst_bwcount.append(count_bw)

        cost = 0.5*torch.norm(Ax(x)-b)**2+lamda*CostCNNMagGrad(x,scale,g_theta)
        lst_cost.append(cost.cpu().numpy())
        lst_iter_diff_normXini.append((torch.norm(x-x_old)**2/x_ini_norm).cpu().numpy())
        lst_iter_diff.append((torch.norm(x-x_old)**2).cpu().numpy())
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
            lst_ssim.append(SSIM(np.abs(original),torch.squeeze(torch.abs(x)).cpu().numpy()))
       
        if save != None:
            if SaveIter:
                Dict_iter = {'im':torch.squeeze(x).cpu().numpy()} 
                scipy.io.savemat("%s/iter_%03d.mat" % (save, k),Dict_iter)
            
        if verbose:
            pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
            pbar.update()
            pbar.refresh()

    if save != None:
        lst_time_cumsum = np.cumsum(lst_time)
        if original is not None:
            Dict = {'lst_time':lst_time_cumsum,'lst_cost':lst_cost,'lst_iter_diff':lst_iter_diff,'lst_iter_diff_norm':lst_iter_diff_normXini,\
                    'lst_psnr':lst_psnr,'lst_ssim':lst_ssim,\
                         'lst_fwcount':lst_fwcount,'lst_bwcount':lst_bwcount}
            scipy.io.savemat("%s/Results.mat" % save,Dict)
            np.savetxt("%s/lst_time.txt" % save,lst_time_cumsum)
            np.savetxt("%s/lst_cost.txt" % save,lst_cost)
            np.savetxt("%s/lst_iter_diff.txt" % save,lst_iter_diff)
            np.savetxt("%s/lst_iter_diff_norm.txt" % save,lst_iter_diff_normXini)
            np.savetxt("%s/lst_psnr.txt" % save,lst_psnr)
            np.savetxt("%s/lst_ssim.txt" % save,lst_ssim)
            np.savetxt("%s/lst_fwcount.txt" % save,lst_fwcount)
            np.savetxt("%s/lst_bwcount.txt" % save,lst_bwcount)
            data = np.column_stack((lst_time_cumsum, lst_cost))  # combine into 2D array
            np.savetxt("%s/lst_timecost.txt" % save,data)
            data = np.column_stack((lst_time_cumsum, lst_psnr))  # combine into 2D array
            np.savetxt("%s/lst_timepsnr.txt" % save,data)
        else:
            Dict = {'lst_time':lst_time_cumsum,'lst_cost':lst_cost,'lst_iter_diff':lst_iter_diff,'lst_iter_diff_norm':lst_iter_diff_normXini,\
                     'lst_fwcount':lst_fwcount,'lst_bwcount':lst_bwcount}
            scipy.io.savemat("%s/Results.mat" % save,Dict)
            np.savetxt("%s/lst_time.txt" % save,lst_time_cumsum)
            np.savetxt("%s/lst_cost.txt" % save,lst_cost)
            np.savetxt("%s/lst_iter_diff.txt" % save,lst_iter_diff)
            np.savetxt("%s/lst_fwcount.txt" % save,lst_fwcount)
            np.savetxt("%s/lst_bwcount.txt" % save,lst_bwcount)
            np.savetxt("%s/lst_iter_diff_norm.txt" % save,lst_iter_diff_normXini)
            data = np.column_stack((lst_time_cumsum, lst_cost))  # combine into 2D array
            np.savetxt("%s/lst_timecost.txt" % save,data)
    if verbose:
        pbar.set_postfix(psnr="%0.5f%%" % lst_psnr[-1])
        pbar.close()
    return torch.squeeze(x).cpu().numpy(),lst_psnr,lst_ssim,np.cumsum(lst_time),lst_cost,lst_iter_diff,lst_iter_diff_normXini

