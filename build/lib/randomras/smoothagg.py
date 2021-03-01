#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:06:04 2021

@author: quentin
"""
import torch
from torch.nn import Module
from torch.autograd import Function
#from torch.distributions.transforms import SigmoidTransform

class randomArgmax(Function):
    
    @staticmethod
    def forward(ctx,z,nb_samples = 1,noise_intensity = 1e-1, noise_type ="gaussian"):
      device = z.device
      z_size = z.size()
      noise_dict ={"gaussian": torch.tensor(0),"gumbel": torch.tensor(1), "cauchy":torch.tensor(2)}
      noise_type = noise_dict[noise_type]
      if noise_type == noise_dict["gaussian"]:
        noise = torch.normal(mean = torch.zeros((nb_samples,z_size[0],z_size[1],z_size[2],z_size[3]),device=device),std = 1. )
      elif noise_type == noise_dict["gumbel"]:
        m = torch.distributions.gumbel.Gumbel(torch.tensor([0.]).to(device=device), torch.tensor([1.]).to(device=device))
        noise = m.sample((nb_samples,z_size[0],z_size[1],z_size[2],z_size[3])).squeeze(-1)
      elif noise_type == noise_dict["cauchy"]:
        m = torch.distributions.cauchy.Cauchy(torch.tensor([0.]).to(device=device), torch.tensor([1.]).to(device=device))
        noise = torch.clamp(m.sample((nb_samples,z_size[0],z_size[1],z_size[2],z_size[3])).squeeze(-1),min=-1e7, max=1e7)
      else:
        print("noise type not implemented")
      z_pert = z + noise_intensity*noise
      _, indices = torch.max(z_pert, dim =-1, keepdim=True)
      weights = torch.zeros(z_pert.size(), device = device)
      weights.scatter_(-1, indices, 1)
      _, indices = torch.max(z, dim =-1, keepdim=True)
      vr_var = torch.zeros(z.size(), device = device)
      vr_var.scatter_(-1, indices, 1) #used during backward to reduce variance of gradient estimator
      ctx.save_for_backward(weights,noise,torch.tensor(noise_intensity),vr_var,noise_type)
      weight = weights.mean(dim = 0)
      return weight
    
    @staticmethod
    def backward(ctx, grad_l):
      grad_z = None
      weights, noise, noise_intensity,vr_var, noise_type = ctx.saved_tensors
      noise_dict ={"gaussian": torch.tensor(0),"gumbel":torch.tensor(1),"cauchy":torch.tensor(2)}
      if noise_type == noise_dict["gaussian"]:
        grad_z = torch.matmul(grad_l.repeat(noise.size()[0],1,1,1,1).unsqueeze(-2),weights.unsqueeze(-1)-vr_var.unsqueeze(0).repeat(weights.size()[0],1,1,1,1).unsqueeze(-1))
        grad_z = torch.matmul(grad_z,noise.unsqueeze(-2))/noise_intensity
        grad_z = grad_z.squeeze(-2)
      elif noise_type == noise_dict["cauchy"]:
        grad_z = torch.matmul(grad_l.repeat(noise.size()[0],1,1,1,1).unsqueeze(-2),weights.unsqueeze(-1)-vr_var.unsqueeze(0).repeat(weights.size()[0],1,1,1,1).unsqueeze(-1))
        grad_z = torch.matmul(grad_z,(2*noise/(1.+torch.square(noise))).unsqueeze(-2))/noise_intensity #need to replace with grad of density
        grad_z = grad_z.squeeze(-2)
      else:
        print("noise_type not implemented")
      grad_z = grad_z.mean(dim=0)
      return grad_z, None, None, None

class SmoothAggBase(Module):
    
    def __init__(self,
                 gamma,
                 alpha,
                 eps):
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps # Weight for background color
    
    def update_smoothing(self, gamma = 4e-2, alpha = 1.):
        self.gamma = gamma
        self.alpha = alpha
        
        
class SoftAgg(SmoothAggBase):
    
    def __init__(self,
                 gamma = 4e-2,
                 alpha = 1.,
                 eps= 1e-10):
        super().__init__(gamma,alpha,eps)
        
    def aggregate(self, zbuf,zfar,znear,prob_map,mask):
        device =zbuf.device
        z_inv = (zfar - zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=self.eps)
        #z_map = ((1./self.alpha)*torch.log(1e-12+prob_map)+ z_inv)/ self.gamma # substract z_inv_max ?
        #z_map = ((1./self.alpha)*torch.log(1e-12+prob_map)+ z_inv- z_inv_max)/ self.gamma
        z_map = ((self.gamma/self.alpha)*torch.log(prob_map)+ z_inv- z_inv_max)
        #z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps / self.gamma),dim=-1)
        z_map =torch.cat((z_map,(torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps -z_inv_max)),dim=-1)
        weights = torch.softmax(z_map/self.gamma,dim=-1)
        
        # z_inv = (zfar - zbuf) / (zfar - znear) * mask
        # # pyre-fixme[16]: `Tuple` has no attribute `values`.
        # # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        # z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=self.eps)
        # # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        # #weights_num = prob_map * torch.exp((z_inv - z_inv_max) / self.gamma)
        # weights_num = torch.exp(torch.log(prob_map)+(z_inv - z_inv_max) / self.gamma)
    
        # # Also apply exp normalize trick for the background color weight.
        # # Clamp to ensure delta is never 0.
        # # pyre-fixme[20]: Argument `max` expected.
        # # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        # delta = torch.exp((self.eps - z_inv_max) / self.gamma).clamp(min=self.eps)
    
        # # Normalize weights.
        # # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
        # denom = weights_num.sum(dim=-1)[..., None] + delta
        # weights = torch.cat((weights_num,delta),dim=-1)/denom
        # # Sum: weights * textures + background color
        # #weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)/denom
        # #weighted_background = delta * background/denom
        return weights
    
    
class GaussianAgg(SmoothAggBase):
    
    def __init__(self,
                 nb_samples=16,
                 gamma = 4e-2,
                 alpha = 1.,
                 eps= 1e-10):
        self.nb_samples = nb_samples
        super().__init__(gamma,alpha,eps)
        
    def aggregate(self,zbuf,zfar,znear,prob_map,mask):
        device =zbuf.device
        z_inv = (zfar - zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=self.eps)
        z_map = ((self.gamma/self.alpha)*torch.log(1e-12+prob_map)+ z_inv-z_inv_max) # substract z_inv_max ?
        z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps-z_inv_max),dim=-1)
        randomarg = randomArgmax.apply
        randomax = randomarg(z_map, self.nb_samples, self.gamma, "gaussian")
        return randomax
    
    
class CauchyAgg(SmoothAggBase):
    
    def __init__(self,
                 nb_samples=16,
                 gamma = 4e-2,
                 alpha = 1.,
                 eps = 1e-10):
        self.nb_samples = nb_samples
        super().__init__(gamma,alpha,eps)
        
    def aggregate(self,zbuf,zfar,znear,prob_map,mask):
        device =zbuf.device
        z_inv = (zfar - zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=self.eps)
        #z_map = ((self.gamma/self.alpha)*torch.log(1e-12+prob_map)+ z_inv) # substract z_inv_max ?
        #z_map = ((self.gamma/self.alpha)*torch.log(prob_map.clamp(min=1e-12))+ z_inv-z_inv_max) # substract z_inv_max ? add 1e-12 to prob map ? 
        z_map = ((self.gamma/self.alpha)*log_corrected.apply(prob_map)+ z_inv-z_inv_max)
        z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps-z_inv_max ),dim=-1)
        #z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps ),dim=-1)
        randomarg = randomArgmax.apply
        randomax = randomarg(z_map, self.nb_samples, self.gamma, "cauchy")
        return randomax
    

class log_corrected(Function):    
    
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return x.log()
    
    @staticmethod
    def backward(ctx,grad_l):
        grad_log = None
        if ctx.needs_input_grad[0]:
            (x,) = ctx.saved_tensors
            device = x.device
            grad_log = torch.ones(x.size(),device= device)/x
            grad_log = grad_log*grad_l
            grad_log = torch.where(torch.isnan(grad_log), torch.zeros_like(grad_log), grad_log)
        return grad_log
    