#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:06:04 2021

@author: quentin
"""
import torch
from torch.nn import Module
from torch.autograd import Function

class randomArgmax(Function):
    
    @staticmethod
    def forward(ctx,z,nb_samples = 1,noise_intensity = 1e-1, noise_type ="gaussian", fixed_noise = False):
        device = z.device
        z_size = z.size()
        noise_dict ={"gaussian": torch.tensor(0),"gumbel": torch.tensor(1), "cauchy":torch.tensor(2), "uniform": torch.tensor(3)}
        noise_type = noise_dict[noise_type]
        if fixed_noise:
            torch.manual_seed(1)
        if noise_type == noise_dict["gaussian"]:
          noise = torch.normal(mean = torch.zeros((nb_samples,z_size[0],z_size[1],z_size[2],z_size[3]),device=device),std = 1. )
        elif noise_type == noise_dict["gumbel"]:
          m = torch.distributions.gumbel.Gumbel(torch.tensor([0.]).to(device=device), torch.tensor([1.]).to(device=device))
          noise = m.sample((nb_samples,z_size[0],z_size[1],z_size[2],z_size[3])).squeeze(-1)
        elif noise_type == noise_dict["cauchy"]:
          m = torch.distributions.cauchy.Cauchy(torch.tensor([0.]).to(device=device), torch.tensor([1.]).to(device=device))
          noise = torch.clamp(m.sample((nb_samples,z_size[0],z_size[1],z_size[2],z_size[3])).squeeze(-1),min=-1e7, max=1e7)
        elif noise_type == noise_dict["uniform"]:
          m = torch.distributions.uniform.Uniform(torch.tensor([-0.5]).to(device=device), torch.tensor([0.5]).to(device=device))
          noise = m.sample((nb_samples,z_size[0],z_size[1],z_size[2],z_size[3])).squeeze(-1)
        else:
          print("noise type not implemented")
        z_pert = z + noise_intensity*noise
        _, indices = torch.max(z_pert, dim =-1, keepdim=True)
        weights = torch.zeros(z_pert.size(), device = device)
        weights.scatter_(-1, indices, 1)
        _, indices = torch.max(z, dim =-1, keepdim=True)
        vr_var = torch.zeros(z.size(), device = device)
        vr_var.scatter_(-1, indices, 1) #used during backward to reduce variance of gradient estimator
        ctx.save_for_backward(weights,noise,noise_intensity,vr_var,noise_type)
        weight = weights.mean(dim = 0)
        return weight
    
    @staticmethod
    def backward(ctx, grad_l):
        grad_z = None
        grad_gamma= None
        weights, noise, noise_intensity,vr_var, noise_type = ctx.saved_tensors
        noise_dict ={"gaussian": torch.tensor(0),"gumbel":torch.tensor(1),"cauchy":torch.tensor(2), "uniform": torch.tensor(3)}
        if noise_type == noise_dict["gaussian"]:
          grad_z = torch.matmul(grad_l.repeat(noise.size()[0],1,1,1,1).unsqueeze(-2),weights.unsqueeze(-1)-vr_var.unsqueeze(0).repeat(weights.size()[0],1,1,1,1).unsqueeze(-1))
          grad_z = torch.matmul(grad_z,noise.unsqueeze(-2))/noise_intensity
          grad_z = grad_z.squeeze(-2)
          grad_gamma = weights*(torch.square(torch.norm(noise,dim=-1,keepdim=True))- 1.)/noise_intensity
          grad_gamma = grad_l*grad_gamma
          grad_gamma = grad_gamma.sum(dim=(1,2,3,4))
        elif noise_type == noise_dict["cauchy"]:
          grad_z = torch.matmul(grad_l.repeat(noise.size()[0],1,1,1,1).unsqueeze(-2),weights.unsqueeze(-1)-vr_var.unsqueeze(0).repeat(weights.size()[0],1,1,1,1).unsqueeze(-1))
          grad_z = torch.matmul(grad_z,(2*noise/(1.+torch.square(noise))).unsqueeze(-2))/noise_intensity #need to replace with grad of density
          grad_z = grad_z.squeeze(-2)
          grad_gamma = weights*(torch.matmul((2*noise/(1.+torch.square(noise))).unsqueeze(-2),noise.unsqueeze(-1)).squeeze(-1)- 1.)/noise_intensity
          grad_gamma = grad_l*grad_gamma
          grad_gamma = grad_gamma.sum(dim=(1,2,3,4))
        elif noise_type == noise_dict["uniform"]:
            print("noise_type not implemented")
        elif noise_type == noise_dict["gumbel"]:
            print("noise_type not implemented")
        else:
            print("noise_type not implemented")
        grad_z = grad_z.mean(dim=0)
        grad_gamma = grad_gamma.mean(dim=0)
        return grad_z, None, grad_gamma, None, None

class SmoothAggBase(Module):
    
    def __init__(self,
                 gamma,
                 alpha,
                 eps,
                 nb_samples=1):
        super(SmoothAggBase,self).__init__()
        self.gamma = torch.tensor(gamma,requires_grad= True)
        self.alpha = torch.tensor(alpha,requires_grad=True)
        self.nb_samples = nb_samples
        self.eps = eps # Weight for background color
    
    def update_smoothing(self, gamma = 4e-2, alpha = 1.):
        self.gamma = torch.tensor(gamma,requires_grad= True)
        self.alpha = torch.tensor(alpha,requires_grad=True)
        
    def update_nb_samples(self, nb_samples):
        self.nb_samples = nb_samples
        
class SoftAgg(SmoothAggBase):
    
    def __init__(self,
                 gamma = 4e-2,
                 alpha = 1.,
                 eps= 1e-10):
        super(SoftAgg,self).__init__(gamma,alpha,eps)
        
    def aggregate(self, zbuf,zfar,znear,prob_map,mask):
        device =zbuf.device
        #print(zfar.size(), zbuf.size(), znear.size(), mask.size())
        z_inv = (zfar - zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=self.eps)
        #prob_map.register_hook(lambda x: print("prob_map grad",torch.max(x),x[0,0:3,0:3,0:3]))
        #z_map = ((self.gamma/self.alpha)*torch.log(prob_map)+ z_inv- z_inv_max)
        log_prob = log_corrected.apply(prob_map)
        gal = self.gamma/self.alpha
        #gal.register_hook(lambda x: print("gal grad",torch.max(x)))
        z_map = (prod_corrected.apply(gal,log_prob)+ z_inv-z_inv_max)
        #z_map = ((self.gamma/self.alpha)*log_corrected.apply(prob_map).clamp(min=-1e12)+ z_inv- z_inv_max)
        #print(z_map.size())
        #print(z_map[0,0:3,0:3,0:3])
        #z_map.register_hook(lambda x: print("zmap2 grad",torch.max(x)))
        z_map =torch.cat((z_map,(torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps -z_inv_max)),dim=-1)    
        #z_map.register_hook(lambda x: print("zmap3 grad",torch.max(x)))
        weights = torch.softmax(prod_corrected.apply(1./self.gamma,z_map),dim=-1)
        return weights
    
    
class GaussianAgg(SmoothAggBase):
    
    def __init__(self,
                 nb_samples=16,
                 gamma = 4e-2,
                 alpha = 1.,
                 eps= 1e-10,
                 fixed_noise=False):
        super(GaussianAgg,self).__init__(gamma,alpha,eps,nb_samples)
        self.fixed_noise = fixed_noise
        
    def aggregate(self,zbuf,zfar,znear,prob_map,mask):
        device =zbuf.device
        z_inv = (zfar - zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=self.eps)
        #z_map = ((self.gamma/self.alpha)*log_corrected.apply(prob_map).clamp(min=-1e12)+ z_inv-z_inv_max)
        #prob_map.register_hook(lambda x: print("prob_map grad",torch.max(torch.abs(x))))
        log_prob = log_corrected.apply(prob_map)
        z_map = (prod_corrected.apply(self.gamma/self.alpha,log_prob)+ z_inv-z_inv_max)
        z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps-z_inv_max ),dim=-1)
        #z_map.register_hook(lambda x: print("zmap2 grad",torch.max(torch.abs(x))))
        randomarg = randomArgmax.apply
        randomax = randomarg(z_map, self.nb_samples, self.gamma, "gaussian", self.fixed_noise)
        #randomax.register_hook(lambda x: print("randomax grad",torch.max(torch.abs(x))))
        return randomax
    
    
class CauchyAgg(SmoothAggBase):
    
    def __init__(self,
                 nb_samples=16,
                 gamma = 4e-2,
                 alpha = 1.,
                 eps = 1e-10,
                 fixed_noise=False):
        super(CauchyAgg,self).__init__(gamma,alpha,eps,nb_samples)
        self.fixed_noise = fixed_noise
        
    def aggregate(self,zbuf,zfar,znear,prob_map,mask):
        device =zbuf.device
        z_inv = (zfar - zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=self.eps)
        #z_map = ((self.gamma/self.alpha)*torch.log(1e-12+prob_map)+ z_inv) # substract z_inv_max ?
        #z_map = ((self.gamma/self.alpha)*torch.log(prob_map.clamp(min=1e-12))+ z_inv-z_inv_max) # substract z_inv_max ? add 1e-12 to prob map ? 
        #prob_map.register_hook(lambda x: print("probmap grad",torch.max(x),x[0,0:3,0:3,0:3]))
        ####
        #gal = self.gamma/self.alpha
        #gal.register_hook(lambda x: print("gal grad",torch.max(x)))
        log_prob = log_corrected.apply(prob_map)
        #log_prob.register_hook(lambda x: print("log_prob grad",torch.max(x)))
        #mask_is_finite = torch.cat((torch.isfinite(log_prob),torch.ones((log_prob.size()[0],log_prob.size()[1],log_prob.size()[2],1),device=device, dtype=torch.bool)),dim = -1)
        #print("mask size",mask_is_finite.size())
        
        #z_map = (gal*log_prob.clamp(min=-1e12)+ z_inv-z_inv_max)
        z_map = (prod_corrected.apply(self.gamma/self.alpha,log_prob)+ z_inv-z_inv_max)
        ####
        #z_map = ((self.gamma/self.alpha)*log_corrected.apply(prob_map).clamp(min=-1e12)+ z_inv-z_inv_max)
        #add background component with inverse depth of eps
        #print(z_map.size())
        #print(z_map[0,0:3,0:3,0:3])
        #z_map.register_hook(lambda x: print("zmap2 grad",torch.max(x)))
        z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps-z_inv_max ),dim=-1)
        #z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps ),dim=-1)
        #z_map.register_hook(lambda x: print("zmap3 grad",torch.max(x)))
        randomarg = randomArgmax.apply
        randomax = randomarg(z_map, self.nb_samples, self.gamma, "cauchy", self.fixed_noise)
        #print("randomax", randomax.size())
        return randomax
    
class UniformAgg(SmoothAggBase):
    
    def __init__(self,
                 nb_samples=16,
                 gamma = 4e-2,
                 alpha = 1.,
                 eps = 1e-10,
                 fixed_noise=False):
        self.fixed_noise = fixed_noise
        super().__init__(gamma,alpha,eps, nb_samples)
        
    def aggregate(self,zbuf,zfar,znear,prob_map,mask):
        device =zbuf.device
        z_inv = (zfar - zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=self.eps)
        z_map = ((self.gamma/self.alpha)*log_corrected.apply(prob_map)+ z_inv-z_inv_max)
        z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps-z_inv_max ),dim=-1)
        randomarg = randomArgmax.apply
        randomax = randomarg(z_map, self.nb_samples, self.gamma, "uniform", self.fixed_noise)
        return randomax
    
    
class HardAgg():
    
    def __init__(self,eps=1e-10):
        self.eps = eps
        return
        
    def aggregate(self,zbuf,zfar,znear,prob_map,mask):
        device =zbuf.device
        z_inv = (zfar - zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=self.eps)
        z_map = ((1./1e6)*log_corrected.apply(prob_map)+ z_inv-z_inv_max)
        z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*self.eps-z_inv_max ),dim=-1)
        _, indices = torch.max(z_map, dim =-1, keepdim=True)
        weight = torch.zeros(z_map.size(), device = device)
        weight.scatter_(-1, indices, 1)
        return weight
    

class log_corrected(Function):    
    """
    logarithm whose backward pass returns 0 instead of nan when x is null and backward pass vector is null.
    """
    
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
            #print("x",x[torch.where(torch.isinf(grad_log))],x[torch.where(torch.isnan(grad_log))])
            grad_log = torch.where(torch.isinf(grad_log), torch.zeros_like(grad_log), grad_log)
            #print("grad log", torch.max(grad_log),"grad_l", torch.max(grad_l))
            grad_log = grad_log*grad_l
            #grad_log = torch.where(torch.isnan(grad_log), torch.zeros_like(grad_log), grad_log)
        return grad_log
    

class prod_corrected(Function):    
    """
    product whose backward pass returns 0 instead of nan when x is null and y is infty.
    """
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.save_for_backward(x,y)
        return x*y
    
    @staticmethod
    def backward(ctx,grad_l):
        grad_prod_x = None
        grad_prod_y = None
        (x,y) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            y = torch.where(torch.isinf(y), torch.zeros_like(y), y)
            grad_prod_x = y*grad_l
            grad_prod_x = grad_prod_x.nansum()
            #print("grad_x", grad_prod_x)
        if ctx.needs_input_grad[1]:
            device = x.device
            grad_prod_y = x*grad_l
            grad_prod_y = torch.where(torch.isnan(grad_prod_y), torch.zeros_like(grad_prod_y), grad_prod_y)
        return grad_prod_x, grad_prod_y
    
    
