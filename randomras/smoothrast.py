#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 09:56:29 2021

@author: quentin
"""
import torch
from torch.nn import Module
from torch.autograd import Function
import numpy as np
from torch.distributions.transforms import SigmoidTransform

class randomHeaviside(Function):
    
    @staticmethod
    def forward(ctx,distances,nb_samples = 1,noise_intensity = 1e-1, noise_type ="gaussian"):
      device = distances.device
      dist_size = distances.size()
      noise_dict ={"gaussian": torch.tensor(0),"cauchy": torch.tensor(1), "logistic": torch.tensor(2)}
      noise_type = noise_dict[noise_type]
      if noise_type == noise_dict["gaussian"]:
        noise = torch.normal(mean = torch.zeros((nb_samples,dist_size[0],dist_size[1],dist_size[2],dist_size[3]),device=device),std = 1. )
      elif noise_type == noise_dict["cauchy"]:
        m = torch.distributions.cauchy.Cauchy(torch.tensor([0.]).to(device=device), torch.tensor([1.]).to(device=device))
        noise = torch.clamp(m.sample((nb_samples,dist_size[0],dist_size[1],dist_size[2],dist_size[3])).squeeze(-1),min=-1e7, max=1e7) #we need to clamp the noise to avoid inf values
      elif noise_type == noise_dict["logistic"]:
        base_distribution = torch.distributions.uniform.Uniform(0, 1)
        transforms = [SigmoidTransform().inv]
        logistic = torch.distributions.transformed_distribution.TransformedDistribution(base_distribution, transforms)
        noise  = logistic.sample((nb_samples,dist_size[0],dist_size[1],dist_size[2],dist_size[3]))
      else:
        print("noise type not implemented")
      maps = distances + noise_intensity*noise
      maps = torch.heaviside(maps, values = torch.ones(maps.size(), device=device))
      vr_var = torch.heaviside(distances, values = torch.ones(distances.size(), device=device)) #used during backward to reduce variance of gradient estimator
      ctx.save_for_backward(maps,noise,torch.tensor(noise_intensity),vr_var,noise_type)
      map = maps.mean(dim=0)
      return map
    
    @staticmethod
    def backward(ctx, grad_l):
      grad_dist = None
      maps, noise, noise_intensity,vr_var, noise_type = ctx.saved_tensors
      noise_dict ={"gaussian": torch.tensor(0), "cauchy": torch.tensor(1), "logistic": torch.tensor(2) }
      if noise_type == noise_dict["gaussian"]:
        grad_maps = (maps - vr_var.unsqueeze(0).repeat(maps.size()[0],1,1,1,1)) * noise/noise_intensity
      elif noise_type == noise_dict["cauchy"]:
        grad_maps = (maps - vr_var.unsqueeze(0).repeat(maps.size()[0],1,1,1,1)) * ((2*noise)/(1+torch.square(noise)))/noise_intensity
      else:
        print("noise_type not implemented")
      grad_maps = grad_maps.mean(dim=0)
      if ctx.needs_input_grad[0]:
          grad_dist = grad_maps*grad_l
      return grad_dist, None, None, None


class SmoothRastBase(Module):
    
    def __init__(self,
                 sigma=2e-4):
        self.sigma = sigma
        
    def update_smoothing(self, sigma):
        self.sigma = sigma
    

class SoftRast(SmoothRastBase):
    
    def __init__(self,
                 sigma=2e-4):
        super().__init__(sigma)
    
    def rasterize(self,dists):
        prob_map = torch.sigmoid(-dists/self.sigma)
        return prob_map
    
class GaussianRast(SmoothRastBase):
    
    def __init__(self,
                 nb_samples=16,
                 sigma= 2e-4):
        self.nb_samples = nb_samples
        super().__init__(sigma)
    
    def rasterize(self,dists):
        randomheavi = randomHeaviside().apply
        prob_map = randomheavi(-dists, self.sigma, self.nb_samples)
        return prob_map

class ArctanRast(SmoothRastBase):
    def __init__(self,
                 nb_samples=16,
                 sigma= 2e-4):
        self.nb_samples = nb_samples
        super().__init__(sigma)
    
    def rasterize(self,dists):
        prob_map= torch.arctan(-dists/self.sigma)/np.pi + .5
        return prob_map
    
class AffineRast(SmoothRastBase):
    def __init__(self,
                 nb_samples=16,
                 sigma= 2e-4):
        self.nb_samples = nb_samples
        super().__init__(sigma)
    
    def rasterize(self,dists):
        prob_map = torch.where(-dists/self.sigma > .5, torch.ones_like(dists),-dists/self.sigma + .5)
        prob_map = torch.where(prob_map < 0., torch.zeros_like(prob_map),prob_map)
        return prob_map
    
class HardRast():
    def __init__(self):
        return
    
    def rasterize(self,dists):
        device = dists.device
        prob_map = torch.heaviside(-dists, values = torch.ones(dists.size(), device=device))
        return prob_map