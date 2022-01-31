"""
Inspired from Pytorch3D.
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
      ctx.save_for_backward(maps,noise,noise_intensity,vr_var,noise_type)
      map = maps.mean(dim=0)
      return map
    
    @staticmethod
    def backward(ctx, grad_l):
        grad_dist = None
        grad_sigma = None
        maps, noise, noise_intensity,vr_var, noise_type = ctx.saved_tensors
        noise_dict ={"gaussian": torch.tensor(0), "cauchy": torch.tensor(1), "logistic": torch.tensor(2) }
        if noise_type == noise_dict["gaussian"]:
          grad_maps = (maps - vr_var.unsqueeze(0).repeat(maps.size()[0],1,1,1,1)) * noise/noise_intensity
          grad_sigma = (maps - vr_var.unsqueeze(0).repeat(maps.size()[0],1,1,1,1))*(torch.square(noise) - 1.)/noise_intensity
        elif noise_type == noise_dict["cauchy"]:
          grad_maps = (maps - vr_var.unsqueeze(0).repeat(maps.size()[0],1,1,1,1)) * ((2*noise)/(1+torch.square(noise)))/noise_intensity
          grad_sigma = maps*(noise*((2*noise)/(1+torch.square(noise))) - 1.)/noise_intensity
        else:
          print("noise_type not implemented")
        grad_maps = grad_maps.mean(dim=0)
        grad_sigma = grad_sigma.mean(dim=0)
        if ctx.needs_input_grad[0]:
            grad_dist = grad_maps*grad_l
        if ctx.needs_input_grad[2]:
            grad_sigma = torch.sum(grad_maps*grad_l)
        return grad_dist, None, grad_sigma, None

class randomHeaviside_wovr(Function):
    
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
      ctx.save_for_backward(maps,noise,noise_intensity,vr_var,noise_type)
      map = maps.mean(dim=0)
      return map
    
    @staticmethod
    def backward(ctx, grad_l):
        grad_dist = None
        grad_sigma = None
        maps, noise, noise_intensity,vr_var, noise_type = ctx.saved_tensors
        noise_dict ={"gaussian": torch.tensor(0), "cauchy": torch.tensor(1), "logistic": torch.tensor(2) }
        if noise_type == noise_dict["gaussian"]:
          grad_maps = (maps) * noise/noise_intensity
          grad_sigma = maps*(torch.square(noise) - 1.)/noise_intensity
        elif noise_type == noise_dict["cauchy"]:
          grad_maps = (maps) * ((2*noise)/(1+torch.square(noise)))/noise_intensity
          grad_sigma = maps*(noise*((2*noise)/(1+torch.square(noise))) - 1.)/noise_intensity
        else:
          print("noise_type not implemented")
        grad_maps = grad_maps.mean(dim=0)
        grad_sigma = grad_sigma.mean(dim=0)
        if ctx.needs_input_grad[0]:
            grad_dist = grad_maps*grad_l
        if ctx.needs_input_grad[2]:
            grad_sigma = torch.sum(grad_maps*grad_l)
        return grad_dist, None, grad_sigma, None


class SmoothRastBase(Module):
    
    def __init__(self,
                 sigma=2e-4):
        super(SmoothRastBase,self).__init__()
        self.sigma = torch.tensor(sigma, requires_grad= True)
        self.nb_samples = 1
        
    def update_smoothing(self, sigma):
        self.sigma = torch.tensor(sigma, requires_grad= True)
        
    def update_nb_samples(self, nb_samples):
        self.nb_samples = nb_samples
    
    
class SoftRast(SmoothRastBase):
    
    def __init__(self,
                 sigma=2e-4):
        super(SoftRast,self).__init__(sigma)
    
    def rasterize(self,dists):
        prob_map = torch.sigmoid(-dists/self.sigma)
        return prob_map
    
class GaussianRast(SmoothRastBase):
    
    def __init__(self,
                 nb_samples=16,
                 sigma= 2e-4):
        super(GaussianRast,self).__init__(sigma)
        self.nb_samples = nb_samples
    
    def rasterize(self,dists):
        randomheavi = randomHeaviside().apply
        prob_map = randomheavi(-dists, self.nb_samples, self.sigma)
        return prob_map
    
class GaussianRast_wovr(SmoothRastBase):
    
    def __init__(self,
                 nb_samples=16,
                 sigma= 2e-4):
        super(GaussianRast_wovr,self).__init__(sigma)
        self.nb_samples = nb_samples
    
    def rasterize(self,dists):
        randomheavi = randomHeaviside_wovr().apply
        prob_map = randomheavi(-dists, self.nb_samples, self.sigma)
        return prob_map

class ArctanRast(SmoothRastBase):
    def __init__(self,
                 nb_samples=16,
                 sigma= 2e-4):
        super(ArctanRast,self).__init__(sigma)
        self.nb_samples = nb_samples
    
    def rasterize(self,dists):
        randomheavi = randomHeaviside().apply
        prob_map = randomheavi(-dists, self.nb_samples, self.sigma,"cauchy")
        #prob_map= torch.arctan(-dists/self.sigma)/np.pi + .5
        return prob_map
    
class AffineRast(SmoothRastBase):
    def __init__(self,
                 nb_samples=16,
                 sigma= 2e-4):
        super().__init__(sigma)
        self.nb_samples = nb_samples
    
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