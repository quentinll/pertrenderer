#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:11:24 2021

@author: quentin
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    BlendParams
)


class randomHeaviside(Function):
    
    @staticmethod
    def forward(ctx,distances,nb_samples = 1,noise_intensity = 1e-1, noise_type ="gaussian"):
      device = distances.device
      dist_size = distances.size()
      noise_dict ={"gaussian": torch.tensor(0)}
      noise_type = noise_dict[noise_type]
      if noise_type == noise_dict["gaussian"]:
        noise = torch.normal(mean = torch.zeros((nb_samples,dist_size[0],dist_size[1],dist_size[2],dist_size[3]),device=device),std = 1. )
      else:
        print("noise type not implemented")
      maps = distances + noise_intensity*noise
      maps = torch.heaviside(maps, values = torch.ones(maps.size(), device=device))
      ctx.save_for_backward(maps,noise,torch.tensor(noise_intensity),noise_type)
      map = maps.mean(dim=0)
      return map
    
    @staticmethod
    def backward(ctx, grad_l):
      '''
      Compute derivatives of the solution of the QCQP with respect to 
      '''
      grad_dist = None
      maps, noise, noise_intensity, noise_type = ctx.saved_tensors
      noise_dict ={"gaussian": torch.tensor(0)}
      if noise_type == noise_dict["gaussian"]:
        grad_maps = maps * noise/noise_intensity
      else:
        print("noise_type not implemented")
      grad_maps = grad_maps.mean(dim=0)
      if ctx.needs_input_grad[0]:
          grad_dist = grad_maps*grad_l
      return grad_dist, None, None, None

class randomArgmax(Function):
    
    @staticmethod
    def forward(ctx,z,nb_samples = 1,noise_intensity = 1e-1, noise_type ="gaussian"):
      device = z.device
      z_size = z.size()
      noise_dict ={"gaussian": torch.tensor(0)}
      noise_type = noise_dict[noise_type]
      if noise_type == noise_dict["gaussian"]:
        noise = torch.normal(mean = torch.zeros((nb_samples,z_size[0],z_size[1],z_size[2],z_size[3]),device=device),std = 1. )
      else:
        print("noise type not implemented")
      z_pert = z + noise_intensity*noise
      _, indices = torch.max(z_pert, dim =-1, keepdim=True)
      weights = torch.zeros(z_pert.size(), device = device)
      weights.scatter_(-1, indices, 1)
      ctx.save_for_backward(weights,noise,torch.tensor(noise_intensity),noise_type)
      weight = weights.mean(dim = 0)
      return weight
    
    @staticmethod
    def backward(ctx, grad_l):
      '''
      Compute derivatives of the solution of the QCQP with respect to 
      '''
      grad_z = None
      weights, noise, noise_intensity, noise_type = ctx.saved_tensors
      noise_dict ={"gaussian": torch.tensor(0)}
      if noise_type == noise_dict["gaussian"]:
        grad_z = torch.matmul(weights.unsqueeze(-1),noise.unsqueeze(-2))/noise_intensity
      else:
        print("noise_type not implemented")
      grad_z = grad_z.mean(dim=0)
      if ctx.needs_input_grad[0]:
        grad_z = torch.matmul(grad_l,grad_z)
      return grad_z, None, None, None


def random_rgb_blend(
    colors, fragments, blend_params, znear: float = 1.0, zfar: float = 100, nb_samples: int = 1
) -> torch.Tensor:

    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    background = blend_params.background_color
    if not torch.is_tensor(background):
        background = torch.tensor(background, dtype=torch.float32, device=device)
    else:
        background = background.to(device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Perturbed Heaviside function map based on the distance of the pixel to the face.
    randomheavi = randomHeaviside().apply
    prob_map = randomheavi(-fragments.dists, nb_samples, blend_params.sigma,"gaussian")*mask
    alpha = torch.prod((1.0 - prob_map), dim=-1)
    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    #need to balance between color map and depth ?Why log ?
    z_map = 1.*torch.log(eps+prob_map)+ (z_inv-z_inv_max)/ blend_params.gamma
    #add background
    z_map =torch.cat((z_map,torch.ones((z_map.size()[0],z_map.size()[1],z_map.size()[2],1),device=device)*(eps - z_inv_max) / blend_params.gamma),dim=-1)

    randomarg = randomArgmax.apply
    randomax = randomarg(z_map, nb_samples, blend_params.gamma, "gaussian")
    wz,wb = randomax[...,:-1],randomax[...,-1:]
    weighted_colors = (wz[..., None] * colors).sum(dim=-2)

    weighted_background = wb * background
    pixel_colors[..., :3] = (weighted_colors + weighted_background)
    pixel_colors[..., 3] = 1.0 - alpha

    return pixel_colors



class RandomPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None,nb_samples=1
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.nb_samples = nb_samples

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = random_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar, nb_samples = self.nb_samples
        )
        return images