#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:11:24 2021

@author: quentin
"""

import torch
import torch.nn as nn
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
    hard_rgb_blend,
    softmax_rgb_blend,
    TexturesVertex,
    BlendParams
)

from torch.distributions.transforms import SigmoidTransform

from .smoothrast import SoftRast
from .smoothagg import SoftAgg


def smooth_rgb_blend(
    colors, fragments, smoothrast, smoothagg, blend_params, znear: float = 1.0, zfar: float = 100) -> torch.Tensor:

    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    background = blend_params.background_color
    if not torch.is_tensor(background):
        background = torch.tensor(background, dtype=torch.float32, device=device)
    else:
        background = background.to(device)

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    #rasterization
    #fragments.dists.register_hook(lambda x: print("dists grad",torch.max(x)))
    prob_map = smoothrast.rasterize(fragments.dists)*mask
    #prob_map.register_hook(lambda x: print("probmap grad",torch.max(x)))
    alpha_chan = torch.prod((1.0 - prob_map), dim=-1)
    
    #aggregation
    #fragments.zbuf.register_hook(lambda x: print("z_buf grad",torch.max(x)))
    randomax = smoothagg.aggregate(fragments.zbuf,zfar,znear,prob_map,mask)
    #randomax.register_hook(lambda x: print("aggmap grad",torch.max(x)))
    wz,wb = randomax[...,:-1],randomax[...,-1:]
    weighted_colors = (wz[..., None] * colors).sum(dim=-2)
    weighted_background = wb * background
    
    pixel_colors[..., :3] = (weighted_colors + weighted_background)
    pixel_colors[..., 3] = 1.0 - alpha_chan

    return pixel_colors



class RandomPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.
    """

    def __init__(
        self,
        device="cpu",
        cameras=None,
        lights=None, materials=None,
        smoothrast=SoftRast(),
        smoothagg=SoftAgg(),
        blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.smoothrast = smoothrast
        self.smoothagg = smoothagg

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
        images = smooth_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar, nb_samples = self.nb_samples, alpha = self.alpha, noise_type= self.noise_type
        )
        return images
    
    def get_smoothing(self):
        return self.smoothrast.sigma, self.smoothagg.gamma, self.smoothagg.alpha 
    
    def update_smoothing(self,sigma=4e-4,gamma=4e-2,alpha =1.):
        self.smoothrast.update_smoothing(sigma)
        self.smoothagg.update_smoothing(gamma,alpha)
    
class RandomSimpleShader(nn.Module):

    def __init__(
        self,
        device="cpu",
        cameras=None,
        lights=None,
        materials=None,
        smoothrast=SoftRast(),
        smoothagg=SoftAgg(),
        blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.smoothrast = smoothrast
        self.smoothagg = smoothagg

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
        #meshes.verts_padded().register_hook(lambda x: print("mesh grad", torch.max(x)))
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = smooth_rgb_blend(
            texels, fragments, self.smoothrast, self.smoothagg, blend_params, znear=znear, zfar=zfar
        )
        return images
    
    def get_smoothing(self):
        return self.smoothrast.sigma, self.smoothagg.gamma, self.smoothagg.alpha 
    
    def update_smoothing(self,sigma=4e-4,gamma=4e-2,alpha =1.):
        self.smoothrast.update_smoothing(sigma)
        self.smoothagg.update_smoothing(gamma,alpha)
        
        
class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images 
    
class SoftSimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = softmax_rgb_blend(texels, fragments, blend_params)
        return images 