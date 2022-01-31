"""
Inspired from Pytorch3D.
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
    prob_map = smoothrast.rasterize(fragments.dists)*mask
    alpha_chan = torch.prod((1.0 - prob_map), dim=-1)
    randomax = smoothagg.aggregate(fragments.zbuf,zfar,znear,prob_map,mask)
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
        super(RandomPhongShader,self).__init__()
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
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))[:,None,None,None]
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))[:,None,None,None]
        images = smooth_rgb_blend(
            colors, fragments, self.smoothrast, self.smoothagg, blend_params, znear=znear, zfar=zfar
        )
        return images
    
    def get_smoothing(self):
        return self.smoothrast.sigma, self.smoothagg.gamma, self.smoothagg.alpha 
    
    def get_nb_samples(self):
        return self.smoothagg.nb_samples 
    
    def update_smoothing(self,sigma=4e-4,gamma=4e-2,alpha =1.):
        self.smoothrast.update_smoothing(sigma)
        self.smoothagg.update_smoothing(gamma,alpha)
        
    def update_nb_samples(self,nb_samples =16):
        self.smoothrast.update_nb_samples(nb_samples)
        self.smoothagg.update_nb_samples(nb_samples)
    
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
        super(RandomSimpleShader,self).__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        if cameras is not None:
            self.cameras = cameras
        else:
            R, T = look_at_view_transform(dist=2.7, elev=torch.zeros((1)), azim=torch.zeros((1)))
            self.cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.smoothrast = smoothrast
        self.smoothagg = smoothagg

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = None if self.cameras is None else self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = None if self.lights is None else self.lights.to(device)

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))[:,None,None,None]
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))[:,None,None,None]
        images = smooth_rgb_blend(
            texels, fragments, self.smoothrast, self.smoothagg, blend_params, znear=znear, zfar=zfar
        )
        return images
    
    def get_smoothing(self):
        return self.smoothrast.sigma, self.smoothagg.gamma, self.smoothagg.alpha 
    
    def get_nb_samples(self):
        return self.smoothagg.nb_samples 
    
    def update_smoothing(self,sigma=4e-4,gamma=4e-2,alpha =1.):
        self.smoothrast.update_smoothing(sigma)
        self.smoothagg.update_smoothing(gamma,alpha)
        
    def update_nb_samples(self,nb_samples =16):
        self.smoothrast.update_nb_samples(nb_samples)
        self.smoothagg.update_nb_samples(nb_samples)
        
        
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