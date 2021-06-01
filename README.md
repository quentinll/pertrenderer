# pertrenderer
Implementation of a differentiable renderer using random perturbations. This renderer is using Pytorch3d.


## Installation
The python package can be installed by running:

```
pip install -e .
```

## Running pose optimization
Pose optimization task can be runned by doing:
```
python eval.py
```


## Usage
The perturbed renderer can be used by doing:
```
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    BlendParams)
from randomras.random_rasterizer import  RandomSimpleShader
from randomras.smoothagg import GaussianAgg
from randomras.smoothrast import GaussianRast


sigma, gamma, alpha = 1e-4, 1e-3, 1.
blend_settings=BlendParams(sigma = sigma, gamma = gamma, background_color = (1.0,1.0,1.0)) 
random_rast = GaussianRast(sigma = sigma)
random_agg = GaussianAgg(gamma = gamma, alpha = alpha, nb_samples=nb_samples)
raster_settings_soft = RasterizationSettings(
        image_size=64, 
        blur_radius= np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=50
    )
lights = PointLights(device=device, location=[[0.0,2.0, -2.0]])

R, T = look_at_view_transform(dist=6.7, elev=50, azim=120)
R, T = R.to(device), T.to(device)
camera = OpenGLPerspectiveCameras(device=device, R=R[None, 0, ...], 
                                      T=T[None, 0, ...]) 

pertrenderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings_soft
    ),
    shader=RandomSimpleShader(device=images.device,
        cameras=camera,
        lights=lights,
        blend_params=blend_settings,
        smoothrast = random_rast,
        smoothagg = random_agg
        )
)

```
