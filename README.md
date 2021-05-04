# pertrenderer
Implementation of a differentiable renderer using random perturbations. This renderer is using Pytorch3d.


## Installation
The python package can be installed by running:

```
pip install -e .
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


random_rast = GaussianRast(sigma = self.sigma)
        random_agg = GaussianAgg(gamma = self.gamma, alpha = self.alpha,nb_samples=self.nb_samples)
        renderer_random = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_soft
            ),
            shader=RandomSimpleShader(device=images.device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_settings,
                smoothrast = random_rast,
                smoothagg = random_agg
                )
        )

 ```
