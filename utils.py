#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:16:05 2021

@author: quentin
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import ast 
from pathlib import Path
import torch
from tqdm import tqdm
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
from pytorch3d.transforms import Rotate, euler_angles_to_matrix, random_rotations,so3_exponential_map,so3_log_map,so3_rotation_angle,so3_relative_angle
from random_rasterizer import RandomPhongShader
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
print("device used",device)

def init_renderers(camera, lights, sigma = 1e-2, gamma = 5e-1):
  R_init = random_rotations(1)
  log_rot_init = so3_log_map(R_init)
  blend_settings=BlendParams(sigma = sigma, gamma = gamma) #smoothing parameters

  raster_settings_soft = RasterizationSettings(
      image_size=158, 
      blur_radius= np.log(1. / 1e-4 - 1.)*blend_settings.sigma, 
      faces_per_pixel=50, 
  )

  renderer_softras = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=camera, 
          raster_settings=raster_settings_soft
      ),
      shader=SoftPhongShader(device=device,
          cameras=camera,
          lights=lights,
          blend_params=blend_settings)
  )

  renderer_random = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=camera, 
          raster_settings=raster_settings_soft
      ),
      shader=RandomPhongShader(device=device,
          cameras=camera,
          lights=lights,
          blend_params=blend_settings,
          nb_samples = 1)
  )
  return log_rot_init, renderer_softras, renderer_random

def init_target():
  DATA_DIR = "./data"
  obj_filename = os.path.join(DATA_DIR, "rubiks/cube2.obj")
  mesh = load_objs_as_meshes([obj_filename], device=device)
  verts = mesh.verts_packed()
  N = verts.shape[0]
  center = verts.mean(0)
  scale = max((verts - center).abs().max(0)[0])
  mesh.offset_verts_(-center.expand(N, 3))
  mesh.scale_verts_((1.0 / float(scale)));

  num_views = 1
  
  elev = torch.linspace(20, 240, num_views)
  azim = torch.linspace(120,150, num_views)

  lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

  R, T = look_at_view_transform(dist=4.7, elev=elev, azim=azim)
  #cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
  R,T = R.to(device),T.to(device)
  cameras = [OpenGLPerspectiveCameras(device=device, R=R[None, i, ...], 
                                           T=T[None, i, ...]) for i in range(num_views)]
  camera = OpenGLPerspectiveCameras(device=device, R=R[None, 0, ...], 
                                    T=T[None, 0, ...]) 

  raster_settings = RasterizationSettings(
      image_size=158, 
      blur_radius=0.,
      faces_per_pixel=1, 
  )

  renderer = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=camera, 
          raster_settings=raster_settings
      ),
      shader=HardPhongShader(
          device=device, 
          cameras=camera,
          lights=lights,
          #blend_params = blend_settings
      )
  )


  meshes = mesh.extend(num_views)
  R_true = random_rotations(1)
  rotation_true = Rotate(R_true, device=device)
  # rotate the mesh
  meshes = meshes.update_padded(rotation_true.transform_points(meshes.verts_padded()))


  #target_images = renderer(meshes, cameras=cameras, lights=lights)
  target_images = renderer(meshes, R=R, T=T, lights=lights)

  target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
  return meshes, cameras, lights, target_rgb, R_true


def optimize_pose(mesh,cameras,lights,init_pose,diff_renderer,target_rgb,exp_id,lr_init=1e-2,Niter=100,optimizer = "sgd"):
  losses = {"rgb": {"weight": 1.0, "values": []},
            #"silhouette": {"weight": 1.0, "values": []},
            "chamfer":{"values":[]},
            "angle_error":{"values":[]}
          }
  gradient_values = []
  num_views_per_iteration = 1
  num_views = len(target_rgb)
  # Plot period for the losses
  plot_period = 100
  gradient_values = []
  loop = tqdm(range(Niter))
  adapt_reg = False
  images_from_training = target_rgb[0].detach().cpu().unsqueeze(0)
  SGD_pixel = False
  backtrack_ls  = False
  log_rot = init_pose.clone()
  log_rot.requires_grad_(True)
  if optimizer == "sgd":
      optimizer = torch.optim.SGD([log_rot], lr=lr_init, momentum=0.9)#torch.optim.Adam([log_rot], lr=lr_init)
  else:
      optimizer = torch.optim.Adam([log_rot], lr=lr_init)
  for i in loop:
    R = so3_exponential_map(log_rot)
    rotation = Rotate(R, device=device)
    # rotate the mesh
    predicted_mesh = mesh.update_padded(rotation.transform_points(mesh.verts_padded()))
    
    loss = {k: torch.tensor(0.0, device=device) for k in losses}
    for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
        images_predicted = diff_renderer(predicted_mesh, cameras=cameras[j], lights=lights)
        # Squared L2 distance between the predicted RGB image and the target 
        # image from our dataset
        predicted_rgb = images_predicted[..., :3]
        if SGD_pixel:
          mask_pixel = torch.rand(predicted_rgb.size()[:3],device=device).unsqueeze(3).repeat(1,1,1,3)>=0.3
          loss_rgb = (((predicted_rgb - target_rgb[j])*mask_pixel) ** 2).mean() # add stochasticity
        else:
          loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
        loss["rgb"] += loss_rgb / num_views_per_iteration
    sum_loss = torch.tensor(0.0, device=device)
    for k, l in loss.items():
        if k!="chamfer" and k!="angle_error" :
            #print(k,l)
            sum_loss += l * losses[k]["weight"]
        losses[k]["values"].append(l.detach().cpu().item())
    
    # Print the losses
    loop.set_description("total_loss = %.6f" % sum_loss)

    # Plot mesh
    if i % plot_period == 0:
        images_from_training = torch.cat((images_from_training,images_predicted[:,:,:,:3].detach().cpu()), dim = 0)
        #visualize_prediction(predicted_mesh, renderer=renderer_textured,target_image=target_rgb[j], title="iter: %d" % i, silhouette=False)
    optimizer.zero_grad()
    # Optimization step
    sum_loss.backward()
    #print(grad_pred)
    gradient_values += [torch.norm(log_rot.grad).detach().cpu().item()]
    #print("log rot before", log_rot, "log rot grad", grad_pred)
    optimizer.step()
  fig = plt.figure(figsize=(13, 5))
  ax = fig.gca()
  ax.semilogy(losses["rgb"]['values'], label="rgb" + " loss")
  ax.legend(fontsize="16")
  ax.set_xlabel("Iteration", fontsize="16")
  ax.set_ylabel("Loss", fontsize="16")
  ax.set_title("Loss vs iterations", fontsize="16")
  path_fig = Path().cwd()
  path_fig = path_fig/('experiments/results/'+str(exp_id))
  datenow = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
  if not os.path.exists(path_fig):
      os.mkdir(path_fig)
      os.mkdir(path_fig/"optimization_details")
  if not os.path.exists(path_fig/"optimization_details"/datenow):
      os.mkdir(path_fig/"optimization_details"/datenow)
  plt.savefig(path_fig/"optimization_details"/datenow/'loss_values.png', bbox_inches='tight')
  plt.close()
  plt.figure()
  plt.semilogy([i for i in range(len(gradient_values))],gradient_values)
  image_grid(images_from_training.numpy(), rows=4, cols=1+images_from_training.size()[0]//4, rgb=True,title = path_fig/"optimization_details"/datenow)
  plt.close()
  return log_rot


def compare_pose_opt(params_file):
    file = open(Path().cwd()/"experiments"/params_file, "r")
    params_dic = ast.literal_eval(file.read())
    file.close()
    mean_errors = {"random_rasterizer":[], "softras":[]}
    lr_list = [1e-2,1e-3]
    smoothing_list = [(1e-2,1e-3)]
    exp_id = params_dic["exp_id"]
    N_benchmark = params_dic["N_benchmark"]
    Niter = params_dic["Niter"]
    optimizer = params_dic["optimizer"]
    lr_list = params_dic["lr_list"]
    smoothing_list = params_dic["smoothing_list"]
    params = {"lr-smoothing":[]}
    for j,lr in enumerate(lr_list):
        for k,smoothing in enumerate(smoothing_list):
            print(j*len(smoothing_list) + k +1,'/',len(lr_list)*len(smoothing_list),'params')
            (sigma,gamma) = smoothing
            angle_errors = {"random_rasterizer":[], "softras":[]}
            for i in range(N_benchmark):
              print(i+1,'/', N_benchmark, 'test problem')
              meshes,cameras,lights,target_rgb,R_true = init_target()
              log_rot_init, renderer_softras, renderer_random = init_renderers(cameras,lights,sigma,gamma)
              log_rot = optimize_pose(meshes,cameras,lights,log_rot_init, renderer_softras, target_rgb,exp_id, Niter = Niter, optimizer = optimizer)
              angle_errors["softras"]+=[so3_relative_angle(so3_exponential_map(log_rot), R_true).detach().cpu().numpy()*180./np.pi]
              log_rot = optimize_pose(meshes,cameras,lights,log_rot_init, renderer_random, target_rgb,exp_id, Niter = Niter, optimizer = optimizer)
              angle_errors["random_rasterizer"]+=[so3_relative_angle(so3_exponential_map(log_rot), R_true).detach().cpu().numpy()*180./np.pi]
            mean_errors["softras"] += [sum(angle_errors["softras"])/len(angle_errors["softras"])]
            mean_errors["random_rasterizer"] += [sum(angle_errors["random_rasterizer"])/len(angle_errors["random_rasterizer"])]
            params["lr-smoothing"] += [(lr,sigma,gamma)]
    path_res = Path().cwd()
    path_res = path_res/('experiments/results/'+str(exp_id))
    file_res = open(path_res/'angle_error.txt', 'w')
    print(mean_errors, file = file_res)
    file_params = open(path_res/'params.txt', 'w')
    print(params, file = file_params)

def visualize_prediction(predicted_mesh, renderer,R,T,
                         target_image, title='', 
                         silhouette=False):
    inds = 3 if silhouette else range(3)
    predicted_images = renderer(predicted_mesh, R=R[0:1], T=T[0:1])
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.grid("off")
    plt.axis("off")
    plt.close()
    
    
def image_grid(
    images,
    title,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
    path_fig = Path().cwd()
    path_fig = path_fig/'results/'/title
    plt.savefig(path_fig/'grid_cube.png', bbox_inches='tight')
