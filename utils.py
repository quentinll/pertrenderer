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
from torch.autograd import Function
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
    Textures,
    TexturesVertex,
    BlendParams
)
from pytorch3d.transforms import (
    Rotate,
    euler_angles_to_matrix,
    random_rotations,
    so3_exponential_map,
    so3_log_map,
    so3_rotation_angle,
    so3_relative_angle
)
from pytorch3d.structures import Meshes
from randomras.random_rasterizer import RandomPhongShader, RandomSimpleShader, SimpleShader, SoftSimpleShader
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
import torch.autograd.profiler as profiler
import pinocchio as pin

from randomras.smoothagg import SoftAgg, CauchyAgg, GaussianAgg
from randomras.smoothrast import SoftRast, ArctanRast

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
print("device used",device)

def init_renderers(camera, lights, sigma = 1e-2, gamma = 5e-1, alpha = 1., nb_samples = 16):
    R_init = random_rotations(1)
    log_rot_init = so3_log_map(R_init)
    blend_settings=BlendParams(sigma = sigma, gamma = gamma, background_color = (.0,.0,.0)) #smoothing parameters
      
    raster_settings_soft = RasterizationSettings(
        image_size=128, 
        blur_radius= np.log(1. / 1e-4 - 1.)*blend_settings.sigma, 
        faces_per_pixel=12, 
    )
    alpha = 1.
    softras_agg = SoftAgg(gamma= gamma, alpha = alpha )
    softras_rast = SoftRast(sigma = sigma)
    
    renderer_softras = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft
        ),
        # shader=SoftSimpleShader(device=device,
        #                         blend_params=blend_settings)
        shader=RandomSimpleShader(device=device,
            cameras=camera,
            lights=lights,
            blend_params=blend_settings,
            smoothrast = softras_rast,
            smoothagg =softras_agg)
    )
      
    random_rast = ArctanRast(sigma = sigma)
    random_agg = CauchyAgg(gamma = gamma, alpha = alpha,nb_samples=nb_samples)
    renderer_random = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft
        ),
        #shader=RandomPhongShader(device=device,
        shader=RandomSimpleShader(device=device,
            cameras=camera,
            lights=lights,
            blend_params=blend_settings,
            smoothrast = softras_rast,
            smoothagg = random_agg
            )
    )
    return log_rot_init, renderer_softras, renderer_random

def init_target():
    # DATA_DIR = "./data/rubiks"
    # obj_filename = os.path.join(DATA_DIR, "cube2.obj")
    # mesh = load_objs_as_meshes([obj_filename], device=device)
    # verts = mesh.verts_packed()
    # N = verts.shape[0]
    # center = verts.mean(0)
    # scale = max((verts - center).abs().max(0)[0])
    # mesh.offset_verts_(-center.expand(N, 3))
    # mesh.scale_verts_((1.0 / float(scale)));
    
    datadir = "./data/rubiks"
    obj_filename = os.path.join(datadir, "cube2.obj")
    fn = 'cube_p.npz'
    with np.load(f'{datadir}/{fn}') as f:
        pos_idx, pos, col_idx, col = f.values()
      
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))
    if pos.shape[1] == 4: pos = pos[:, 0:3]
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).to(device = device).unsqueeze(0)
    vtx_pos = torch.from_numpy(pos.astype(np.float32)).to(device = device).unsqueeze(0)
    col_idx = torch.from_numpy(col_idx.astype(np.int32)).to(device = device).unsqueeze(0)
    vtx_col = torch.from_numpy(col.astype(np.float32)).to(device = device)
    
    #reorder color to have same cube as softras
    green_col  = vtx_col[3,:].clone()
    vtx_col[3,:] =vtx_col[0,:]
    vtx_col[0,:] = green_col
      
    verts, faces, aux= load_obj(obj_filename)
    l = aux.texture_images['cube'].size()[1]//6
    for i in range(6):
      aux.texture_images['cube'][:,i*l:(i+1)*l,:] = vtx_col[i,:].unsqueeze(0).unsqueeze(0).repeat(aux.texture_images['cube'].size()[0],l,1)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images
    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex).to(device=device)
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
        image_size=128, 
        blur_radius=0.,
        faces_per_pixel=1, 
    )
    
    blend_settings = BlendParams(background_color = (0.,0.,0.))
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        
        shader=SimpleShader(
            device=device,
            blend_params= blend_settings
        )
    )
    
    
    meshes = mesh.extend(num_views)
    R_true = random_rotations(1)
    rotation_true = Rotate(R_true, device=device)
    # rotate the mesh
    meshes_rotated = meshes.update_padded(rotation_true.transform_points(meshes.verts_padded()))
    
    
    target_images = renderer(meshes_rotated, cameras=cameras[0], lights=lights)
    #target_images = renderer(meshes, R=R, T=T, lights=lights)
    
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
  plot_period = 1
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
    #print(R)
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
    #print(torch.norm(log_rot.grad).detach().cpu().item())
    #print("AD grad",log_rot.grad)
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
  np.save(path_fig/"optimization_details"/datenow/'loss_values.npy', losses["rgb"]['values'])
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
    MC_samples = params_dic["MC_samples"]
    noise_type = params_dic["noise_type"]
    params = {"lr-smoothing":[]}
    for j,lr in enumerate(lr_list):
        for k,smoothing in enumerate(smoothing_list):
            print(j*len(smoothing_list) + k +1,'/',len(lr_list)*len(smoothing_list),'params')
            (sigma,gamma) = smoothing
            angle_errors = {"random_rasterizer":[], "softras":[]}
            for i in range(N_benchmark):
              print(i+1,'/', N_benchmark, 'test problem')
              meshes,cameras,lights,target_rgb,R_true = init_target()
              #print(R_true)
              log_rot_init, renderer_softras, renderer_random = init_renderers(cameras,lights,sigma= sigma,gamma=gamma,nb_samples=MC_samples)
              #log_rot_init = torch.tensor(pin.log3(R_true.detach().cpu()[0].numpy()), dtype=torch.float32).unsqueeze(0)
              #log_rot_init = so3_log_map(R_true)
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
    
def hat(v):
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = v.new_zeros(N, 3, 3)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h    
    

def so3_exponential_map_corrected(log_rot):
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")
    
    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    batch_size = nrms.size()[0]
    R = torch.zeros(batch_size,3,3)
    for j in range(batch_size):
        if nrms[j]!=0.:
            rot_angles = nrms[j].sqrt()
            rot_angles_inv = 1.0 / rot_angles
            fac1 = rot_angles_inv * rot_angles.sin().unsqueeze(0)
            fac2 = (rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())).unsqueeze(0)
            skews = hat(log_rot[j].unsqueeze(0))
            R[j] = (
                fac1[:, None, None] * skews
                + fac2[:, None, None] * torch.bmm(skews, skews)
                + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
            ).squeeze(0)

        else:
            R[j] = exp_map0.apply(log_rot[j])
    return R

class exp_map0(Function):
    @staticmethod
    def forward(ctx,log_rot):
        R = torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
        ctx.save_for_backward(log_rot)
        return R
    
    @staticmethod
    def backward(ctx,grad_l):
        log_rot = ctx.saved_tensors[0]
        grad_log_rot = torch.zeros(log_rot.size())
        for i in range(log_rot.size()[0]):
            e_i = torch.zeros(1,3,1)
            e_i[0,i,0] = 1.
            grad_log_rot[i] = (hat(e_i.squeeze(-1))*grad_l).sum()
        return grad_log_rot
  
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
