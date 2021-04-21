#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:16:05 2021

@author: quentin
"""

import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json

import os
import ast 
from pathlib import Path
import torch
from tqdm import tqdm
from torch.autograd import Function
from pytorch3d.utils import ico_sphere
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
    TexturesAtlas,
    BlendParams
)
from pytorch3d.transforms import (
    Rotate,
    random_rotations,
    so3_exponential_map,
    so3_log_map,
    so3_rotation_angle,
    so3_relative_angle
)

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)

from pytorch3d.structures import Meshes
from randomras.random_rasterizer import RandomPhongShader, RandomSimpleShader, SimpleShader, SoftSimpleShader
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj

from randomras.smoothagg import SoftAgg, CauchyAgg, GaussianAgg, HardAgg
from randomras.smoothrast import SoftRast, ArctanRast, GaussianRast, AffineRast

if torch.cuda.is_available() and 1:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

torch.set_printoptions(8)
    
print("device used",device)

def init_renderers(camera, lights, R_true, pert_init_intensity = 30., sigma = 1e-2, gamma = 5e-1, alpha = 1., nb_samples = 16, noise_type=["cauchy"]):
    if pert_init_intensity == 0.:
        print("random init ")
        R_init= random_rotations(1).to(device=device)
    else:    
        R_pert = torch.normal(torch.zeros((1,3),device = device))
        R_pert = so3_exponential_map((pert_init_intensity*np.pi/180.)*R_pert/R_pert.norm(dim=1))
        R_init = torch.bmm(R_true.clone(),R_pert).detach().clone()
    log_rot_init = so3_log_map(R_init)
    blend_settings=BlendParams(sigma = sigma, gamma = gamma, background_color = (.0,.0,.0)) #smoothing parameters
      
    raster_settings_soft = RasterizationSettings(
        image_size=128, 
        blur_radius= np.log(1. / 1e-4 - 1.)*blend_settings.sigma, 
        faces_per_pixel=12, 
        max_faces_per_bin=30,
        perspective_correct=True
    )
    alpha = 1.
    
    renderers = []
    for i in range(len(noise_type)):
        if noise_type[i] == "cauchy":
            random_rast = ArctanRast(sigma = sigma)
            random_agg = CauchyAgg(gamma = gamma, alpha = alpha,nb_samples=nb_samples)
        if noise_type[i] == "gaussian":
            random_rast = GaussianRast(sigma = sigma)
            random_agg = GaussianAgg(gamma = gamma, alpha = alpha,nb_samples=nb_samples)
        if noise_type[i] == "uniform":
            random_rast = AffineRast(sigma=sigma)
            random_agg = HardAgg()
        if noise_type[i] =="softras":
            random_rast = SoftRast(sigma = sigma)
            random_agg = SoftAgg(gamma= gamma, alpha = alpha )
            
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
                smoothrast = random_rast,
                smoothagg = random_agg
                )
        )
        renderers+=[renderer_random]
    #log_rot_init = torch.tensor([[ 0.45747742,  0.36187533, -0.92777318]], device=device)
    #log_rot_init = torch.tensor([[-0.01748946,  2.94965553, -1.79850745]], device=device)
    
    return log_rot_init, renderers

def init_render_mesh(camera, lights, sigma = 1e-2, gamma = 5e-1, alpha = 1., nb_samples = 16, noise_type=["cauchy"]):
    blend_settings=BlendParams(sigma = sigma, gamma = gamma, background_color = (.0,.0,.0)) #smoothing parameters
      
    raster_settings_soft = RasterizationSettings(
        image_size=128, 
        blur_radius= np.log(1. / 1e-4 - 1.)*blend_settings.sigma, 
        faces_per_pixel=12, 
        max_faces_per_bin=30,
        perspective_correct=True
    )
    alpha = 1.
    
    renderers = []
    for i in range(len(noise_type)):
        if noise_type[i] == "cauchy":
            random_rast = ArctanRast(sigma = sigma)
            random_agg = CauchyAgg(gamma = gamma, alpha = alpha,nb_samples=nb_samples)
        if noise_type[i] == "gaussian":
            random_rast = GaussianRast(sigma = sigma)
            random_agg = GaussianAgg(gamma = gamma, alpha = alpha,nb_samples=nb_samples)
        if noise_type[i] == "uniform":
            random_rast = AffineRast(sigma=sigma)
            random_agg = HardAgg()
        if noise_type[i] =="softras":
            random_rast = SoftRast(sigma = sigma)
            random_agg = SoftAgg(gamma= gamma, alpha = alpha )
            
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
                smoothrast = random_rast,
                smoothagg = random_agg
                )
        )
        renderers+=[renderer_random]
    src_mesh = ico_sphere(4, device)
    verts = src_mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])*2.
    src_mesh.offset_verts_(-center.expand(N, 3))
    src_mesh.scale_verts_((1.0 / float(scale)));
    deform_init = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    verts_rgb_init = torch.full([1, N, 3], 0.5, device=device)
    return src_mesh, deform_init, verts_rgb_init, renderers


def init_target():
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
    
    lights = PointLights(device=device, location=[[0.0, 0.0, -100.0]])
    
    #R, T = look_at_view_transform(dist=4.2, elev=elev, azim=azim)
    R, T = look_at_view_transform(dist=6.7, elev=elev, azim=azim)
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
    R_true = random_rotations(1).to(device=device)
    #R_true = torch.tensor([[[ 0.27466613,  0.95916265, -0.06756864],
    #     [-0.90048081,  0.23194659, -0.36787909],
    #     [-0.33718359,  0.16188823,  0.92741549]]], device= device)
    rotation_true = Rotate(R_true, device=device)
    # rotate the mesh
    meshes_rotated = meshes.update_padded(rotation_true.transform_points(meshes.verts_padded()))
    
    
    target_images = renderer(meshes_rotated, cameras=cameras[0], lights=lights)
    #target_images = renderer(meshes, R=R, T=T, lights=lights)
    
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    return meshes, cameras, lights, target_rgb, R_true



def init_target_shapenet(category="airplane", shapenet_path = "../ShapeNetCore.v1"):
    dic_categories = {
    "04379243": "table",
    "02958343": "car",
    "03001627": "chair",
    "02691156": "airplane",
    "04256520": "sofa",
    "04090263": "rifle",
    "03636649": "lamp" }
    SHAPENET_PATH = shapenet_path
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH,synsets=[category])
    shapenet_model = shapenet_dataset[0]
    print("This model belongs to the category " + shapenet_model["synset_id"] + ".")
    print("This model has model id " + shapenet_model["model_id"] + ".")
    model_verts, model_faces, model_textures = shapenet_model["verts"], shapenet_model["faces"], shapenet_model["textures"]
    model_textures = TexturesAtlas(model_textures[None]).to(device)
    target_mesh = Meshes(
        verts=[model_verts.to(device)],   
        faces=[model_faces.to(device)],
        textures=model_textures
    )
    verts = target_mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    target_mesh.offset_verts_(-center.expand(N, 3))
    target_mesh.scale_verts_((1.0 / float(scale)));
    
    num_views = 1
    
    elev = torch.linspace(20, 240, num_views)
    azim = torch.linspace(120,150, num_views)
    
    lights = PointLights(device=device, location=[[0.0, 0.0, -100.0]])
    
    #R, T = look_at_view_transform(dist=4.2, elev=elev, azim=azim)
    R, T = look_at_view_transform(dist=1.7, elev=elev, azim=azim)
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
    target_meshes = target_mesh.extend(num_views)
    
    
    target_images = renderer(target_meshes, cameras=cameras[0], lights=lights)
    #print(target_images.size())
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    return target_meshes, cameras, lights, target_rgb


def optimize_pose(mesh,cameras,lights,init_pose,diff_renderer,target_rgb,exp_id,lr_init=5e-2,Niter=100,optimizer = "adam", adapt_reg= False, adapt_params = (1.1,1.5)):
    losses = {"rgb": {"weight": 1.0, "values": []},
              "angle_error":{"values":[]}
            }
    gradient_values = []
    # Plot period for the losses
    plot_period = max(Niter/50,1)
    gradient_values = []
    loop = tqdm(range(Niter))
    images_from_training = target_rgb[0].detach().cpu().unsqueeze(0)
    log_rot = init_pose.clone()
    log_rot.requires_grad_(True)
    lr = lr_init
    if optimizer == "sgd":
        optimizer = torch.optim.SGD([log_rot], lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam([log_rot], lr=lr)
    best_log_rot = log_rot.clone()
    best_loss = np.inf
    for i in loop:
      R = so3_exponential_map(log_rot)
      rotation = Rotate(R, device=device)
      # rotate the mesh
      predicted_mesh = mesh.update_padded(rotation.transform_points(mesh.verts_padded()))
      
      loss = {k: torch.tensor(0.0, device=device) for k in losses}
      images_predicted = diff_renderer(predicted_mesh, cameras=cameras[0], lights=lights)
      # Squared L2 distance between the predicted RGB image and the target 
      # image from our dataset
      predicted_rgb = images_predicted[..., :3]
      loss_rgb = ((predicted_rgb - target_rgb[0]) ** 2).mean()
      loss["rgb"] += loss_rgb 
      for k, l in loss.items():
          losses[k]["values"].append(l.detach().cpu().item())
      
      # Print the losses
      loop.set_description("total_loss = %.6f" % loss_rgb)
    
      # Plot mesh
      if i % plot_period == 0:
          images_from_training = torch.cat((images_from_training,images_predicted[:,:,:,:3].detach().cpu()), dim = 0)
      optimizer.zero_grad()
      # Optimization step
      loss_rgb.backward()
      if loss_rgb.detach().cpu().numpy() < best_loss:
          best_loss = loss_rgb.detach().cpu().numpy()
          best_log_rot = log_rot.clone()
      gradient_values += [torch.norm(log_rot.grad).detach().cpu().item()]
      if gradient_values[-1]> 1000.: #clipping gradients
          print("grad",log_rot.grad)
          print("log_rot",log_rot)
          log_rot.grad = 1e-5*torch.normal(torch.zeros_like(log_rot.grad))
          #optimizer.zero_grad()
          #continue
          #log_rot.grad = log_rot.grad / gradient_values[-1]*.01
      optimizer.step()
      if adapt_reg and i>200 and i%50==0:
          sigma,gamma,_ = diff_renderer.shader.get_smoothing()
          blend_settings = BlendParams(sigma = sigma/adapt_params[0],gamma = gamma/adapt_params[1])
          nb_samples = diff_renderer.shader.get_nb_samples()
          diff_renderer.rasterizer.raster_settings.blur_radius = np.log(1. / 1e-4 - 1.)*blend_settings.sigma
          diff_renderer.shader.update_smoothing(sigma=blend_settings.sigma,gamma= blend_settings.gamma)
          diff_renderer.shader.update_nb_samples(nb_samples = min(2*nb_samples, 128) )
          lr = lr/1.5
          optimizer = torch.optim.Adam([log_rot], lr=lr)
    #fig = plt.figure(figsize=(13, 5))
    #ax = fig.gca()
    #ax.semilogy(losses["rgb"]['values'], label="rgb" + " loss")
    #ax.legend(fontsize="16")
    #ax.set_xlabel("Iteration", fontsize="16")
    #ax.set_ylabel("Loss", fontsize="16")
    #ax.set_title("Loss vs iterations", fontsize="16")
    path_fig = Path().cwd()
    path_fig = path_fig/('experiments/results/'+str(exp_id))
    datenow = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
        os.mkdir(path_fig/"optimization_details")
    if not os.path.exists(path_fig/"optimization_details"/datenow):
        os.mkdir(path_fig/"optimization_details"/datenow)
    np.save(path_fig/"optimization_details"/datenow/'loss_values.npy', losses["rgb"]['values'])
    np.save(path_fig/"optimization_details"/datenow/'gradient_values.npy', gradient_values)
    #plt.savefig(path_fig/"optimization_details"/datenow/'loss_values.png', bbox_inches='tight')
    #plt.close(fig)
    #plt.figure()
    #plt.semilogy([i for i in range(len(gradient_values))],gradient_values)
    image_grid(images_from_training.numpy(), rows=4, cols=1+images_from_training.size()[0]//4, rgb=True,title = path_fig/"optimization_details"/datenow)
    #plt.close()
    return best_log_rot

def optimize_mesh_deformation(base_mesh,cameras,lights,deform_init,verts_rgb_init,diff_renderer,target_rgb,exp_id,lr_init=5e-2,Niter=100,optimizer = "adam", adapt_reg= False, adapt_params = (1.1,1.5)):
    verts_shape = base_mesh.verts_packed().shape
    sphere_verts_rgb = verts_rgb_init.detach().clone().requires_grad_(True)
    deform = deform_init.detach().clone().requires_grad_(True)
    mesh = base_mesh.detach().clone().offset_verts(deform)
    losses = {"rgb": {"weight": 1.0, "values": []},
              "angle_error":{"values":[]}
            }
    gradient_values = []
    # Plot period for the losses
    plot_period = max(Niter/50,1)
    gradient_values = []
    loop = tqdm(range(Niter))
    images_from_training = target_rgb[0].detach().cpu().unsqueeze(0)
    lr = lr_init
    if optimizer == "sgd":
        optimizer = torch.optim.SGD([deform,sphere_verts_rgb], lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam([deform,sphere_verts_rgb], lr=lr)
    best_deform = deform.clone()
    best_rgb = sphere_verts_rgb.clone()
    best_loss = np.inf
    for i in loop:
      predicted_mesh = base_mesh.detach().clone().offset_verts(deform)
      predicted_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb) 
      loss = {k: torch.tensor(0.0, device=device) for k in losses}
      images_predicted = diff_renderer(predicted_mesh, cameras=cameras[0], lights=lights)
      # Squared L2 distance between the predicted RGB image and the target 
      # image from our dataset
      predicted_rgb = images_predicted[..., :3]
      loss_rgb = ((predicted_rgb - target_rgb[0]) ** 2).mean()
      loss["rgb"] += loss_rgb 
      for k, l in loss.items():
          losses[k]["values"].append(l.detach().cpu().item())
      
      # Print the losses
      loop.set_description("total_loss = %.6f" % loss_rgb)
      
      optimizer.zero_grad()
      # Optimization step
      loss_rgb.backward()
      if loss_rgb.detach().cpu().numpy() < best_loss:
          best_loss = loss_rgb.detach().cpu().numpy()
          best_deform = deform.clone()
      gradient_values += [torch.norm(deform.grad).detach().cpu().item()]
      if gradient_values[-1]> 1000.: #clipping gradients
          deform.grad = 1e-5*torch.normal(torch.zeros_like(deform.grad))
      optimizer.step()
      if adapt_reg and i>200 and i%50==0:
          sigma,gamma,_ = diff_renderer.shader.get_smoothing()
          blend_settings = BlendParams(sigma = sigma/adapt_params[0],gamma = gamma/adapt_params[1])
          nb_samples = diff_renderer.shader.get_nb_samples()
          diff_renderer.rasterizer.raster_settings.blur_radius = np.log(1. / 1e-4 - 1.)*blend_settings.sigma
          diff_renderer.shader.update_smoothing(sigma=blend_settings.sigma,gamma= blend_settings.gamma)
          diff_renderer.shader.update_nb_samples(nb_samples = min(2*nb_samples, 128) )
          lr = lr/1.5
          optimizer = torch.optim.Adam([deform], lr=lr)
    path_fig = Path().cwd()
    path_fig = path_fig/('experiments/results/'+str(exp_id))
    datenow = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
        os.mkdir(path_fig/"optimization_details")
    if not os.path.exists(path_fig/"optimization_details"/datenow):
        os.mkdir(path_fig/"optimization_details"/datenow)
    np.save(path_fig/"optimization_details"/datenow/'loss_values.npy', losses["rgb"]['values'])
    np.save(path_fig/"optimization_details"/datenow/'gradient_values.npy', gradient_values)
    image_grid(images_from_training.numpy(), rows=4, cols=1+images_from_training.size()[0]//4, rgb=True,title = path_fig/"optimization_details"/datenow)
    return best_deform, best_rgb, mesh


def compare_pose_opt(params_file):
    file = open(Path().cwd()/"experiments"/params_file, "r")
    params_dic = ast.literal_eval(file.read())
    file.close()
    lr_list = [1e-2,1e-3]
    smoothing_list = [(1e-2,1e-3)]
    exp_id = params_dic["exp_id"]
    N_benchmark = params_dic["N_benchmark"]
    pert_init_intensity = params_dic["pert_init_intensity"]
    Niter = params_dic["Niter"]
    optimizer = params_dic["optimizer"]
    lr_list = params_dic["lr_list"]
    smoothing_list = params_dic["smoothing_list"]
    noise_type = params_dic["noise_type"]
    adapt_reg  = params_dic["adapt_reg"]
    adapt_params  = params_dic["adapt_params"] if adapt_reg else [(1.,1.)]
    MC_samples = params_dic["MC_samples"] if not adapt_reg else [8]
    params = {"lr-smoothing-MC":[], "lr": [],"sigma": [],"gamma": [],"MC": [] , "adapt_params":[]}
    mean_errors = {}
    var_errors = {}
    mean_solved = {}
    exp_setup = {"perturbation": pert_init_intensity, "Niter": Niter, "optimizer": optimizer,"N_benchmark": N_benchmark ,"adaptive_regularization": adapt_reg}
    for x in noise_type:
        mean_errors[x]= []
        var_errors[x] = []
        mean_solved[x] = []
    test_problems = []
    meshes,cameras,lights,_,_ = init_target()    
    for i in range(N_benchmark):
        _,_,_,target_rgb,R_true = init_target()
        log_rot_init, _ = init_renderers(cameras,lights,R_true,pert_init_intensity=pert_init_intensity,sigma= .1,gamma=.1,nb_samples=1,noise_type= noise_type)    
        test_problems += [([x.detach().clone() for x in target_rgb],R_true.detach().clone(),log_rot_init.detach().clone())]
    for j,lr in enumerate(lr_list):
        for k,smoothing in enumerate(smoothing_list):
            for kk, nb_MC in enumerate(MC_samples):
                for jj, adapt_param in enumerate(adapt_params):
                    print(j*len(smoothing_list)*len(MC_samples)*len(adapt_params) + k*len(MC_samples)*len(adapt_params) +kk*len(adapt_params)+jj +1,'/',len(lr_list)*len(smoothing_list)*len(MC_samples)*len(adapt_params),'params')
                    (sigma,gamma) = smoothing
                    angle_errors = {}
                    for x in noise_type:
                        angle_errors[x]= []
                    #angle_errors = {"random_rasterizer":[], "softras":[]}
                    for i in range(N_benchmark):
                        print(i+1,'/', N_benchmark, 'test problem')
                        #meshes,cameras,lights,target_rgb,R_true = init_target()
                        (target_rgb,R_true,log_rot_init) = test_problems[i]
                        _, renderers = init_renderers(cameras,lights,R_true,pert_init_intensity=pert_init_intensity,sigma= sigma,gamma=gamma,nb_samples=nb_MC,noise_type= noise_type)
                        for l in range(len(noise_type)):
                            print(noise_type[l], log_rot_init)
                            log_rot = optimize_pose(meshes,cameras,lights,log_rot_init, renderers[l], target_rgb,exp_id, Niter = Niter, optimizer = optimizer, adapt_reg = adapt_reg, adapt_params = adapt_param)
                            angle_errors[noise_type[l]]+=[so3_relative_angle(so3_exponential_map(log_rot), R_true).detach().cpu().item()*180./np.pi]
                            if angle_errors [noise_type[l]][-1] >10:
                                print("error angle ",angle_errors [noise_type[l]][-1],"init",log_rot_init, "log_rot_true", so3_log_map(R_true), "final log_rot", log_rot)
                    for l in range(len(noise_type)):
                        mean_errors[noise_type[l]] += [sum(angle_errors[noise_type[l]])/len(angle_errors[noise_type[l]])]
                        var_errors[noise_type[l]] += [np.std(angle_errors[noise_type[l]])]
                        mean_solved[noise_type[l]] += [sum([1 if angle <10. else 0 for angle in angle_errors[noise_type[l]]])/len(angle_errors[noise_type[l]])]
                    params["lr-smoothing-MC"] += [(lr,sigma,gamma,nb_MC)]
                    params["lr"] += [lr]
                    params["sigma"] += [sigma]
                    params["gamma"] += [gamma]
                    params["MC"] += [nb_MC]
                    params["adapt_params"] += [adapt_param]
                    
    path_res = Path().cwd()
    path_res = path_res/('experiments/results/'+str(exp_id))
    file_res = open(path_res/'angle_error.txt', 'w')
    json.dump(mean_errors, file_res)
    #print(mean_errors, file = file_res)
    file_res = open(path_res/'angle_std.txt', 'w')
    json.dump(var_errors, file_res)
    #print(var_errors, file = file_res)
    file_res = open(path_res/'solved_percentage.txt', 'w')
    json.dump(mean_solved, file_res)
    #print(mean_solved, file = file_res)
    file_params = open(path_res/'params.txt', 'w')
    json.dump(params, file_params)
    #print(params, file = file_params)
    file_params = open(path_res/'exp_setup.txt', 'w')
    json.dump(exp_setup, file_params)
    #print(exp_setup, file = file_params)

def IoU_meshes(mesh1,mesh2):
    return 0

def compare_deform_opt(params_file):
    file = open(Path().cwd()/"experiments"/params_file, "r")
    params_dic = ast.literal_eval(file.read())
    file.close()
    lr_list = [1e-2,1e-3]
    smoothing_list = [(1e-2,1e-3)]
    exp_id = params_dic["exp_id"]
    categories = params_dic["categories"]
    N_categories = len(categories)
    shapenet_location = params_dic["shapenet_location"]
    pert_init_intensity = params_dic["pert_init_intensity"]
    Niter = params_dic["Niter"]
    optimizer = params_dic["optimizer"]
    lr_list = params_dic["lr_list"]
    smoothing_list = params_dic["smoothing_list"]
    noise_type = params_dic["noise_type"]
    adapt_reg  = params_dic["adapt_reg"]
    adapt_params  = params_dic["adapt_params"] if adapt_reg else [(1.,1.)]
    MC_samples = params_dic["MC_samples"] if not adapt_reg else [8]
    params = {"lr-smoothing-MC":[], "lr": [],"sigma": [],"gamma": [],"MC": [] , "adapt_params":[]}
    exp_setup = {"perturbation": pert_init_intensity, "Niter": Niter, "optimizer": optimizer,"categories": categories ,"adaptive_regularization": adapt_reg}
    IoUs = []
    final_picture = []
    test_problems = []
    _,cameras,lights,_ = init_target_shapenet( shapenet_path=shapenet_location)  
    raster_settings = RasterizationSettings(
        image_size=128, 
        blur_radius=0.,
        faces_per_pixel=1, 
    )
    
    blend_settings = BlendParams(background_color = (0.,0.,0.))
    
    hard_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        
        shader=SimpleShader(
            device=device,
            blend_params= blend_settings
        )
    )
    # for i in range(N_categories):
    #     target_mesh,_,_,target_rgb = init_target_shapenet(category = categories[i], shapenet_path=shapenet_location)
    #     _, deform_init, verts_rgb_init, _ = init_render_mesh(cameras,lights,sigma= .1,gamma=.1,nb_samples=1,noise_type= noise_type)    
    #     test_problems += [([x.detach().clone() for x in target_rgb], target_mesh.detach().clone())]
    #     os.makedirs(Path().cwd()/"experiments"/"results"/str(exp_id)/categories[i],exist_ok=True)
    #     plt.figure()
    #     plt.imshow(target_rgb[0].detach().cpu().numpy())
    #     path_fig = Path().cwd()/"experiments"/"results"/str(exp_id)/categories[i]
    #     plt.savefig(path_fig/("target.png"), bbox_inches='tight')
    #     plt.close()
    for j,lr in enumerate(lr_list):
        for k,smoothing in enumerate(smoothing_list):
            for kk, nb_MC in enumerate(MC_samples):
                for jj, adapt_param in enumerate(adapt_params):
                    print(j*len(smoothing_list)*len(MC_samples)*len(adapt_params) + k*len(MC_samples)*len(adapt_params) +kk*len(adapt_params)+jj +1,'/',len(lr_list)*len(smoothing_list)*len(MC_samples)*len(adapt_params),'params')
                    (sigma,gamma) = smoothing
                    IoU = {}
                    for x in noise_type:
                        IoU[x]= {cat:[] for cat in categories}
                    for i in range(N_categories):
                        print(i+1,'/', N_categories, 'test problem')
                        #(target_rgb, target_mesh) = test_problems[i]
                        target_mesh,_,_,target_rgb = init_target_shapenet(category = categories[i], shapenet_path=shapenet_location)
                        os.makedirs(Path().cwd()/"experiments"/"results"/str(exp_id)/categories[i],exist_ok=True)
                        plt.figure()
                        plt.imshow(target_rgb[0].detach().cpu().numpy())
                        path_fig = Path().cwd()/"experiments"/"results"/str(exp_id)/categories[i]
                        plt.savefig(path_fig/("target.png"), bbox_inches='tight')
                        plt.close()
                        base_mesh, deform_init, verts_rgb_init, renderers = init_render_mesh(cameras,lights,sigma= sigma,gamma=gamma,nb_samples=nb_MC,noise_type= noise_type)
                        for l in range(len(noise_type)):
                            print(noise_type[l])
                            deform, verts_rgb, _ = optimize_mesh_deformation(base_mesh,cameras,lights,deform_init, verts_rgb_init, renderers[l], target_rgb,exp_id, Niter = Niter, optimizer = optimizer, adapt_reg = adapt_reg, adapt_params = adapt_param)
                            final_mesh = base_mesh.offset_verts(deform)
                            final_mesh.textures = TexturesVertex(verts_features=verts_rgb) 
                            IoU[noise_type[l]][categories[i]]+=[IoU_meshes(final_mesh, target_mesh)]
                            plt.figure()
                            plt.imshow(hard_renderer(final_mesh, cameras=cameras[0], lights=lights)[0,...,:3].detach().cpu().numpy())
                            path_fig = Path().cwd()/"experiments"/"results"/str(exp_id)/categories[i]
                            plt.savefig(path_fig/("result_"+noise_type[l]+"smooth="+str(smoothing) +"MC="+str(nb_MC)+"lr="+str(lr)+".png"), bbox_inches='tight')
                            plt.close()
                            if IoU[noise_type[l]][categories[i]][-1] >10:
                                print("error angle ",IoU[noise_type[l]][categories[i]][-1],"init",deform_init)
                    IoUs +=[IoU]
                    params["lr-smoothing-MC"] += [(lr,sigma,gamma,nb_MC)]
                    params["lr"] += [lr]
                    params["sigma"] += [sigma]
                    params["gamma"] += [gamma]
                    params["MC"] += [nb_MC]
                    params["adapt_params"] += [adapt_param]
                    
    path_res = Path().cwd()
    path_res = path_res/('experiments/results/'+str(exp_id))
    file_params = open(path_res/'params.txt', 'w')
    json.dump(params, file_params)
    file_params = open(path_res/'exp_setup.txt', 'w')
    json.dump(exp_setup, file_params)

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
            im[...,:3] = np.clip(im[...,:3],0.,1.)
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
    path_fig = Path().cwd()
    path_fig = path_fig/'results/'/title
    plt.savefig(path_fig/'grid_cube.png', bbox_inches='tight')
    plt.close(fig)
