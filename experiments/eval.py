#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:16:05 2021

@author: 
"""
import argparse

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pandas as pd
import numpy as np
import json 
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json

import time

import os
import ast 
from pathlib import Path
import torch
from tqdm import tqdm


from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    HardPhongShader,
    SoftSilhouetteShader,
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
    so3_relative_angle
)



from pytorch3d.structures import Meshes
from randomras.random_rasterizer import RandomPhongShader, RandomSimpleShader, SimpleShader
from pytorch3d.io import load_objs_as_meshes, load_obj

from randomras.smoothagg import SoftAgg, CauchyAgg, GaussianAgg, HardAgg, GaussianAgg_wovr
from randomras.smoothrast import SoftRast, ArctanRast, GaussianRast, AffineRast, GaussianRast_wovr


DATASET_DIRECTORY = str(Path().cwd().parents[1]/"SubsetShapenet/ShapeNetCore.v2")
NUM_ITERATIONS = 800
OPTIMIZER = "adam"
LR_VALUES = [3e-2]
SMOOTHING_VALUES = [(1e-3,1e-2)]
SMOOTHING_NOISE = ["softras","gaussian"]
MC_SAMPLES = [8]
ADAPTIVE_REGULARIZATION = 1
ADAPTIVE_PARAMS = [(1.1,1.1)]
INITIAL_PERTURBATION = 20.
CATEGORIES = ["cube"]
TASK = "pose_opt"
EXP_ID = 10
IMAGE_SIZE =128
NUM_PROB = 100
RANDOM_SEED=1
EXP_TYPE = "pose_opt"

def parse_tuples(s):
    try:
        x, y = map(float, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Tuple must be x,y")

parser = argparse.ArgumentParser()
parser.add_argument('-et', '--experiment-type', type=str, default=EXP_TYPE)
parser.add_argument('-eid', '--experiment-id', type=int, default=EXP_ID)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY) #path to shapenet 
parser.add_argument('-ni', '--num-iterations', type=int, default=NUM_ITERATIONS)
parser.add_argument('-opt', '--optimizer', type=str, default=OPTIMIZER)
parser.add_argument('-lr', '--lr-values', nargs='+', type=float, default=LR_VALUES)
parser.add_argument('-sv', '--smoothing-values', nargs='+', type=parse_tuples, default=SMOOTHING_VALUES)
parser.add_argument('-sn', '--smoothing-noise', nargs='+', type=str, default=SMOOTHING_NOISE)
parser.add_argument('-mc', '--mc-samples', nargs='+', type=int, default=MC_SAMPLES)
parser.add_argument('-ar', '--adaptive-regularization', type=bool, default=ADAPTIVE_REGULARIZATION)
parser.add_argument('-ap', '--adaptive-params', nargs='+', type=parse_tuples, default=ADAPTIVE_PARAMS)
parser.add_argument('-ip', '--initial-perturbation', type=float, default=INITIAL_PERTURBATION)
parser.add_argument('-cat', '--categories', nargs='+', type=str, default=CATEGORIES)
parser.add_argument('-tsk', '--task', type=str, default=TASK)
parser.add_argument('-np', '--num-prob', type=int, default=NUM_PROB)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available() and 1:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

torch.set_printoptions(8)
    
print("device used",device)
path_curr = Path().cwd()
os.makedirs(path_curr/'results', exist_ok=True)

def init_renderers(camera, lights, R_true, pert_init_intensity = 30., sigma = 1e-2, gamma = 5e-1, alpha = 1., nb_samples = 16, noise_type=["cauchy"], imsize = 128):
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
        image_size=imsize, 
        blur_radius= np.log(1. / 1e-4 - 1.)*blend_settings.sigma, 
        faces_per_pixel=50,
        max_faces_per_bin=50000,
        perspective_correct=False
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
        if noise_type[i] == "gaussian_wovr":
            random_rast = GaussianRast_wovr(sigma = sigma)
            random_agg = GaussianAgg_wovr(gamma = gamma, alpha = alpha,nb_samples=nb_samples)
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
            shader=RandomPhongShader(device=device,
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


def init_target(category="cube", shapenet_path = "../ShapeNetCore.v1", imsize=128):
    if category=="cube":
        datadir = "../data/objs/rubiks"
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
          
        model_verts, model_faces, aux= load_obj(obj_filename)
        l = aux.texture_images['cube'].size()[1]//6
        for i in range(6):
          aux.texture_images['cube'][:,i*l:(i+1)*l,:] = vtx_col[i,:].unsqueeze(0).unsqueeze(0).repeat(aux.texture_images['cube'].size()[0],l,1)
        verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = model_faces.textures_idx[None, ...]  # (1, F, 3)
        tex_maps = aux.texture_images
        texture_image = list(tex_maps.values())[0]
        texture_image = texture_image[None, ...]  # (1, H, W, 3)
        model_textures = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
        mesh = Meshes(verts=[model_verts], faces=[model_faces.verts_idx], textures=model_textures).to(device=device)
    else:
        dic_categories = {
        "table":"04379243",
        "car":"02958343",
        "chair":"03001627" ,
        "airplane":"02691156",
        "sofa": "04256520",
        "rifle": "04090263",
        "lamp":"03636649",
        "mug":"03797390",
        "microwave":"03761084",
        "mailbox": "03710193",
        "bus":"02924116",
        "speaker":"03691459",
        "display":"03211117",
        "dishwasher": "03207941",
        "bag": "02773838",
        "lamp": "03636649",
        "birdhouse": "02843684"
        }
        model_per_category={
            "mug":"bea77759a3e5f9037ae0031c221d81a4",
            "airplane": "ffccda82ecc0d0f71740529c616cd4c7",#"fff513f407e00e85a9ced22d91ad7027"
            "microwave": "c1851c910969d154df78375e5c76ea3d",
            "mailbox": "10e1051cbe10626e30a706157956b491",
            "bus": "7ad09b362de71bfaadcb6d6a1ff60276",
            "speaker": "1d4bb07ac73996182339c28050e32573",
            "display": "2e6204b4aa7ba83fbd28395acf9af65e",#"4744bc26253dd076174f1b91e00d9f2d",
            "dishwasher": "fb15942e4096d8f0263a7f81856f9708",
            "bag": "a55b721ea5a29d7f639ff561fa3f5bac",
            "lamp": "4a868756ae6404a5c0bc57897eddf6f",#"a0a87d63af355b45615810b8eabca5b3"
            "birdhouse": "7f53db3b31fe08283c2748dd7bf1793a"
            }
        
        SHAPENET_PATH = shapenet_path
        available_models= os.listdir(SHAPENET_PATH+'/'+dic_categories[category])
        model_id = np.random.randint(low = 0, high = len(available_models))
        print("model id: ", model_per_category[category])
        verts, faces, aux = load_obj(
        SHAPENET_PATH+'/'+dic_categories[category]+'/'+model_per_category[category]+'/'+'models'+'/'+'model_normalized.obj',
        device=device,
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=4,
        texture_wrap="repeat",
        )
        
        atlas = aux.texture_atlas
        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=TexturesAtlas(atlas=[atlas]),
        )
    model_verts = mesh.verts_packed()
    N = model_verts.shape[0]
    center = model_verts.mean(0)
    scale = max((model_verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center.expand(N, 3))
    mesh.scale_verts_((1.0 / float(scale)));
    
    
    num_views = 1
    
    elev = torch.linspace(30, 240, num_views)
    azim = torch.linspace(120,150, num_views)
    
    lights = PointLights(device=device, location=[[0.0,2.0, -2.0]])
    
    if category=="cube":
        R, T = look_at_view_transform(dist=6.7, elev=elev, azim=azim)
    else:
        mesh.scale_verts_(3.)
        R, T = look_at_view_transform(dist=6.7, elev=elev, azim=azim)
    R,T = R.to(device),T.to(device)
    cameras = [OpenGLPerspectiveCameras(device=device, R=R[None, i, ...], 
                                             T=T[None, i, ...],fov=60) for i in range(num_views)]
    camera = OpenGLPerspectiveCameras(device=device, R=R[None, 0, ...], 
                                      T=T[None, 0, ...]) 
    
    raster_settings = RasterizationSettings(
        image_size=imsize, 
        blur_radius=0.,
        faces_per_pixel=1,
        max_faces_per_bin=100000
    )
    
    blend_settings = BlendParams(background_color = (0.,0.,0.))
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        
        #shader=SimpleShader(
        shader=HardPhongShader(
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
    meshes_rotated = meshes.update_padded(rotation_true.transform_points(meshes.verts_padded()))
    
    target_images = renderer(meshes_rotated, cameras=cameras[0], lights=lights)
    
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    return meshes, cameras, lights, target_rgb, R_true


def optimize_pose(mesh,cameras,lights,init_pose,diff_renderer,target_rgb,exp_id,lr_init=5e-2,Niter=100,optimizer = "adam", adapt_reg= False, adapt_params = (1.1,1.5)):
    
    losses = {"rgb": {"weight": 1.0, "values": []},
              "angle_error":{"values":[]}
            }
    gradient_values = []
    # Plot period for the losses
    plot_period = max(int(Niter/50),1)
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
    v_sigma,v_gamma,v_alpha = torch.zeros(1),torch.zeros(1),torch.zeros(1)
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
      if gradient_values[-1]> 1000.: 
          print("grad",log_rot.grad)
          print("log_rot",log_rot)
          log_rot.grad = 1e-5*torch.normal(torch.zeros_like(log_rot.grad))
      optimizer.step()
      if adapt_reg and i>100:
          sigma,gamma,alpha = diff_renderer.shader.get_smoothing()
          grad_sigma, grad_gamma, grad_alpha = sigma.grad, gamma.grad, alpha.grad
          v_sigma, v_gamma, v_alpha =.9*v_sigma.detach().clone() + .1*grad_sigma.detach().clone(), .9*v_gamma.detach().clone() + .1*grad_gamma.detach().clone(), .9*v_alpha.detach().clone() + .1*grad_alpha.detach().clone()
          sigma.grad, gamma.grad, alpha.grad = torch.zeros_like(sigma.grad), torch.zeros_like(gamma.grad), torch.zeros_like(alpha.grad)
          blend_settings = BlendParams(sigma = sigma.detach().clone()/adapt_params[0],gamma = gamma.detach().clone()/adapt_params[1])
          nb_samples = diff_renderer.shader.get_nb_samples()
          if v_gamma >0 and (i+1)%50==0 :
                diff_renderer.rasterizer.raster_settings.blur_radius = np.log(1. / 1e-4 - 1.)*max(blend_settings.sigma,5e-5)
                diff_renderer.shader.update_smoothing(sigma=max(blend_settings.sigma,5e-5),gamma= max(blend_settings.gamma,5e-4))
                diff_renderer.shader.update_nb_samples(nb_samples = min(2*nb_samples, 128) )
                lr = max(lr/1.5, 1e-4)
                optimizer = torch.optim.Adam([log_rot], lr=lr)
    path_fig = Path().cwd().parent
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
    return best_log_rot

def compare_runtime(args):
    lr_list = args.lr_values
    smoothing_list = args.smoothing_values
    exp_id = args.experiment_id
    shapenet_location = args.dataset_directory
    N_benchmark = args.num_prob
    imsize = args.image_size
    categories = args.categories
    pert_init_intensity = args.initial_perturbation
    Niter= args.num_iterations
    optimizer = args.optimizer
    noise_type = args.smoothing_noise
    adapt_reg = args.adaptive_regularization
    adapt_params = args.adaptive_params if adapt_reg else [(1.,1.)]
    MC_samples = args.mc_samples
    params = {"lr-smoothing-MC":[], "lr": [],"sigma": [],"gamma": [],"MC": [] , "adapt_params":[]}
    mean_runtimes = {}
    for x in noise_type:
        mean_runtimes[x]= []
    test_problems = []
    meshes,cameras,lights,_,_ = init_target(category=categories[0],shapenet_path=shapenet_location, imsize=imsize)
    for i in range(N_benchmark):
        _,_,_,target_rgb,R_true = init_target(category=categories[0],shapenet_path=shapenet_location, imsize=imsize)
        log_rot_init, _ = init_renderers(cameras,lights,R_true,pert_init_intensity=pert_init_intensity,sigma= .1,gamma=.1,nb_samples=1,noise_type= noise_type)    
        test_problems += [([x.detach().clone() for x in target_rgb],R_true.detach().clone(),log_rot_init.detach().clone())]
    for j,lr in enumerate(lr_list):
        for k,smoothing in enumerate(smoothing_list):
            for kk, nb_MC in enumerate(MC_samples):
                for jj, adapt_param in enumerate(adapt_params):
                    print(j*len(smoothing_list)*len(MC_samples)*len(adapt_params) + k*len(MC_samples)*len(adapt_params) +kk*len(adapt_params)+jj +1,'/',len(lr_list)*len(smoothing_list)*len(MC_samples)*len(adapt_params),'params')
                    (sigma,gamma) = smoothing
                    runtimes = {}
                    for x in noise_type:
                        runtimes[x]= []
                    for i in range(N_benchmark):
                        print(i+1,'/', N_benchmark, 'test problem')
                        (target_rgb,R_true,log_rot_init) = test_problems[i]
                        _, renderers = init_renderers(cameras,lights,R_true,pert_init_intensity=pert_init_intensity,sigma= sigma,gamma=gamma,nb_samples=nb_MC,noise_type= noise_type, imsize= imsize)
                        for l in range(len(noise_type)):
                            print(noise_type[l])
                            t_start = time.time()
                            log_rot = optimize_pose(meshes,cameras,lights,log_rot_init, renderers[l], target_rgb,exp_id, Niter = Niter, optimizer = optimizer, adapt_reg = adapt_reg, adapt_params = adapt_param)
                            timing = time.time() - t_start
                            mean_runtimes[noise_type[l]]+=[timing]
                    params["lr-smoothing-MC"] += [(lr,sigma,gamma,nb_MC)]
                    params["lr"] += [lr]
                    params["sigma"] += [sigma]
                    params["gamma"] += [gamma]
                    params["MC"] += [nb_MC]
                    params["adapt_params"] += [adapt_param]
                    
    path_res = Path().cwd().parent
    path_res = path_res/('experiments/results/'+str(exp_id))
    file_res = open(path_res/'runtimes.txt', 'w')
    json.dump(mean_runtimes, file_res)
    return

def compare_pose_opt(args):
    lr_list = args.lr_values
    smoothing_list = args.smoothing_values
    exp_id = args.experiment_id
    shapenet_location = args.dataset_directory
    N_benchmark = args.num_prob
    imsize = args.image_size
    categories = args.categories
    pert_init_intensity = args.initial_perturbation
    Niter= args.num_iterations
    optimizer = args.optimizer
    noise_type = args.smoothing_noise
    adapt_reg = args.adaptive_regularization
    adapt_params = args.adaptive_params if adapt_reg else [(1.,1.)]
    MC_samples = args.mc_samples if not adapt_reg else [8]
    params = {"lr-smoothing-MC":[], "lr": [],"sigma": [],"gamma": [],"MC": [] , "adapt_params":[]}
    mean_errors = {}
    init_errors = {}
    final_errors = {}
    var_errors = {}
    mean_solved = {}
    exp_setup = {"perturbation": pert_init_intensity, "Niter": Niter, "optimizer": optimizer,"N_benchmark": N_benchmark ,"adaptive_regularization": adapt_reg, "category":categories}
    for x in noise_type:
        mean_errors[x]= []
        var_errors[x] = []
        init_errors[x] = []
        final_errors[x] = []
        mean_solved[x] = {1:[],2:[], 5:[], 10:[], 15:[], 20:[], 25: [], 35: [], 45:[] }
    test_problems = []
    meshes,cameras,lights,_,_ = init_target(category=categories[0],shapenet_path=shapenet_location, imsize=imsize)    
    for i in range(N_benchmark):
        _,_,_,target_rgb,R_true = init_target(category=categories[0],shapenet_path=shapenet_location, imsize=imsize)
        log_rot_init, _ = init_renderers(cameras,lights,R_true,pert_init_intensity=pert_init_intensity,sigma= .1,gamma=.1,nb_samples=1,noise_type= noise_type, imsize = imsize)    
        test_problems += [([x.detach().clone() for x in target_rgb],R_true.detach().clone(),log_rot_init.detach().clone())]
    for j,lr in enumerate(lr_list):
        for k,smoothing in enumerate(smoothing_list):
            for kk, nb_MC in enumerate(MC_samples):
                for jj, adapt_param in enumerate(adapt_params):
                    print(j*len(smoothing_list)*len(MC_samples)*len(adapt_params) + k*len(MC_samples)*len(adapt_params) +kk*len(adapt_params)+jj +1,'/',len(lr_list)*len(smoothing_list)*len(MC_samples)*len(adapt_params),'params')
                    (sigma,gamma) = smoothing
                    angle_errors = {}
                    angle_errors_init = {}
                    for x in noise_type:
                        angle_errors[x]= []
                        angle_errors_init[x]= []
                    for i in range(N_benchmark):
                        print(i+1,'/', N_benchmark, 'test problem')
                        (target_rgb,R_true,log_rot_init) = test_problems[i]
                        _, renderers = init_renderers(cameras,lights,R_true,pert_init_intensity=pert_init_intensity,sigma= sigma,gamma=gamma,nb_samples=nb_MC,noise_type= noise_type, imsize = imsize)
                        for l in range(len(noise_type)):
                            print(noise_type[l])
                            angle_errors_init[noise_type[l]]+=[so3_relative_angle(so3_exponential_map(log_rot_init), R_true).detach().cpu().item()*180./np.pi]
                            log_rot = optimize_pose(meshes,cameras,lights,log_rot_init, renderers[l], target_rgb,exp_id, Niter = Niter, optimizer = optimizer, adapt_reg = adapt_reg, adapt_params = adapt_param)
                            angle_errors[noise_type[l]]+=[so3_relative_angle(so3_exponential_map(log_rot), R_true).detach().cpu().item()*180./np.pi]
                            if angle_errors [noise_type[l]][-1] >10:
                                print("error angle ",angle_errors [noise_type[l]][-1],"init",log_rot_init, "log_rot_true", so3_log_map(R_true), "final log_rot", log_rot)
                    for l in range(len(noise_type)):
                        mean_errors[noise_type[l]] += [sum(angle_errors[noise_type[l]])/len(angle_errors[noise_type[l]])]
                        var_errors[noise_type[l]] += [np.std(angle_errors[noise_type[l]])]
                        init_errors[noise_type[l]] += [angle_errors_init[noise_type[l]]]
                        final_errors[noise_type[l]] += [angle_errors[noise_type[l]]]
                        for thresh in mean_solved[noise_type[l]]:
                            mean_solved[noise_type[l]][thresh] += [sum([1 if angle <thresh else 0 for angle in angle_errors[noise_type[l]]])/len(angle_errors[noise_type[l]])]
                    params["lr-smoothing-MC"] += [(lr,sigma,gamma,nb_MC)]
                    params["lr"] += [lr]
                    params["sigma"] += [sigma]
                    params["gamma"] += [gamma]
                    params["MC"] += [nb_MC]
                    params["adapt_params"] += [adapt_param]
                    
    path_res = Path().cwd().parent
    path_res = path_res/('experiments/results/'+str(exp_id))
    file_res = open(path_res/'angle_error.txt', 'w')
    json.dump(mean_errors, file_res)
    file_res = open(path_res/'angle_error_final.txt', 'w')
    json.dump(final_errors, file_res)
    file_res = open(path_res/'angle_error_init.txt', 'w')
    json.dump(init_errors, file_res)
    file_res = open(path_res/'angle_std.txt', 'w')
    json.dump(var_errors, file_res)
    file_res = open(path_res/'solved_percentage.txt', 'w')
    json.dump(mean_solved, file_res)
    file_params = open(path_res/'params.txt', 'w')
    json.dump(params, file_params)
    file_params = open(path_res/'exp_setup.txt', 'w')
    json.dump(exp_setup, file_params)
    plot_results = False
    if plot_results:
        df_mean = pd.read_json(path_res/'angle_error.txt')
        df_std = pd.read_json(path_res/'angle_std.txt')
        df_percent = pd.read_json(path_res/'solved_percentage.txt')
        df_params = pd.read_json(path_res/'params.txt', precise_float = True)
        dic_setup = json.load(open(path_res/'exp_setup.txt'))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = colors[:2]+colors[3:]
        solved_threshold = []
        for column in df_mean:
            df_new_percent = []
            for column2 in df_percent[column]:
                df_new_percent += [column2]
            df_new_percent = pd.DataFrame(data = np.array(df_new_percent).transpose(),columns = df_percent.index)
            df_concat = pd.concat([df_params[['lr', 'sigma', 'gamma', 'MC', 'adapt_params']],df_mean[column], df_std[column],df_new_percent], axis = 1)
            df_concat.columns = ['lr', 'sigma', 'gamma', 'MC', 'adapt_params','avg error', 'std error', '% task solved (<1°)','% task solved (<2°)','% task solved (<5°)','% task solved (<10°)','% task solved (<15°)','% task solved (<20°)','% task solved (<25°)','% task solved (<35°)','% task solved (<45°)' ]
            df_concat.sort_values(by="avg error", inplace=True)
            solved_threshold += [df_concat.iloc[0,-9:].to_list()]
        
        plt.figure(dpi=200)    
        for i in range(len(solved_threshold)):
            plt.plot([1,2,5,10,15,20,25,35,45],solved_threshold[i],label=df_mean.columns[i],color=colors[2*i])
        plt.title(str(dic_setup['perturbation'])+'° initial perturbation')
        plt.xlabel('Threshold (°)')
        plt.ylabel('% of task solved')
        plt.ylim((0.,1.))
        plt.legend()
        plt.savefig(path_res/"results_plot.png")

  
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

if args.experiment_type=="pose_opt":
    compare_pose_opt(args)
elif args.experiment_type=="runtime":
    compare_runtime(args)
