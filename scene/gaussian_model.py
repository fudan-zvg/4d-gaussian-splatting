#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_rotation_4d, build_scaling_rotation_4d
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d

from typing import List, Optional

import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize, ResidualVQ

from dahuffman import HuffmanCodec
import math
from tqdm import tqdm

class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x: torch.Tensor) -> torch.Tensor:
        return x.half().float()

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L.transpose(1, 2) @ L
            symm = strip_symmetric(actual_covariance)
            return symm
        
        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0, return_covt=False):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            if not return_covt:
                return symm, mean_offset.squeeze(-1)
            else:
                return symm, mean_offset.squeeze(-1), cov_t.squeeze(-1)
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        if not self.rot_4d:
            self.covariance_activation = build_covariance_from_scaling_rotation
        else:
            self.covariance_activation = build_covariance_from_scaling_rotation_4d

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, force_sh_3d: bool = False, sh_degree_t : int = 0,
                 vq_attributes: List[str] = [],
                 qa_attributes: List[str] = [],
                 ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)
        
        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t
        self.vq_attributes = vq_attributes
        self.qa_attributes = qa_attributes
        self.indexed = False
        self.quantization = False
        self.split_t = True
        
        self.setup_functions()

        self.hybrid = False
        self.hybrid_mask = None

    def capture(self):
        if self.gaussian_dim == 3:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        elif self.gaussian_dim == 4:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.t_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t
            )
    
    def restore(self, model_args, training_args):
        if self.gaussian_dim == 3:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif self.gaussian_dim == 4:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            t_gradient_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._t,
            self._scaling_t,
            self._rotation_r,
            self.rot_4d,
            self.env_map,
            self.active_sh_degree_t) = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if not (self.indexed and '_scaling' in self.vq_attributes):
            return self.scaling_activation(self._scaling)
        else:
            if self.quantization and '_scaling' in self.qa_attributes:
                return self.scaling_activation(
                    self.scaling_qa(self.vq_scaling_model.get_codes_from_indices(self.rvq_indices_scaling).sum(dim=0).squeeze(0))
                )
            else:
                return self.scaling_activation(
                    self.vq_scaling_model.get_codes_from_indices(self.rvq_indices_scaling).sum(dim=0).squeeze(0)
                )
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t if not self.hybrid else self.scaling_t_hybrid)

    @property
    def scaling_t_hybrid(self):
        scaling_t = self._scaling_t.clone()
        scaling_t[self.hybrid_mask] = self.init_scales_t
        return scaling_t

    @property
    def rotation_r_hybrid(self):
        rotation_r = self._rotation_r.clone()
        rotation_r[self.hybrid_mask] = self._rotation[self.hybrid_mask]
        return rotation_r

    @property
    def get_scaling_xyzt(self):
        if not (self.indexed and '_scaling' in self.vq_attributes):
            return self.scaling_activation(torch.cat([self._scaling, self._scaling_t if not self.hybrid else self.scaling_t_hybrid], dim = 1))
        else:
            if self.quantization and '_scaling' in self.qa_attributes:
                return self.scaling_activation(
                    torch.cat([
                        self.scaling_qa(self.vq_scaling_model.get_codes_from_indices(self.rvq_indices_scaling).sum(dim=0).squeeze(0))
                    , self._scaling_t], dim = 1)
                )
            else:    
                return self.scaling_activation(
                    torch.cat([self.vq_scaling_model.get_codes_from_indices(self.rvq_indices_scaling).sum(dim=0).squeeze(0), self._scaling_t], dim = 1)
                )
    @property
    def get_rotation(self):
        if not (self.indexed and '_rotation' in self.vq_attributes):
            return self.rotation_activation(self._rotation)
        else:
            if self.quantization and '_rotation' in self.qa_attributes:
                return self.rotation_activation(
                    self.rotation_qa(self.vq_rotation_model.get_codes_from_indices(self.rvq_indices_rotation).sum(dim=0).squeeze(0))
                )
            else:
                return self.rotation_activation(
                    self.vq_rotation_model.get_codes_from_indices(self.rvq_indices_rotation).sum(dim=0).squeeze(0)
                )
    
    @property
    def get_rotation_r(self):
        if not (self.indexed and '_rotation' in self.vq_attributes):
            if not self.hybrid:
                return self.rotation_activation(self._rotation_r)
            else:
                return self.rotation_activation(self.rotation_r_hybrid)
        else:
            if self.quantization and '_rotation' in self.qa_attributes:
                return self.rotation_activation(
                    self.rotation_qa(self.vq_rotation_model.get_codes_from_indices(self.rvq_indices_rotation_r).sum(dim=0).squeeze(0))
                )
            else:
                return self.rotation_activation(
                    self.vq_rotation_model.get_codes_from_indices(self.rvq_indices_rotation_r).sum(dim=0).squeeze(0)
                )
    
    @property
    def get_xyz(self):
        if self.quantization and '_xyz' in self.qa_attributes:
            return self.position_qa(self._xyz)
        else:
            return self._xyz
    
    @property
    def get_t(self):
        return self._t
    
    @property
    def get_xyzt(self):
        return torch.cat([self.get_xyz, self.get_t], dim = 1)
    
    @property
    def get_features(self):
        if not (self.indexed and '_features_dc' in self.vq_attributes):
            features_dc = self._features_dc
        else:
            features_dc = self.vq_features_dc_model.get_codes_from_indices(self.rvq_indices_features_dc).sum(dim=0).squeeze(0).unsqueeze(1).contiguous()
        if not (self.indexed and '_features_rest' in self.vq_attributes):
            features_rest = self._features_rest
        else:
            features_rest = self.vq_features_rest_model.get_codes_from_indices(self.rvq_indices_features_rest).sum(dim=0).squeeze(0).view(self._features_rest.shape[0], -1, 3).contiguous()
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        if self.quantization and '_opacity' in self.qa_attributes:
            return self.opacity_qa(self.opacity_activation(self._opacity))
        else:
            return self.opacity_activation(self._opacity)
    
    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree+1)**2 * (self.max_sh_degree_t + 1)
    
    def get_cov_t(self, scaling_modifier = 1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r if not self.hybrid else self.rotation_r_hybrid)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_cov_t(scaling_modifier)
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_current_covariance_and_mean_offset(self, scaling_modifier = 1, timestamp = 0.0, return_covt=False):
        return self.covariance_activation(self.get_scaling_xyzt, scaling_modifier, 
                                                              self._rotation, 
                                                              self._rotation_r if not self.hybrid else self.rotation_r_hybrid,
                                                              dt = timestamp - self.get_t,
                                                              return_covt=return_covt)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_scale_factor: float = 5.0):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / time_scale_factor
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1
            self.init_scales_t = scales_t[0,0].detach().clone()

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4 and self.rot_4d
        self.spatial_lr_scale = spatial_lr_scale
        init_4d_gaussian = torch.load(path)
        fused_point_cloud = init_4d_gaussian['xyz'].cuda()
        features_dc = init_4d_gaussian['features_dc'].cuda()
        features_rest = init_4d_gaussian['features_rest'].cuda()
        fused_times = init_4d_gaussian['t'].cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = init_4d_gaussian['scaling'].cuda()
        rots = init_4d_gaussian['rotation'].cuda()
        scales_t = init_4d_gaussian['scaling_t'].cuda()
        rots_r = init_4d_gaussian['rotation_r'].cuda()

        opacities = init_4d_gaussian['opacity'].cuda()
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.gaussian_dim == 4: # TODO: tune time_lr_scale
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_t_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr, "name": "rotation_r"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            # if param_group["name"] == "t" and self.gaussian_dim == 4:
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

        if self.hybrid:
            self.hybrid_mask = self.hybrid_mask[valid_points_mask]
        
        if self.indexed:
            if '_scaling' in self.vq_attributes:
                self.rvq_indices_scaling = self.rvq_indices_scaling[:,valid_points_mask]
            if '_rotation' in self.vq_attributes:
                self.rvq_indices_rotation = self.rvq_indices_rotation[:,valid_points_mask]
                self.rvq_indices_rotation_r = self.rvq_indices_rotation_r[:,valid_points_mask]
            if '_features_dc' in self.vq_attributes:
                self.rvq_indices_features_dc = self.rvq_indices_features_dc[:,valid_points_mask]
            if '_features_rest' in self.vq_attributes:
                self.rvq_indices_features_rest = self.rvq_indices_features_rest[:,valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d:
                d["rotation_r"] = new_rotation_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / ((0.8*N) if self.split_t else 1.0))
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 4),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask] if not self.hybrid else self.rotation_r_hybrid[selected_pts_mask]).repeat(N,1,1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N, 1)
            new_xyz = new_xyzt[...,0:3]
            new_t = new_xyzt[...,3:4]
            new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / ((0.8*N) if self.split_t else 1.0))
            new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N,1)

        if self.hybrid:
            new_hybrid_mask = self.hybrid_mask[selected_pts_mask].repeat(N)
            self.hybrid_mask = torch.cat([self.hybrid_mask, new_hybrid_mask], dim=0)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]
        if self.hybrid:
            new_hybrid_mask = self.hybrid_mask[selected_pts_mask]
            self.hybrid_mask = torch.cat([self.hybrid_mask, new_hybrid_mask], dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False):
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
            else:
                grads_t = None

            self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t)
            self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
        
    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
        
    def post_quant(self, param, prune=False):
        max_val = torch.amax(param)
        min_val = torch.amin(param)
        param = (param - min_val)/(max_val - min_val)
        quant = torch.round(param * 255.0) / 255.0
        out = (max_val - min_val)*quant + min_val
        if prune:
            quant = quant*(torch.abs(out) > 0.1)
            out = out*(torch.abs(out) > 0.1)
        return torch.nn.Parameter(out), quant
    
    def huffman_encode(self, param):
        input_code_list = param.view(-1).tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        codec = HuffmanCodec.from_data(input_code_list)

        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        total_mb = total_bits/8/10**6
        return total_mb
    
    def vector_quantization(self, training_args, finetuning_lr_scale=0.1):
        prune_mask = (self.get_opacity <= 0.05).squeeze()
        self.prune_points(prune_mask)
        
        codebook_params = []
        
        other_params = [
            {'params': [self._xyz], 'lr': training_args.position_lr_final * self.spatial_lr_scale * finetuning_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * finetuning_lr_scale, "name": "opacity"},
            {'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale * finetuning_lr_scale, "name": "t"},
            {'params': [self._scaling_t], 'lr': training_args.scaling_t_lr * finetuning_lr_scale, "name": "scaling_t"}
        ]
        
        if self.vq_attributes:
            
            if '_features_rest' in self.vq_attributes:
                self.rvq_size_features_rest = 256
                self.rvq_num_features_rest = 8
                self.rvq_iter_features_rest = 0 # 2048
                self.rvq_bit_features_rest = math.log2(self.rvq_size_features_rest)
                rest_features_rest_params = self._features_rest.flatten(1)
                self.vq_features_rest_model = ResidualVQ(dim = rest_features_rest_params.shape[1], 
                                                         codebook_size=self.rvq_size_features_rest, num_quantizers=self.rvq_num_features_rest, 
                        commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, 
                        learnable_codebook=True, 
                        in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
                for _ in tqdm(range(self.rvq_iter_features_rest - 1)):
                    _, _, _ = self.vq_features_rest_model(rest_features_rest_params.unsqueeze(0).detach())
                _, self.rvq_indices_features_rest, _ = self.vq_features_rest_model(rest_features_rest_params.unsqueeze(0))
                codebook_params.append({'params': [p for p in self.vq_features_rest_model.parameters()], 'lr': training_args.feature_lr / 20.0 * finetuning_lr_scale, "name": "vq_features_rest"})
            else:
                other_params.append({'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 * finetuning_lr_scale, "name": "f_rest"},)

            if '_features_dc' in self.vq_attributes:
                self.rvq_size_features_dc = 64
                self.rvq_num_features_dc = 6
                self.rvq_iter_features_dc = 0 # 1024
                self.rvq_bit_features_dc = math.log2(self.rvq_size_features_dc)
                base_features_dc_params = self._features_dc
                self.vq_features_dc_model = ResidualVQ(dim = 3, codebook_size=self.rvq_size_features_dc, num_quantizers=self.rvq_num_features_dc, 
                        commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, 
                        learnable_codebook=True, 
                        in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
                for _ in tqdm(range(self.rvq_iter_features_dc - 1)):
                    _, _, _ = self.vq_features_dc_model(base_features_dc_params.squeeze(1).unsqueeze(0))
                _, self.rvq_indices_features_dc, _ = self.vq_features_dc_model(base_features_dc_params.squeeze(1).unsqueeze(0))
                codebook_params.append({'params': [p for p in self.vq_features_dc_model.parameters()], 'lr': training_args.feature_lr * finetuning_lr_scale, "name": "vq_features_dc"})
            else:
                other_params.append({'params': [self._features_dc], 'lr': training_args.feature_lr * finetuning_lr_scale, "name": "f_dc"})
            
            if '_scaling' in self.vq_attributes:
                self.rvq_size_scaling = 64
                self.rvq_num_scaling = 6
                self.rvq_iter_scaling = 0 # 512
                self.rvq_bit_scaling = math.log2(self.rvq_size_scaling)
                scaling_params = self._scaling
                self.vq_scaling_model = ResidualVQ(dim = 3, codebook_size = self.rvq_size_scaling, num_quantizers = self.rvq_num_scaling, commitment_weight = 0., 
                                    kmeans_init = True, kmeans_iters = 1, ema_update = False, 
                                    learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0004)).cuda()
                
                for _ in tqdm(range(self.rvq_iter_scaling - 1)):
                    _, _, _ = self.vq_scaling_model(scaling_params.unsqueeze(0))
                _, self.rvq_indices_scaling, _ = self.vq_scaling_model(scaling_params.unsqueeze(0)) # 1, N, 6
                codebook_params.append({'params': [p for p in self.vq_scaling_model.parameters()], 'lr': training_args.scaling_lr * finetuning_lr_scale, "name": "vq_scaling"})
            else:
                other_params.append({'params': [self._scaling], 'lr': training_args.scaling_lr * finetuning_lr_scale, "name": "scaling"})

            if '_rotation' in self.vq_attributes:
                self.rvq_size_rotation = 64
                self.rvq_num_rotation = 6
                self.rvq_iter_rotation = 0 # 1024
                self.rvq_bit_rotation = math.log2(self.rvq_size_rotation)
                rotation_params = torch.cat([self.get_rotation, self.get_rotation_r], dim=0)
                self.vq_rotation_model = ResidualVQ(dim = 4, codebook_size=self.rvq_size_rotation, num_quantizers=self.rvq_num_rotation, 
                                commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, 
                                learnable_codebook=True, 
                                in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0008)).cuda()
                for _ in tqdm(range(self.rvq_iter_rotation - 1)):
                    _, _, _ = self.vq_rotation_model(rotation_params.unsqueeze(0))
                _, rvq_indices_rotation, _ = self.vq_rotation_model(rotation_params.unsqueeze(0))
                self.rvq_indices_rotation, self.rvq_indices_rotation_r = rvq_indices_rotation.split(self._xyz.shape[0], dim=1)
                codebook_params.append({'params': [p for p in self.vq_rotation_model.parameters()], 'lr': training_args.rotation_lr * finetuning_lr_scale, "name": "vq_rotation"})
            else:
                other_params.append({'params': [self._rotation], 'lr': training_args.scaling_lr * finetuning_lr_scale, "name": "scaling"})
                other_params.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr * finetuning_lr_scale, "name": "rotation_r"})
            
            self.indexed = True
        if codebook_params:
            self.optimizer_codebook = torch.optim.Adam(codebook_params, lr=0.0, eps=1e-15)
        else:
            self.optimizer_codebook = None
        self.optimizer_others = torch.optim.Adam(other_params, lr=0.0, eps=1e-15)
        
    def fake_quantization(self):
        if self.qa_attributes:
            if '_scaling' in self.qa_attributes:
                self.scaling_qa = torch.ao.quantization.FakeQuantize(
                    dtype=torch.qint8
                ).cuda()
                # self.scaling_qa = FakeQuantizationHalf.apply
                
            if '_rotation' in self.qa_attributes:
                self.rotation_qa = torch.ao.quantization.FakeQuantize(
                    dtype=torch.qint8
                ).cuda()
                # self.rotation_qa = FakeQuantizationHalf.apply
                
            if '_xyz' in self.qa_attributes:
                self.position_qa = FakeQuantizationHalf.apply
                
            if '_opacity' in self.qa_attributes:
                self.opacity_qa = torch.ao.quantization.FakeQuantize(
                    dtype=torch.qint8
                ).cuda()
                
            self.quantization = True
    
    def compute_storage(self, encode=False):
        '''
        Adapted from https://github.com/maincold2/Compact-3DGS/blob/main/scene/gaussian_model.py#L558
        '''
        if self.quantization and '_xyz' in self.qa_attributes:
            pos_precision = 16
        else:
            pos_precision = 32
            
        if self.quantization and '_opacity' in self.qa_attributes:
            opa_precision = 8
        else:
            opa_precision = 32
            
        if self.quantization and '_scaling' in self.qa_attributes:
            scaling_precision = 8
        else:
            scaling_precision = 32
        
        if self.quantization and '_rotation' in self.qa_attributes:
            rotation_precision = 8
        else:
            rotation_precision = 32
            
        if self.quantization and '_features_dc' in self.qa_attributes:
            color_precision = 8
        else:
            color_precision = 32
        
        position_mb = self._xyz.shape[0]*4*pos_precision/8/10**6
        
        if self.quantization and '_opacity' in self.qa_attributes and encode:
            opacity_q = torch.quantize_per_tensor(
                    self.opacity_activation(self._opacity).detach(),
                    scale=self.opacity_qa.scale,
                    zero_point=self.opacity_qa.zero_point,
                    dtype=self.opacity_qa.dtype,
            ).int_repr()
            opacity_mb = self.huffman_encode(opacity_q)
        else:
            opacity_mb = self._xyz.shape[0]*opa_precision/8/10**6
                
        if self.indexed and '_scaling' in self.vq_attributes:
            scale_indices_mb = self.huffman_encode(self.rvq_indices_scaling) if encode else self._xyz.shape[0]*self.rvq_bit_scaling*self.rvq_num_scaling/8/10**6
            scale_codebook_mb = self.huffman_encode(self.post_quant(self.vq_scaling_model.codebooks.view(-1, self.vq_scaling_model.codebooks.shape[-1]))[1]) if self.quantization and '_scaling' in self.qa_attributes and encode else 2**self.rvq_bit_scaling*self.rvq_num_scaling*4*scaling_precision/8/10**6
            scale_mb = scale_indices_mb + scale_codebook_mb
        else:
            scale_mb = self._xyz.shape[0]*4*scaling_precision/8/10**6
            
        if self.indexed and '_rotation' in self.vq_attributes:
            rotation_indices_mb = 2 * self.huffman_encode(self.rvq_indices_rotation) if encode else 2 * self._xyz.shape[0]*self.rvq_bit_rotation*self.rvq_num_rotation/8/10**6
            rotation_codebook_mb = self.huffman_encode(self.post_quant(self.vq_rotation_model.codebooks.view(-1, self.vq_rotation_model.codebooks.shape[-1]))[1]) if self.quantization and '_scaling' in self.qa_attributes and encode else 2**self.rvq_bit_rotation*self.rvq_num_rotation*8*rotation_precision/8/10**6
            rotation_mb = rotation_indices_mb + rotation_codebook_mb
        else:
            rotation_mb = self._xyz.shape[0]*8*rotation_precision/8/10**6
        
        if self.indexed and '_features_dc' in self.vq_attributes:
            dc_indices_mb = self.huffman_encode(self.rvq_indices_features_dc) if encode else 2 * self._xyz.shape[0]*self.rvq_bit_features_dc*self.rvq_num_features_dc/8/10**6
            rest_indices_mb = self.huffman_encode(self.rvq_indices_features_dc) if encode else 2 * self._xyz.shape[0]*self.rvq_bit_features_dc*self.rvq_num_features_dc/8/10**6
            dc_codebook_mb = self.huffman_encode(self.post_quant(self.vq_features_dc_model.codebooks.view(-1, self.vq_features_dc_model.codebooks.shape[-1]))[1]) if self.quantization and '_features_dc' in self.qa_attributes and encode else 2**self.rvq_bit_features_dc*self.rvq_num_features_dc*8*color_precision/8/10**6
            rest_codebook_mb = self.huffman_encode(self.post_quant(self.vq_features_rest_model.codebooks.view(-1, self.vq_features_rest_model.codebooks.shape[-1]))[1]) if self.quantization and '_features_rest' in self.qa_attributes and encode else 2**self.rvq_bit_features_rest*self.rvq_num_features_rest*8*color_precision/8/10**6
            color_mb = dc_indices_mb + rest_indices_mb + dc_codebook_mb + rest_codebook_mb
        else:
            color_mb = 0
            color_mb += self._xyz.shape[0]*3*32/8/10**6
            color_mb += self._xyz.shape[0]*3*(self.get_max_sh_channels-1)*32/8/10**6
        sum_mb = position_mb+opacity_mb+scale_mb+rotation_mb+color_mb
        
        mb_str = "Storage\nposition: "+str(position_mb)+"\nscale: "+str(scale_mb)+"\nrotation: "+str(rotation_mb)+"\nopacity: "+str(opacity_mb)
        mb_str = mb_str + "\ncolor: "+str(color_mb)
        mb_str = mb_str + "\ntotal: "+str(sum_mb)+" MB"
            
        torch.cuda.empty_cache()
        return mb_str