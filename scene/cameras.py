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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
from kornia import create_meshgrid
from copy import deepcopy

class Camera:
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", timestamp = 0.0,
                 cx=-1, cy=-1, fl_x=-1, fl_y=-1, depth=None, resolution=None, image_path=None, meta_only=False,
                 ):

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.cx = cx
        self.cy = cy
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.resolution = resolution
        self.image_path = image_path
        self.image = image
        self.gt_alpha_mask = gt_alpha_mask
        self.meta_only = meta_only
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]
        
        if not self.meta_only:
            if gt_alpha_mask is not None:
                self.image *= gt_alpha_mask.to(self.image.device)
            else:
                self.image *= torch.ones((1, self.image_height, self.image_width), device=self.image.device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        if cx > 0:
            self.projection_matrix = getProjectionMatrixCenterShift(self.znear, self.zfar, cx, cy, fl_x, fl_y, self.image_width, self.image_height).transpose(0,1)
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        self.timestamp = timestamp
        
    def get_rays(self):
        grid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False)[0] + 0.5
        i, j = grid.unbind(-1)
        pts_view = torch.stack([(i-self.cx)/self.fl_x, (j-self.cy)/self.fl_y, torch.ones_like(i), torch.ones_like(i)], -1).to(self.data_device)
        c2w = torch.linalg.inv(self.world_view_transform.transpose(0, 1))
        pts_world =  pts_view @ c2w.T
        directions = pts_world[...,:3] - self.camera_center[None,None,:]
        return self.camera_center[None,None], directions / torch.norm(directions, dim=-1, keepdim=True)
    
    def cuda(self):
        cuda_copy = deepcopy(self)
        for k, v in cuda_copy.__dict__.items():
            if isinstance(v, torch.Tensor):
                cuda_copy.__dict__[k] = v.to(cuda_copy.data_device)
        return cuda_copy
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

