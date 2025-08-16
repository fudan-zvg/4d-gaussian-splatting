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
from torch.nn import functional as F
import math
from .diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_shfs_4d

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           time_shift=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if time_shift is not None:
        viewpoint_camera.timestamp -= time_shift
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sh_degree_t=pc.active_sh_degree_t,
        campos=viewpoint_camera.camera_center,
        timestamp=viewpoint_camera.timestamp,
        time_duration=pc.time_duration[1]-pc.time_duration[0],
        rot_4d=pc.rot_4d,
        gaussian_dim=pc.gaussian_dim,
        force_sh_3d=pc.force_sh_3d,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    scales_t = None
    rotations = None
    rotations_r = None
    ts = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if pc.rot_4d:
            cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
            means3D = means3D + delta_mean
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        if pc.gaussian_dim == 4:
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
            opacity = opacity * marginal_t
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if pc.gaussian_dim == 4:
            scales_t = pc.get_scaling_t
            ts = pc.get_t
            if pc.rot_4d:
                rotations_r = pc.get_rotation_r

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            if pipe.compute_cov3D_python:
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            else:
                _, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
                dir_pp = ((means3D + delta_mean) - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if pc.gaussian_dim == 3 or pc.force_sh_3d:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            elif pc.gaussian_dim == 4:
                dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            if pc.gaussian_dim == 4 and ts is None:
                ts = pc.get_t
    else:
        colors_precomp = override_color
    
    flow_2d = torch.zeros_like(pc.get_xyz[:,:2])
    
    # Prefilter
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        mask = marginal_t[:,0] > 0.05
        if means2D is not None:
            means2D = means2D[mask]
        if means3D is not None:
            means3D = means3D[mask]
        if ts is not None:
            ts = ts[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if opacity is not None:
            opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if scales_t is not None:
            scales_t = scales_t[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if rotations_r is not None:
            rotations_r = rotations_r[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if flow_2d is not None:
            flow_2d = flow_2d[mask]
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha, flow, covs_com = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        flow_2d = flow_2d,
        opacities = opacity,
        ts = ts,
        scales = scales,
        scales_t = scales_t,
        rotations = rotations,
        rotations_r = rotations_r,
        cov3D_precomp = cov3D_precomp)
    
    
    if pipe.env_map_res > 0:
        rendered_image_before = rendered_image
        assert pc.env_map is not None
        rays_d = viewpoint_camera.get_world_directions().permute(1, 2, 0)
        bg_color_from_envmap = pc.env_map.query_cube(rays_d).permute(2, 0, 1)
        rendered_image = rendered_image + (1 - alpha) * bg_color_from_envmap
    
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
    else:
        radii_all = radii

    if time_shift is not None:
        viewpoint_camera.timestamp += time_shift

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_all > 0,
            "radii": radii_all,
            "depth": depth,
            "alpha": alpha,
            "flow": flow}
