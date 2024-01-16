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

from typing import NamedTuple
import torch.nn as nn
import torch
# from . import _C
import os
from torch.utils.cpp_extension import load
parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diff-gaussian-rasterization")
_C = load(
    name='diff_gaussian_rasterization',
    extra_cuda_cflags=["-I " + os.path.join(parent_dir, "third_party/glm/"), "-g"],
    sources=[
        os.path.join(parent_dir, "cuda_rasterizer/rasterizer_impl.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/forward.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/backward.cu"),
        os.path.join(parent_dir, "rasterize_points.cu"),
        os.path.join(parent_dir, "ext.cpp")],
    verbose=True)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    flow_2d,
    opacities,
    ts,
    scales,
    scales_t,
    rotations,
    rotations_r,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        flow_2d,
        opacities,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        flow_2d,
        opacities,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            flow_2d,
            opacities,
            ts,
            scales,
            scales_t,
            rotations,
            rotations_r,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.sh_degree_t,
            raster_settings.campos,
            raster_settings.timestamp,
            raster_settings.time_duration,
            raster_settings.rot_4d,
            raster_settings.gaussian_dim,
            raster_settings.force_sh_3d,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, flow, depth, T, radii, geomBuffer, binningBuffer, imgBuffer, covs_com = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, flow, depth, T, radii, geomBuffer, binningBuffer, imgBuffer, covs_com = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, 
                                flow_2d, opacities, ts, scales_t, rotations_r,
                                geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth, 1-T, flow, covs_com

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth, grad_alpha, grad_flow, grad_covs_com):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, 
         flow_2d, opacities, ts, scales_t, rotations_r,
         geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors
        
        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                flow_2d,
                opacities,
                ts,
                scales,
                scales_t,
                rotations,
                rotations_r,
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_depth,
                grad_alpha,
                grad_flow,
                sh, 
                raster_settings.sh_degree,
                raster_settings.sh_degree_t,
                raster_settings.campos,
                raster_settings.timestamp,
                raster_settings.time_duration,
                raster_settings.rot_4d,
                raster_settings.gaussian_dim,
                raster_settings.force_sh_3d,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, 
                grad_flows, grad_ts, grad_scales, grad_scales_t, 
                grad_rotations, grad_rotations_r) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, 
                grad_flows, grad_ts, grad_scales, grad_scales_t, 
                grad_rotations, grad_rotations_r) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_flows,
            grad_opacities,
            grad_ts,
            grad_scales,
            grad_scales_t,
            grad_rotations,
            grad_rotations_r,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    sh_degree_t: int
    campos : torch.Tensor
    timestamp: float
    time_duration: float
    rot_4d: bool
    gaussian_dim: int
    force_sh_3d: bool
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, flow_2d = None, ts=None,
                scales = None, scales_t=None, 
                rotations = None, rotations_r=None, 
                cov3D_precomp = None):
    
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if self.raster_settings.rot_4d and cov3D_precomp is None and (
                rotations_r is None or scales_t is None or ts is None):
            raise Exception(
                'Please provide exactly rotations_r and scales_t and ts if rot_4d and cov3D_precomp is None!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if flow_2d is None:
            flow_2d = torch.Tensor([])

        if ts is None:
            ts = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if scales_t is None:
            scales_t = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if rotations_r is None:
            rotations_r = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            flow_2d,
            opacities,
            ts,
            scales,
            scales_t,
            rotations,
            rotations_r,
            cov3D_precomp,
            raster_settings,
        )

