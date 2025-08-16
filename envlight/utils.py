import os
import numpy as np

import torch
import nvdiffrast.torch as dr


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x


def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)


def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x  # right
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x  # left
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y  # top
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y  # bottom
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)  # back
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)  # front
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map, res, device='cuda'):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        # tu = torch.atan2(v[..., 0:1], v[..., 1:2]) / (2 * np.pi) + 0.5
        # tv = torch.acos(torch.clamp(v[..., 2:3], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap


def cubemap_to_latlong(cubemap, res, device='cuda'):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta,
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return torch.nn.functional.avg_pool2d(cubemap.permute(0, 3, 1, 2), (2, 2)).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device=dout.device)
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device),
                                   indexing='ij')
            v = safe_normalize(cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out        