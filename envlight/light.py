import torch
import imageio
import torchvision.transforms.functional

from . import renderutils as ru
from .utils import *

class EnvLight(torch.nn.Module):

    def __init__(self, path=None, device=None, scale=1.0, min_res=8, start_res=32, max_res=512, min_roughness=0.08, max_roughness=0.5, trainable=False):
        super().__init__()
        self.device = device if device is not None else 'cuda'  # only supports cuda
        self.scale = scale  # scale of the hdr values
        self.min_res = min_res  # minimum resolution for mip-map
        self.current_res = start_res  # maximum resolution for mip-map
        self.max_res = max_res  # maximum resolution for mip-map
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.trainable = trainable
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")

        # init an empty cubemap
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, self.current_res, self.current_res, 3,
                             dtype=torch.float32, device=self.device, requires_grad=self.trainable),
        )
        # try to load from file
        if path is not None:
            self.load(path)
        # else:
        #     self.base.data[:] = 1
        # self.base.data[:] = 1
        # self.base.data[:] = 0
        # # self.base.data[5] = 1
        # # self.base.data[2, 4:5,4:5] = 100
        # self.base.data[2, 1:2,1:2] = 100
        # # self.base.data[2] = 100
        self.step()

    def load(self, path):
        # load latlong env map from file
        image = imageio.imread(path)
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255
        image = torch.from_numpy(image).to(self.device) * self.scale
        self.image = image
        cubemap = latlong_to_cubemap(image, [self.current_res, self.current_res], self.device)

        self.base.data = cubemap

    def upsample(self):
        if self.current_res < self.max_res:
            self.current_res *= 2
            self.base = torch.nn.Parameter(torchvision.transforms.functional.resize(
                self.base.data.permute(0, 3, 1, 2), [self.current_res, self.current_res], antialias=True).permute(0, 2,
                                                                                                                  3, 1),
                                           requires_grad=True)
            print(f"Upsampling resolution to {self.current_res} !!!")

    def get_envmap_from_base(self):
        self.envmap = cubemap_to_latlong(self.base, [1000, 2000], self.device)

    def step(self):
        self.build_mips()
        self.get_envmap_from_base()

    def direct_light(self, dirs):
        """infer light from cubemap directly"""
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)
        v = dirs @ self.to_opengl.T
        # self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")

        # tu = torch.atan2(v[..., 0:1], v[..., 1:2]) / (2 * np.pi) + 0.5
        # tv = torch.acos(torch.clamp(v[..., 2:3], min=-1, max=1)) / np.pi
        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        # import pdb;pdb.set_trace()
        texcoord = torch.cat((tu, tv), dim=-1)
        light = dr.texture(self.envmap[None, ...], texcoord[None, None, ...], filter_mode='linear')[0, 0]
        # light = light / (light + 1)
        # light = light.clamp(0, 1)
        return light.reshape(*shape)

    def build_mips(self, cutoff=0.99):

        self.direct = [self.base]
        # while self.direct[-1].shape[1] > self.min_res:
        #     self.direct += [cubemap_mip.apply(self.direct[-1])]

        # self.diffuse = ru.diffuse_cubemap(self.direct[-1])
        #
        # self.specular = [None] * len(self.direct)
        # for idx in range(len(self.direct) - 1):
        #     roughness = (idx / (len(self.direct) - 2)) * (
        #                 self.max_roughness - self.min_roughness) + self.min_roughness
        #     self.specular[idx] = ru.specular_cubemap(self.direct[idx], roughness, cutoff)
        #
        # self.specular[-1] = ru.specular_cubemap(self.direct[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        # map roughness to mip_level (float):
        # roughness: 0 --> self.min_roughness --> self.max_roughness --> 1
        # mip_level: 0 --> 0                  --> M - 2              --> M - 1
        return torch.where(
            roughness < self.max_roughness,
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (
                        self.max_roughness - self.min_roughness) * (len(self.specular) - 2),
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(
                self.specular) - 2
        )

    def query_cube(self, l):
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.direct[0][None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        return light

    def __call__(self, l, roughness=None):
        # l: [..., 3], normalized direction pointing from shading position to light
        # roughness: [..., 1]
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()

        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])
            if roughness is not None:
                roughness = roughness.reshape(1, 1, -1, 1)

        if roughness is None:
            # diffuse light
            light = dr.texture(self.diffuse[None, ...], l, filter_mode='linear', boundary_mode='cube')
        else:
            # specular light
            miplevel = self.get_mip(roughness)

            light = dr.texture(
                self.specular[0][None, ...],
                l,
                mip=list(m[None, ...] for m in self.specular[1:]),
                mip_level_bias=miplevel[..., 0],
                filter_mode='linear-mipmap-linear',
                boundary_mode='cube'
            )
        light = light.view(*prefix, -1)

        return light
