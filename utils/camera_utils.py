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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

import torch
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height# cam_info.image.size

    if args.resolution in [1, 2, 3, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    cx = cam_info.cx / scale
    cy = cam_info.cy / scale
    fl_y = cam_info.fl_y / scale
    fl_x = cam_info.fl_x / scale
    
    loaded_mask = None
    if not args.dataloader:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]

        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
    else:
        gt_image = cam_info.image
    
    if cam_info.depth is not None:
        depth = PILtoTorch(cam_info.depth, resolution) * 255 / 10000
    else:
        depth = None

    if cam_info.sky_mask is not None:
        if cam_info.sky_mask.shape[:2] != resolution[::-1]:
            sky_mask = cv2.resize(cam_info.sky_mask, resolution)
        else:
            sky_mask = cam_info.sky_mask
        if len(sky_mask.shape) == 2:
            sky_mask = sky_mask[..., None]
        sky_mask = torch.from_numpy(sky_mask).float().permute(2, 0, 1)
    else:
        sky_mask = None

    if cam_info.pointcloud_camera is not None:
        h, w = gt_image.shape[1:]
        K = np.eye(3)
        if cam_info.cx:
            K[0, 0] = fl_x
            K[1, 1] = fl_y
            K[0, 2] = cx
            K[1, 2] = cy
        else:
            K[0, 0] = fov2focal(cam_info.FovX, w)
            K[1, 1] = fov2focal(cam_info.FovY, h)
            K[0, 2] = w / 2
            K[1, 2] = h / 2
        pts_depth = np.zeros([1, h, w])
        point_camera = cam_info.pointcloud_camera
        uvz = point_camera[point_camera[:, 2] > 0]
        uvz = uvz @ K.T
        uvz[:, :2] /= uvz[:, 2:]
        uvz = uvz[uvz[:, 1] >= 0]
        uvz = uvz[uvz[:, 1] < h]
        uvz = uvz[uvz[:, 0] >= 0]
        uvz = uvz[uvz[:, 0] < w]
        uv = uvz[:, :2]
        uv = uv.astype(int)
        # TODO: may need to consider overlap
        pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]
        pts_depth = torch.from_numpy(pts_depth).float()
    else:
        pts_depth = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  timestamp=cam_info.timestamp,
                  cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y, depth=depth, resolution=resolution, image_path=cam_info.image_path,
                  meta_only=args.dataloader,
                  pts_depth=pts_depth, sky_mask=sky_mask,
                  )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
