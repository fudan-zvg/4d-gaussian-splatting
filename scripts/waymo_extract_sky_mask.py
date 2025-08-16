import os
import numpy as np
import cv2
from tqdm import tqdm
from mmseg.apis import inference_model, init_model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Waymo vis with SegFormer')
    parser.add_argument(
        '--config',
        type=str,
        default='mmsegmentation/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='mmsegmentation/ckpts/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth',
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--root',
        type=str,
        default='data/waymo/processed',
        help='Root directory for kitti-format waymo scenes'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = args.config
    checkpoint = args.checkpoint
    root = args.root
    
    model = init_model(config, checkpoint, device='cuda:0')
    scenes = sorted(os.listdir(root))

    for scene in scenes:
        for cam_id in range(5):
            image_dir = os.path.join(root, scene, f'image_{cam_id}')
            sky_dir = os.path.join(root, scene, f'sky_{cam_id}')
            os.makedirs(sky_dir, exist_ok=True)
            for image_name in tqdm(sorted(os.listdir(image_dir))):
                if not (image_name.endswith(".png") or image_name.endswith(".jpg")):
                    continue
                image_path = os.path.join(image_dir, image_name)
                mask_path = os.path.join(sky_dir, image_name)
                result = inference_model(model, image_path)
                mask = result.pred_sem_seg.data.cpu().numpy().astype(np.uint8)[0]
                mask = ((mask == 10).astype(np.float32) * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask)