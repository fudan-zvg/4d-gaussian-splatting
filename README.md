# Generation branch

This branch is customized for video-to-4D task using 4DGS.

# Installation

Compared to original 4DGS (main branch), we require a higher version of Python (3.7 -> 3.10).

```bash
conda env create --file environment.yml
conda activate 4dgs
```

## Prepare Data

We prepare one case under `./data/feeding_squirrel` for reference. 

The `input` folder contains single-view video frames (e.g., [Consistent4D dataset](https://github.com/yanqinJiang/Consistent4D)).

The `multi_view` folder contains multi-view images at some key frames (e.g., 
[Efficient4D](https://github.com/LogosIntelligenceLab/Efficient4D),
[Diffusion^2](https://github.com/LogosIntelligenceLab/diffusion-square),
[SV4D](https://github.com/Stability-AI/generative-models)).


## Running

```bash
name=feeding_squirrel
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/video.yaml \
  input=./data/${name}/input \
  save_path=${name} \
  outdir=logs/${name} \
  multi_view=True \
  multi_view_input=./data/${name}/multi_view \
  data_mode=syncdreamer \
  batch_size=16 \
  iters=500
```

## Acknowledgement
The implementation refers to 
[DreamGaussian](https://github.com/dreamgaussian/dreamgaussian), 
[DreamGaussian4D](https://github.com/jiawei-ren/dreamgaussian4d), 
[STAG4D](https://github.com/zeng-yifei/STAG4D), 
and we sincerely thank them for their contributions to the community.
