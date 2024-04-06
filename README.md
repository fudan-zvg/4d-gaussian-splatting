# Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting
### [Project page](https://fudan-zvg.github.io/4d-gaussian-splatting/) | [Paper](https://arxiv.org/abs/2310.10642)
> [**Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting**](https://arxiv.org/abs/2310.10642),            
> Zeyu Yang, Hongye Yang, Zijie Pan, [Li Zhang](https://lzrobots.github.io)  
> **Fudan University**  
> **ICLR 2024**


**This repository is the official implementation of "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting".** In this paper, we propose coherent integrated modeling of the space and time dimensions for dynamic scenes by formulating unbiased 4D Gaussian primitives along with a dedicated rendering pipeline.


## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>


## Get started

### Environment

The hardware and software requirements are the same as those of the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), which this code is built upon. To setup the environment, please run the following command:

```shell
git clone https://github.com/fudan-zvg/4d-gaussian-splatting
cd 4d-gaussian-splatting
conda env create --file environment.yml
conda activate 4dgs
```

### Data preparation

**DyNeRF dataset:**

Download the [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video) and extract each scene to `data/N3V`. After that, preprocess the raw video by executing:

```shell
python scripts/n3v2blender.py data/N3V/$scene_name
```

**DNeRF dataset:**

The dataset can be downloaded from [drive](https://drive.google.com/file/d/19Na95wk0uikquivC7uKWVqllmTx-mBHt/view?usp=sharing) or [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0). Then, unzip each scene into `data/dnerf`.


### Running

After the installation and data preparation, you can train the model by running:

```shell
python train.py --config $config_path
```

## 🎥 Videos

### 🎞️ Demo

[![Demo Video](https://i3.ytimg.com/vi/3cXC9e4CujM/maxresdefault.jpg)](https://www.youtube.com/embed/3cXC9e4CujM)

### 🎞️ Dynamic novel view synthesis

https://github.com/fudan-zvg/4d-gaussian-splatting/assets/45744267/5e163b88-4f70-4157-b9f5-8431b13c26b7

### 🎞️ Bullet time

https://github.com/fudan-zvg/4d-gaussian-splatting/assets/45744267/ac5bc3b2-dd17-446d-9ee6-6efcc871eb84

### 🎞️ Free view synthesis from a teleporting camera

https://github.com/fudan-zvg/4d-gaussian-splatting/assets/45744267/6bd0b57b-4857-4722-9851-61250a2521ab

### 🎞️ Monocular dynamic scene reconstruction

https://github.com/fudan-zvg/4d-gaussian-splatting/assets/45744267/2c79974c-1867-4ce6-848b-5d31679b6067


## 📜 BibTex
```bibtex
@inproceedings{yang2023gs4d,
  title={Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting},
  author={Yang, Zeyu and Yang, Hongye and Pan, Zijie and Zhang, Li},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
