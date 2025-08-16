# 4D Gaussian Splatting Extension

This repository contains the official implementation of the [extended paper](https://arxiv.org/abs/2412.20720). 

## Installation

There are some additional packages required to running the following code after setting up environment per the main branch instruction. They can be installed with:

```shell
pip install -r requirements_extension.txt
```

## Technicolor dataset

**Data preparation**
Please obtain the [Technicolor Dataset](https://www.interdigital.com/data_sets/light-field-dataset) and place it in `data/technicolor`. Then you can preprocess it by running the following script:

```shell
./scripts/technicolor_convertor.sh
```

**Running**

We provide configurations for each scene in `configs/technicolor`. You can train them with:

```shell
python train.py --config $config_path
```

## Compression

The post-training quantification is mainly supported via the following arguments:

#### --vq_attributes
Specify which attributes need to compress using vector quantization.
#### --qa_attributes
Specify which attributes to compress using precision reduction.
#### --vq_finetune_iters
Number of steps for quantization-aware fine-tuning after compression.

More fine-grained customization can be achieved by modifiying the function [here](scene/gaussian_model.py:L738). We provide a sample config file in `configs/dynerf/cut_roasted_beef_compact.yaml`, which can be directly use by: 
```shell
python train.py --config configs/dynerf/cut_roasted_beef_compact.yaml
```
Or resume from a pretrained checkpoint:
```shell
`python train.py --config configs/dynerf/cut_roasted_beef_compact.yaml --start_checkpoint $checkpoint_path
```

NOTE: Now the released code mainly aims to demonstrate the possibility and potential of compression for the academic community. It does not really provide interaface for saving / loading a compressed model.

## Results on Waymo Open Dataset

Sequences in Waymo Open Dataset (WOD) is serialized into TFRecords. To make them easier to be read, we need to first convert it to the KITTI format. Assuming the used TFRecords are placed under `$record_root`, you can convert them using the following script:

```shell
python scripts/waymo_convertor.py --root-path $record_root --out-dir $processed_path --workers $num_workers
```

Then we need to extract sky mask for each scene. We provide the a simple [script](scripts/waymo_extract_sky_mask.py) for this process. Please follow [MMSegmentation installation guide](https://mmsegmentation.readthedocs.io/en/main/get_started.html) to set up the required environment.

After that, you can train on the processed sequence using the sample config provided in `configs/waymo/0000.yaml`.

## Acknowledgement

We are sincerely thankful to the following open-source projects, from which we drew inspiration and adapted some codes in this branch:

- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [STG](https://github.com/oppo-us-research/SpacetimeGaussians)
- [Compact3DGS](https://github.com/maincold2/Compact-3DGS)