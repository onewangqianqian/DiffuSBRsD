# DiffuSBRsD
This repository presents a method using a generative model to improve the long - tail detection of solar radio bursts: DiffuSBRD

![Example diagram](./structure.jpg)

## Generation of Solar Radio Burst Spectrogram Images
In this part, BrushNet and text inversion are used to generate solar radio burst spectrograms.

First of all, we would like to express our gratitude for their work, which provides the foundation for ours. The address of the BrushNet repository is: [BrushNet](https://github.com/TencentARC/BrushNet.git). Meanwhile, we have made certain modifications to it to better adapt to the task of generating solar radio burst spectrograms.

### Environment Configuration
BrushNet has been implemented and tested on Pytorch 1.12.1 with Python 3.9.

Clone the repo:

```
git clone https://github.com/TencentARC/BrushNet.git
```

We recommend that you first use `conda` to create a virtual environment and install `pytorch` following [official instructions](https://pytorch.org/). For example:

```
conda create -n diffusers python=3.9 -y
conda activate diffusers
python -m pip install --upgrade pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra - index - url https://download.pytorch.org/whl/cu116
```

Then, you can install diffusers (implemented in this repo) with:

```
pip install -e.
```

After that, you can install the required packages through:

```
cd examples/brushnet/
pip install -r requirements.txt
```

### Weights
Checkpoints of BrushNet can be downloaded from [here](https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n?usp = drive_link). The `ckpt` folder contains:

- BrushNet pretrained checkpoints for Stable Diffusion v1.5 (`segmentation_mask_brushnet_ckpt` and `random_mask_brushnet_ckpt`)
- Pretrained Stable Diffusion v1.5 checkpoint (e.g., realisticVisionV60B1_v51VAE from [Civitai](https://civitai.com/)). You can use `scripts/convert_original_stable_diffusion_to_diffusers.py` to process other models downloaded from Civitai. 
- BrushNet pretrained checkpoints for Stable Diffusion XL (`segmentation_mask_brushnet_ckpt_sdxl_v1` and `random_mask_brushnet_ckpt_sdxl_v0`). A better version will be shortly released by [yuanhang](https://github.com/yuanhangio). Please stay tuned!
- Pretrained Stable Diffusion XL checkpoint (e.g., juggernautXL_juggernautX from [Civitai](https://civitai.com/)). You can use `StableDiffusionXLPipeline.from_single_file("path of safetensors").save_pretrained("path to save",safe_serialization = False)` to process other models downloaded from Civitai. 

### Train
You can train with a segmentation mask using the script:

```
# sdxl
accelerate launch examples/brushnet/train_brushnet_sdxl.py \
--pretrained_model_name_or_path data/ckpt/juggernautXL_juggernautX \
--brushnet_model_name_or_path data/ckpt/segmentation_mask_brushnet_ckpt_sdxl_v1 \
--output_dir /mnt/mydisk2/run/brushnetsdxl_mask_work_sdxl_saveck_type2_1 \
--train_data_dir data/BurstData \
--resolution 512 \
--gradient_accumulation_steps 4 \
--learning_rate 1e - 5 \
--train_batch_size 1 \
--tracker_project_name brushnet \
--image_path data/BurstData/type4_G_1 \
--placeholder_token "<type4_radio_burst>" \
--validation_steps 100 \
--initializer_token "a" \
--checkpointing_steps 100 \
--checkpoints_total_limit 5 \
--validation_prompt "<type4_radio_burst>, high saturability, blue spectrogram background" \
--validation_image "examples/brushnet/test - image/background.jpg" \
--validation_mask "examples/brushnet/test - image/type2.png" \
--seed 3586
```

### Generation
You can generate solar radio burst spectrogram images with the script:

```
# sdxl
python examples/brushnet/test_brushnet_sdxl.py
```

## Solar Radio Burst Generated Image Filter MSSIM
MSSIM scores the generated solar radio burst images and filters out the low - quality ones.

### Environment Configuration
MSSIM is implemented based on MMDetection.

We implement MSSIM using [MMDetection V2.25.3](https://github.com/open - mmlab/mmdetection/releases/tag/v2.25.3) and [MMCV V1.5.0](https://github.com/open - mmlab/mmcv/releases/tag/v1.5.0).
The source code of MMdetection has been included in this repo and you only need to build MMCV following [official instructions](https://github.com/open - mmlab/mmcv/tree/v1.5.0#installation).
We test our models under ```python = 3.7.11,pytorch = 1.11.0,cuda = 11.3```. Other versions may not be compatible. 

If you are a novice in using a computer, you may encounter some difficulties at this step.
Here, we provide some commands to help you complete the configuration of the environment. However, due to the different machine environments of each person, there may be errors in some steps. At this time, don't panic. Observe the error message and then correct it accordingly. (PS: Throwing the error message to GPT may also help you.)

```shell
conda create --name SSA python = 3.7 - y
#You need to install Anaconda first here.
conda install pytorch torchvision - c pytorch
# Here, conda will install PyTorch according to your CUDA version. If the CUDA version is too low, there may be an error, and at this time, you need to upgrade your CUDA version.
pip install -U openmim
mim install mmengine
mim install mmcv - full==1.5.0
git clone https://github.com/onewangqianqian/SSA - Co - Deformable - DETR.git
cd SSA - Co - Deformable - DETR
pip install -v -e.
```

### Script Introduction
The main scripts in MSSIM are placed in the `project`, and its structure is as follows:
```
── project
    ├── MSSIM.py
    │—— COCO2XML.py
    │── FIter.py
    └── XML2COCO.py
```
`MSSIM.py` is used to calculate the scores of the generated images.
The weights used in MSSIM are: [weights](https://drive.google.com/file/d/15ThIuUtTOmD29A9PSZSgfczVqCGWIE36/view?usp = sharing)
`FIter.py` is used to filter the generated images with low scores.
`COCO2XML.py` and `XML2COCO.py` are two convenient format conversion tools.

## Baseline Model RT - DETR
### RT - DETR Environment Configuration
```bash
pip install -r requirements.txt
```
### Datasets and Weights

[Weights](https://drive.google.com/file/d/1sPCuBqqPaZExfviZuJjprs8ruhyab - F - /view?usp = sharing)
[Datasets](https://drive.google.com/file/d/1CUR5aVStHhzUgXutgg3qbtdQnFcpU4k7/view?usp = sharing)

### Training

- Training on a Single GPU:

```shell
# training on single - gpu
export CUDA_VISIBLE_DEVICES = 0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- Training on Multiple GPUs:

```shell
# train on multi - gpu
export CUDA_VISIBLE_DEVICES = 0,1,2,3
torchrun --nproc_per_node = 4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- Evaluation on Multiple GPUs:

```shell
# val on multi - gpu
export CUDA_VISIBLE_DEVICES = 0,1,2,3
torchrun --nproc_per_node = 4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test - only
```
