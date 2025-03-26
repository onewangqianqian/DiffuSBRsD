from diffusers import StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL
import torch
import cv2
import numpy as np
from PIL import Image
import os
import random
import json
import pickle
from diffusers.pipelines.shap_e.renderer import encode_position
import accelerate
from accelerate import Accelerator
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    BrushNetModel,
    DDPMScheduler,
    StableDiffusionXLBrushNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

# choose the base model here
base_model_path = ""
# base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
import shutil
# input brushnet ckpt path
burst_temp=""
brushnet_path =""

# choose whether using blended operation
blended = False

# input source image / mask image path and the text prompt
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

# pkl_path="/mnt/mydisk2/run/brushnetsdxl_mask_work_sdxl_saveck_type4_1/checkpoint-500/random_states_0.pkl"
# with open(pkl_path, 'rb') as f:
#     random_states = pickle.load(f)
# # conditioning scalecd
# print(random_states)
brushnet_conditioning_scale=1.0

tokenizer_one = AutoTokenizer.from_pretrained(
        "data/ckpt/juggernautXL_juggernautX",
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
tokenizer_two = AutoTokenizer.from_pretrained(
    "data/ckpt/juggernautXL_juggernautX",
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)

# import correct text encoder classes
text_encoder_cls_one = import_model_class_from_model_name_or_path(
    burst_temp, None,
)
text_encoder_cls_two = import_model_class_from_model_name_or_path(
    "data/ckpt/juggernautXL_juggernautX", None, subfolder="text_encoder_2",
)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained("data/ckpt/juggernautXL_juggernautX", subfolder="scheduler")
text_encoder_one = text_encoder_cls_one.from_pretrained(
    burst_temp, subfolder="text_encoder", revision=None, variant=None
)
text_encoder_two = text_encoder_cls_two.from_pretrained(
    "data/ckpt/juggernautXL_juggernautX", subfolder="text_encoder_2", revision=None, variant=None
)
vae = AutoencoderKL.from_pretrained("data/ckpt/juggernautXL_juggernautX/vae")
unet = UNet2DConditionModel.from_pretrained(
    "data/ckpt/juggernautXL_juggernautX", subfolder="unet", revision=None, variant=None
)
brushnet = BrushNetModel.from_pretrained(brushnet_path)
# pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
#     base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=True
# )

# change to sdxl-vae-fp16-fix to avoid nan in VAE encoding when using fp16
# pipe.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16,subfolder="vae")
#
# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.

accelerator = Accelerator(
    gradient_accumulation_steps=4,
    mixed_precision="fp16",
)
text_encoder_one, brushnet = accelerator.prepare(
    text_encoder_one, brushnet
)
brushnet = accelerator.unwrap_model(brushnet)
pipeline = StableDiffusionXLBrushNetPipeline.from_pretrained(
    base_model_path,
    text_encoder=accelerator.unwrap_model(text_encoder_one),
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    vae=vae,
    unet=unet,
    brushnet=brushnet,
    revision=None,
    variant=None,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
    use_safetensors=True
)
# pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
# pipeline = pipeline.to(accelerator.device)
# pipeline.set_progress_bar_config(disable=True)
# init_image = cv2.imread(image_path)[:,:,::-1]
# mask_image=cv2.imread(mask_path).sum(-1)>255
# # mask_image =1-(1.*mask_image)
# mask_image=1.*mask_image
#
# # resize image
# h,w,_ = init_image.shape
# scale=1
# if w<h:
#     scale=1024/w
# else:
#     scale=1024/h
#
#
# new_h=int(h*scale)
# new_w=int(w*scale)
#
# init_image=cv2.resize(init_image,(new_w,new_h))
# mask_image=cv2.resize(mask_image,(new_w,new_h))[:,:,np.newaxis]
#
# init_image = init_image * (1-mask_image)
#
# init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
# mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
image_path="examples/brushnet/test-image/image.jpg"
mask_path="examples/brushnet/test-image/output_image.png"
caption="<type4_radio_burst>, high saturation, blue background"

validation_image = Image.open(image_path).convert("RGB")
validation_mask = Image.open(mask_path).convert("RGB")
validation_image = Image.composite(Image.new('RGB', (validation_image.size[0], validation_image.size[1]), (0, 0, 0)),
                                   validation_image, validation_mask.convert("L"))

generator = torch.Generator("cuda").manual_seed(3586)
# generator=None
def get_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                json_files.append(os.path.join(root, file))
    return json_files
def get_jpg_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                json_files.append(os.path.join(root, file))
    return json_files

def get_number(directory):
    files = os.listdir(directory)
    # 过滤掉文件夹，统计文件的数量
    file_count = sum(1 for f in files if os.path.isfile(os.path.join(directory, f)))
    return file_count
mask_list=get_json_files("")
image_list=get_jpg_files("")
# mask_list=get_json_files("/root/code/BurstNet/examples/brushnet/type4add")
# image_list=get_jpg_files("/root/code/BurstNet/examples/brushnet/type4add")


for index in range(0,10000):
    image_path=random.choice(image_list)
    # mask_path=mask_list[index % len(mask_list)]
    mask_path=random.choice(mask_list)
    # mask_path="/root/code/BurstNet/examples/brushnet/test-image/type4_1_output_image.png"

    validation_image = Image.open(image_path).convert("RGB")

    validation_mask = Image.open(mask_path).convert("RGB")
    validation_image = Image.composite(
        Image.new('RGB', (validation_image.size[0], validation_image.size[1]), (0, 0, 0)), validation_image,
        validation_mask.convert("L"))

    image = pipeline(
        prompt=caption,
        image=validation_image,
        mask=validation_mask,
        num_inference_steps=20,
        generator=generator,
    ).images[0]

    dir_path=""
    imagename="type4_G1_"+str(index)+".jpg"
    image.save(dir_path+imagename)
    xml_path=dir_path+imagename.replace('.jpg','.xml')
    shutil.copy(mask_path, xml_path)

if blended:
    image_np=np.array(image)
    init_image_np=cv2.imread(image_path)[:,:,::-1]
    mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]

    # blur, you can adjust the parameters for better performance
    mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:,:,np.newaxis]
    mask_np = 1-(1-mask_np) * (1-mask_blurred)

    image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
    image_pasted=image_pasted.astype(image_np.dtype)
    image=Image.fromarray(image_pasted)


