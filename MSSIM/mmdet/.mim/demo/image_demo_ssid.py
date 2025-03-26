# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import os
import torch
import torch.nn as nn
from torchvision import transforms
from projects import *

def gaussian_kernel(x, y, sigma=1.0):
    """
    计算高斯核函数
    :param x: 第一个样本集，形状为 (n, d)
    :param y: 第二个样本集，形状为 (m, d)
    :param sigma: 高斯核的带宽
    :return: 核矩阵，形状为 (n, m)
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    return torch.exp(-dist / (2 * sigma**2))
# 定义 MMD 计算函数
def mmd(x, y, sigma=0.1):
    """
    计算最大均值差异（MMD）
    :param x: 第一个样本集，形状为 (n, d)
    :param y: 第二个样本集，形状为 (m, d)
    :param sigma: 高斯核的带宽
    :return: MMD 值
    """
    n = x.size(0)
    m = y.size(0)
    # 计算核矩阵
    kxx = gaussian_kernel(x, x, sigma)
    kxy = gaussian_kernel(x, y, sigma)
    kyy = gaussian_kernel(y, y, sigma)
    # 计算 MMD^2
    mmd_squared = (torch.sum(kxx) / (n * n) + torch.sum(kyy) / (m * m) - 2 * torch.sum(kxy) / (n * m))
    return torch.sqrt(torch.clamp(mmd_squared, min=0))
def polynomial_kernel(x, y, gamma=1.0, r=0.0, d=2):
    """
    计算多项式核函数
    :param x: 第一个样本集，形状为 (n, m)
    :param y: 第二个样本集，形状为 (k, m)
    :param gamma: 核系数
    :param r: 偏移项
    :param d: 多项式次数
    :return: 核矩阵，形状为 (n, k)
    """
    # 计算 x 和 y 的点积
    dot_product = torch.matmul(x, y.t())
    # 应用多项式核函数公式
    kernel_matrix = (gamma * dot_product + r) ** d
    return kernel_matrix
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='red',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args

def get_json_image(directory):
    json_files = []
    # 直接遍历目录下的文件，不递归子文件夹
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        # 检查是否为文件且扩展名匹配
        if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.png')):
            json_files.append(file_path)
    return json_files
def get_text(directory):
    json_files = []
    # 直接遍历目录下的文件，不递归子文件夹
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        # 检查是否为文件且扩展名匹配
        if os.path.isfile(file_path) and file.lower().endswith(('.txt')):
            json_files.append(file_path)
    return json_files
def main_branch(img_dir):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image

    imagelist = get_json_image(img_dir)
    real_images = get_json_image(img_dir + "\duibi")
    distance_folder=img_dir
    # textnumber= len(get_text(img_dir))
    # print(textnumber)
    for number, index in enumerate(imagelist[0:]):
        score_list=[]

        midoutput = inference_detector(model, index)
        midoutput = torch.tensor(midoutput).view(midoutput.size(0), -1)

        batch_size = 10  # 可根据显存情况调整批次大小
        num_batches = len(real_images) // batch_size + (
            1 if len(real_images) % batch_size != 0 else 0)
        mmd_values = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(real_images))
            batch_image_paths = real_images[start_idx:end_idx]
            batch_features = inference_detector(model,batch_image_paths)
            batch_features = torch.tensor(batch_features).view(batch_features.size(0), -1)
            k_gen_real = polynomial_kernel(midoutput, batch_features)
            k_real_real = polynomial_kernel(batch_features, batch_features)
            batch_mmd =  torch.mean(k_real_real) - 2 * torch.mean(k_gen_real)
            mmd_values.append(batch_mmd)
            # 释放不必要的显存
            del batch_features
            torch.cuda.empty_cache()
        average_mmd = torch.mean(torch.tensor(mmd_values))
        text_path = os.path.join(distance_folder, os.path.basename(index)) + "mmd.txt"
        text_data = f"mmdKID: {average_mmd}"
        # text_average=f"KID Mean: {kid_Mean}"
        with open(text_path, 'w', encoding='utf-8') as file:
            file.write(text_data)
            file.write(text_data)
        print(index)

def main(args):
    # folders = [f for f in os.listdir(args.img) if os.path.isdir(os.path.join(args.img, f))]
    # # build the model from a config file and a checkpoint file
    # for folder in folders:
    #     main_branch(os.path.join(args.img,folder))
    main_branch("E:\\coco2voc\\VOC\\dataset_chage1\\gender\\type4_G_1")

    # show the results


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
