# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import os
import torch
import json
from mmdet.apis import init_detector, inference_detector
from projects import *


def gaussian_kernel(x, y, sigma=1.0):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    return torch.exp(-dist / (2 * sigma ** 2))


def mmd(x, y, sigma=0.1):
    n = x.size(0)
    m = y.size(0)
    kxx = gaussian_kernel(x, x, sigma)
    kxy = gaussian_kernel(x, y, sigma)
    kyy = gaussian_kernel(y, y, sigma)
    mmd_squared = (torch.sum(kxx) / (n * n) + torch.sum(kyy) / (m * m) - 2 * torch.sum(kxy) / (n * m))
    return torch.sqrt(torch.clamp(mmd_squared, min=0))


def polynomial_kernel(x, y, gamma=1.0, r=0.0, d=2):
    dot_product = torch.matmul(x, y.t())
    return (gamma * dot_product + r)  **  d


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image directory')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='red', choices=['coco', 'voc', 'citys', 'random'])
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--async-test', action='store_true')
    return parser.parse_args()


def get_json_image(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.jpg', '.png'))]


def image_mse(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2)  **  2).item()


def main_branch(img_dir, args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    imagelist = get_json_image(img_dir)
    real_dir = os.path.join(img_dir, "duibi")
    real_images = get_json_image(real_dir)

    # Precompute real images features
    real_features = []
    for img_path in real_images:
        feat = inference_detector(model, img_path)
        real_features.append(torch.tensor(feat).to(args.device))
    real_features_tensor = torch.cat(real_features,dim=0)

    data_list = []
    for img_path in imagelist:
        mid_feat = inference_detector(model, img_path)
        mid_tensor = torch.tensor(mid_feat).to(args.device)
        # mid_tensor = torch.tensor(mid_feat).unsqueeze(0).to(args.device)

        # Vectorized MSE calculation
        squared_diff = (real_features_tensor - mid_tensor)  **  2
        mse_scores = squared_diff.mean(dim=(1, 2, 3))
        # mse_scores = torch.mean((mid_tensor - real_features_tensor)** 2, dim=1).cpu()

        score1 = torch.max(mse_scores).item()
        score2 = torch.mean(mse_scores).item()

        data_list.append({
            "Type": os.path.basename(img_dir).split('_')[0],
            "name": os.path.basename(img_path),
            "score1": score1,
            "score2": score2,
        })
        print(data_list[-1])

    # Save results
    output_dir = "E:\\coco2voc\\VOC_to_COCO-master\\gender\\faster-rcnn_r50_fpn_1x_coco_ASSIMD\\MSEJSON_max_m3"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(img_dir)}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    print(f"Saved results to: {output_path}")


def main(args):
    path_list = ['/type2_G_1', '/type3s_G_1', '/type4_G_1', '/type5_G_1']
    for folder in path_list:
        img_dir = os.path.join(args.img, folder.strip('/'))
        main_branch(img_dir, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)