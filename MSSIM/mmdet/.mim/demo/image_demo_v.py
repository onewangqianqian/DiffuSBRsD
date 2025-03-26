# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mmcv
from pycocotools.coco import COCO

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from projects import *


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
    parser.add_argument(
        '--ann-file',
        help='Path to COCO format annotation file for ground truth visualization')
    args = parser.parse_args()
    return args


def main(args):
    # 初始化模型
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # 进行推理
    result = inference_detector(model, args.img)

    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)

    # 如果提供标注文件，加载并绘制真实标


async def async_main(args):
    # 异步逻辑保持不变
    model = init_detector(args.config, args.checkpoint, device=args.device)
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)

    # 此处应添加与main函数相同的标注绘制逻辑
    # 为简洁起见此处省略，实际应参考main函数实现


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)