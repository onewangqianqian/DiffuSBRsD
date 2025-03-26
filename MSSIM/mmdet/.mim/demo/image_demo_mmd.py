# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
import torch
from argparse import ArgumentParser
from mmdet.apis import (
    async_inference_detector,
    inference_detector,
    init_detector,
    show_result_pyplot
)


# 如果你有其他自定义模块，可在此引入
# from projects import *

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
    return torch.exp(-dist / (2 * sigma ** 2))


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
    kxx = gaussian_kernel(x, x, sigma)
    kxy = gaussian_kernel(x, y, sigma)
    kyy = gaussian_kernel(y, y, sigma)
    mmd_squared = (torch.sum(kxx) / (n * n) +
                   torch.sum(kyy) / (m * m) -
                   2 * torch.sum(kxy) / (n * m))
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
    dot_product = torch.matmul(x, y.t())
    kernel_matrix = (gamma * dot_product + r) ** d
    return kernel_matrix


def parse_args():
    parser = ArgumentParser()
    # query_img 为待测单张图像的路径
    parser.add_argument('query_img', help='待测图像文件路径')
    parser.add_argument('config', help='模型配置文件路径')
    parser.add_argument('checkpoint', help='模型权重文件路径')
    # 新增参数：参考图像所在目录，要求目录下只存放待比较的图片（支持 .jpg, .png, .jpeg, .bmp）
    parser.add_argument('--ref-dir', required=True, help='参考图像所在目录')
    parser.add_argument('--batch-size', type=int, default=10, help='处理参考图像时的批次大小')
    parser.add_argument('--device', default='cuda:0', help='推理所用设备')
    parser.add_argument('--out-file', default=None, help='保存结果的文本文件路径')
    parser.add_argument('--palette', default='red', choices=['coco', 'voc', 'citys', 'random'],
                        help='可视化时使用的颜色')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox 置信度阈值')
    parser.add_argument('--async-test', action='store_true', help='是否使用异步推理')
    args = parser.parse_args()
    return args


def get_image_files(directory, extensions=('.jpg', '.png', '.jpeg', '.bmp')):
    """
    遍历指定目录，返回所有图片文件路径列表
    """
    image_files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and file.lower().endswith(extensions):
            image_files.append(file_path)
    return image_files
def get_json_image(directory):
    json_files = []
    # 直接遍历目录下的文件，不递归子文件夹
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        # 检查是否为文件且扩展名匹配
        if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.png')):
            json_files.append(file_path)
    return json_files

def compute_mmd_single_vs_multiple(query_img, ref_dir, model, batch_size=10):
    """
    计算单张图像与参考图像集之间的 MMD 值
    :param query_img: 单张图像文件路径
    :param ref_dir: 参考图像所在目录
    :param model: 已初始化的检测模型
    :param batch_size: 批处理大小
    :return: 计算得到的 MMD 值（张量）
    """
    # 提取待测图像特征
    query_feature = inference_detector(model, query_img)
    query_feature = torch.tensor(query_feature).view(1, -1)

    # 获取参考图像列表
    ref_images = get_image_files(ref_dir)
    if not ref_images:
        print(f"未在目录 {ref_dir} 中找到参考图像。")
        return None

    # 批量提取参考图像特征
    ref_features_list = []
    num_batches = len(ref_images) // batch_size + (1 if len(ref_images) % batch_size != 0 else 0)
    for i in range(num_batches):
        batch_paths = ref_images[i * batch_size: (i + 1) * batch_size]
        # 如果 inference_detector 支持批量输入则直接传入列表，否则可循环处理
        batch_features = inference_detector(model, batch_paths)
        batch_features = torch.tensor(batch_features).view(len(batch_paths), -1)
        ref_features_list.append(batch_features)
        # 清理显存（可选）
        del batch_features
        torch.cuda.empty_cache()

    ref_features = torch.cat(ref_features_list, dim=0)

    # 计算多项式核
    # query 与自身的核值（标量）
    k_qq = polynomial_kernel(query_feature, query_feature)  # shape (1,1)
    # 参考图像之间的核值（取均值）
    k_rr = polynomial_kernel(ref_features, ref_features)
    # query 与参考图像之间的核值
    k_qr = polynomial_kernel(query_feature, ref_features)

    # 根据公式：MMD^2 = k(query, query) + mean(k(ref, ref)) - 2 * mean(k(query, ref))
    mmd_squared = k_qq[0, 0] + k_rr.mean() - 2 * k_qr.mean()
    mmd_value = torch.sqrt(torch.clamp(mmd_squared, min=0))
    return mmd_value


def main(args):
    # 初始化检测模型
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # 计算单张图像与参考图像集合间的 MMD
    imagelist=get_json_image(args.query_img)
    for image_path in imagelist:
        mmd_value = compute_mmd_single_vs_multiple(image_path, args.ref_dir, model, batch_size=args.batch_size)
        if mmd_value is not None:
            # 若未指定输出文件，则默认保存在待测图像同目录下，文件名为 <原文件名>_mmd.txt
            output_file = args.out_file if args.out_file else os.path.splitext(image_path)[0] + "_mmd.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"MMD: {mmd_value.item()}\n")
            print(f"MMD 值 {mmd_value.item()} 已保存至 {output_file}")


async def async_main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    tasks = asyncio.create_task(async_inference_detector(model, args.query_img))
    result = await asyncio.gather(tasks)
    show_result_pyplot(
        model,
        args.query_img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file
    )


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
