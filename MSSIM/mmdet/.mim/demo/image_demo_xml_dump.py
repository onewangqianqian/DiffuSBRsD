# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from projects import *
import os
dit={
    "0": "Type2",
    "1": "Type3",
    "2": "Type3s",
    "3": "Type4",
    "4": "Type5"
}
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

def save_annotation_xml(result, dir,score_thr=0.3):
    if isinstance(result, tuple):
        bbox_result, segm_result = result# ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]


    # 创建根元素 <annotation>
    annotation = ET.Element("annotation")

    # 添加 <folder> 元素
    folder_elem = ET.SubElement(annotation, "folder")
    folder_elem.text = dir.split("\\")[5]

    # 添加 <filename> 元素
    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = os.path.basename(dir)

    # 添加 <path> 元素
    path_elem = ET.SubElement(annotation, "path")
    # 注意：这里我们假设路径是基于文件夹和文件名的，但你可以根据需要修改它
    path_elem.text = dir

    # 创建 <source> 元素和它的子元素 <database>
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"  # 或者你可以传递一个参数来设置这个值

    # 创建 <size> 元素和它的子元素 <width>, <height>, <depth>
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(616)
    ET.SubElement(size, "height").text = str(408)
    ET.SubElement(size, "depth").text = str(3)

    # 添加 <segmented> 元素
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for index in range(0,bboxes.shape[0]) :
        # 创建 <object> 元素和它的子元素
        object_elem = ET.SubElement(annotation, "object")
        ET.SubElement(object_elem, "name").text = dit[str(labels[index])]
        ET.SubElement(object_elem, "pose").text = "Unspecified"
        ET.SubElement(object_elem, "truncated").text = str(0)
        ET.SubElement(object_elem, "difficult").text = str(0)

        # 创建 <bndbox> 元素和它的子元素
        bndbox = ET.SubElement(object_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bboxes[index][0])
        ET.SubElement(bndbox, "ymin").text = str(bboxes[index][1])
        ET.SubElement(bndbox, "xmax").text = str(bboxes[index][2])
        ET.SubElement(bndbox, "ymax").text = str(bboxes[index][3])

    # 将树结构写入XML文件
    tree = ET.ElementTree(annotation)
    path = dir.replace(".jpg",".xml")
    tree.write(path, encoding='utf-8', xml_declaration=True)
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
        if os.path.isfile(file_path) and file.lower().endswith(('.xml')):
            json_files.append(file_path)
    return json_files

def main_branch(img_dir):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    imagelist = get_text(img_dir)
    distance_folder=img_dir
    textnumber= len(get_text(img_dir))
    print(textnumber)
    for number, index in enumerate(imagelist[textnumber:]):
        result = inference_detector(model,index)
        save_annotation_xml(result,index)
        print(index)

        # show_result_pyplot(
        #     model,
        #     index,
        #     result,
        #     palette=args.palette,
        #     score_thr=args.score_thr,
        #     out_file=args.out_file)
        pass

    pass
def main(args):
    folders = [f for f in os.listdir(args.img) if os.path.isdir(os.path.join(args.img, f))]
    # build the model from a config file and a checkpoint file
    for folder in folders:
        main_branch(os.path.join(args.img,folder))


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
