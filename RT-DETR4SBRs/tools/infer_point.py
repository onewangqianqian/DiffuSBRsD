import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS
import numpy as np


def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou

    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]


def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size

    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))

    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates


def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)
            box[3] = np.clip(box[3] + y_shift, 0, orig_height)
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)


dit_t={
0:'type II',
1:'type III',
2:'type IIIs',
3:'type IV',
4:'type V',
}

dit_r={
1.:'type II',
2.:'type III',
3.:'type IIIs',
4.:'type IV',
5.:'type V',
}

def draw(images, labels, boxes, scores, thrh=0.6, path=""):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        for j, b in enumerate(box):
            if scrs[j] != 1:
                draw.rectangle(list(b), outline='red', width=10)

                draw.text((b[0], b[1]), text=f"{dit_t[lab[j].item()]} {round(scrs[j].item(), 5)}",
                      font=ImageFont.load_default(32), fill='white',)
            else:
                draw.rectangle(list(b), outline='#95a5a6', width=10)
                draw.text((b[0], b[1]), text=f"{dit_r[lab[j].item()]}",
                          font=ImageFont.load_default(32), fill='white', )
        if path == "":
            im.save(f'results_{i}.jpg')
        else:
            im.save(path)

def get_images(directory):
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                images.append(os.path.join(root, file))
    return images


def dew_point(images,Canv, labels, boxes, scores, thrh=0.6,):
    #创建一个620*410的 画布


    for i, im in enumerate(images):
        draw = ImageDraw.Draw(Canv)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        for j, b in enumerate(box):
            cx = int(((b[0] + b[2]) / 2)*2)
            cy = int(((b[1] + b[3]) / 2)*2)
            draw.ellipse([cx-5, cy-5, cx+5, cy+5],
                        outline=(0,255, 0, 255),fill=(0, 200, 0, 255))  # RGBA颜色（红色不透明）
def main(args, ):


    point1=[]
    point2=[]
    point3=[]


    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

    cfg.model.load_state_dict(state)


    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    images = get_images(args.images)

    for index in images:

        im_pil = Image.open(index).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)
        dataset=0

        if dataset==0:
            output = model(im_data, orig_size)
            labels, boxes, scores = output
            scr = scores[0]
            box = boxes[0][scr > 0.6]
            point1.append(box)

    for index in images:
        checkpoint = torch.load('ck/source-0.811.pth', map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        cfg.model.load_state_dict(state)
        im_pil = Image.open(index).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)
        dataset=0

        if dataset==0:
            output = model(im_data, orig_size)
            labels, boxes, scores = output
            point2.append(boxes)
        pass


    for index in images:
            import json
            val_path='dataset/annotations/instances_crop_val2017_change3.json'
            with open(val_path, 'r') as f:
                coco_data = json.load(f)

            image_name_to_info = {img['file_name']: {'id': img['id'], 'width': img['width'], 'height': img['height']}
                                  for img in coco_data['images']}
            category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

            def get_annotations(image_name):
                # 获取图像ID和信息
                image_info = image_name_to_info.get(image_name)
                if not image_info:
                    raise ValueError(f"Image {image_name} not found in the dataset.")
                image_id = image_info['id']
                image_width = image_info['width']
                image_height = image_info['height']

                # 获取对应的标注
                annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
                if not annotations:
                    print(f"No annotations found for {image_name}.")
                    return None, None, None, None

                # 提取bbox和标签
                boxes = []
                labels = []
                scores = []
                for ann in annotations:
                    x, y, w, h = ann['bbox']
                    # 转换为[x_min, y_min, x_max, y_max]
                    x_min = x
                    y_min = y
                    x_max = x + w
                    y_max = y + h
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(ann['category_id'])
                    scores.append(1)

                # 转换为张量
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32).unsqueeze(0)
                labels=torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
                scores=torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
                return boxes_tensor, labels, scores,image_width, image_height

            boxes_tensor, labels, scores,image_width, image_height = get_annotations(os.path.basename(index))


            point3.append(boxes_tensor)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-i', '--images', type=str, )
    parser.add_argument('-di', '--distance_folder', type=str, )
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=10)
    args = parser.parse_args()
    main(args)