import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import shutil
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np
import xml.etree.ElementTree as ET
from src.kidchange import KernelInceptionDistance


def postprocess(labels, boxes, scores, iou_threshold=0.5):
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

dit={
    "0": "Type2",
    "1": "Type3",
    "2": "Type3s",
    "3": "Type4",
    "4": "Type5"
}
def save_annotation_xml(file_path, folder, filename, orig_size,lab,box,
                        truncated=1, difficult=0):
    # 创建根元素 <annotation>
    annotation = ET.Element("annotation")

    # 添加 <folder> 元素
    folder_elem = ET.SubElement(annotation, "folder")
    folder_elem.text = folder

    # 添加 <filename> 元素
    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = filename

    # 添加 <path> 元素
    path_elem = ET.SubElement(annotation, "path")
    # 注意：这里我们假设路径是基于文件夹和文件名的，但你可以根据需要修改它
    path_elem.text = f"E:\\coco2voc\\VOC\\grenateimg\\{folder}\\{filename}"

    # 创建 <source> 元素和它的子元素 <database>
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"  # 或者你可以传递一个参数来设置这个值

    # 创建 <size> 元素和它的子元素 <width>, <height>, <depth>
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(orig_size[0][0].item())
    ET.SubElement(size, "height").text = str(orig_size[0][1].item())
    ET.SubElement(size, "depth").text = str(3)

    # 添加 <segmented> 元素
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    shape = box.shape
    for index in range(0,shape[0]) :
        # 创建 <object> 元素和它的子元素
        object_elem = ET.SubElement(annotation, "object")
        ET.SubElement(object_elem, "name").text = dit[str(lab[index].item())]
        ET.SubElement(object_elem, "pose").text = "Unspecified"
        ET.SubElement(object_elem, "truncated").text = str(truncated)
        ET.SubElement(object_elem, "difficult").text = str(difficult)

        # 创建 <bndbox> 元素和它的子元素
        bndbox = ET.SubElement(object_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(box[index][0].item())
        ET.SubElement(bndbox, "ymin").text = str(box[index][1].item())
        ET.SubElement(bndbox, "xmax").text = str(box[index][2].item())
        ET.SubElement(bndbox, "ymax").text = str(box[index][3].item())

    # 将树结构写入XML文件
    tree = ET.ElementTree(annotation)
    path = "E:\\coco2voc\\VOC\\grenateimg\\resultkidxiaoyu12\\"+filename.replace(".jpg",".xml")
    tree.write(path, encoding='utf-8', xml_declaration=True)

def draw(images, labels, boxes, scores,file_name,orig_size, thrh = 0.6,number=None):
    path = "E:\\coco2voc\\VOC\\grenateimg\\resultkidxiaoyu12"
    folder = "resultkidxiaoyu12"
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        # if scrs>=0.9:
        #     choice_xml(box)
        #     pass
        save_annotation_xml(file_name, folder, file_name, orig_size,lab,box)
        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"label: {lab[j].item()} {round(scrs[j].item(),2)}", font=ImageFont.load_default(20), fill='white')
        if path == "":
            im.save(f'results_{i}.jpg')
        else:
            im.save(path+"\\"+file_name)
def get_json_image(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                json_files.append(os.path.join(root, file))
    return json_files
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes,mid_features=False):
            if mid_features:
                midoutputs=self.model(images,mid_features=True)
                return midoutputs
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)

    def get_img_tensor(path):
        # fake_image_np = cv2.imread(path)
        # fake_image_np = cv2.cvtColor(fake_image_np, cv2.COLOR_BGR2RGB)
        # tensor = torch.from_numpy(fake_image_np.transpose(2, 0, 1)).to(torch.uint8)
        im_pil = Image.open(path).convert('RGB')
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        tensor = transforms(im_pil)[None].to(args.device)
        return tensor
        pass
    imagelist=get_json_image(args.im_file)
    real_images=get_json_image('E:\\coco2voc\\VOC\\valimage\\type5_croping')
    real_tensors=[]
    mid_features=[]
    for index,data in enumerate(real_images):
        real_tensors.append(get_img_tensor(data))
        if (index+1)%8==0:
            real_tensor=torch.cat(real_tensors,dim=0)
            real_tensor_midoutputs_temp=model(real_tensor,None,mid_features=True)
            mid_features.append(real_tensor_midoutputs_temp[2])
            real_tensors=[]
    real_tensor_midoutputs=torch.cat(mid_features,dim=0)
    distance_folder="E:\\coco2voc\VOC\\grenateimg\\type5_G_out_scores1"
    for number,index in enumerate(imagelist[5773:]):
        im_pil = Image.open(index).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)
        if args.sliced:
            num_boxes = args.numberofboxes

            aspect_ratio = w / h
            num_cols = int(np.sqrt(num_boxes * aspect_ratio))
            num_rows = int(num_boxes / num_cols)
            slice_height = h // num_rows
            slice_width = w // num_cols
            overlap_ratio = 0.2
            slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
            predictions = []
            for i, slice_img in enumerate(slices):
                slice_tensor = transforms(slice_img)[None].to(args.device)
                with autocast():  # Use AMP for each slice
                    output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
                torch.cuda.empty_cache()
                labels, boxes, scores = output

                labels = labels.cpu().detach().numpy()
                boxes = boxes.cpu().detach().numpy()
                scores = scores.cpu().detach().numpy()
                predictions.append((labels, boxes, scores))

            merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
            labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
        else:
            midoutput=model(im_data,None,mid_features=True)
            kid = KernelInceptionDistance(subset_size=1)

            kid.update(real_tensor_midoutputs, real=True)
            kid.update(midoutput, real=False)
            kid_mean = kid.compute()
            img_path = os.path.join(distance_folder, os.path.basename(index))
            shutil.copy2(index, img_path)
            text_path = os.path.join(distance_folder, os.path.basename(index)) + ".txt"
            text_data=f"KID Mean: {kid_mean}"
            with open(text_path, 'w', encoding='utf-8') as file:
                file.write(text_data)
            if number%100==0:
                print(number/len(imagelist))
            # output = model(im_data, orig_size)
            # labels, boxes, scores = output

        # draw([im_pil], labels, boxes, scores, os.path.basename(index) ,orig_size,0.5,number)
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)











