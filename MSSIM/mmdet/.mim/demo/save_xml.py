
import xml.etree.ElementTree as ET
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