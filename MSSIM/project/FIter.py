import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import re
import shutil
import json
def get_file(directory,str):
    """
    该函数用于获取指定目录下所有的 .txt 文件路径
    :param directory: 指定的目录路径
    :return: 包含所有 .txt 文件路径的列表
    """
    txt_files = []
    try:
        # 遍历指定目录及其子目录
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(str):
                    txt_files.append(os.path.join(root, file))
            dirs.clear()
    except FileNotFoundError:
        print(f"指定的目录 {directory} 未找到。")
    return txt_files





def filter(score_path, src_xml_dir, dest_dir, type_quota={'type2': True, 'type3s': 0, 'type4': True, 'type5': True}):
    """
    Parameter description:
    score_path: Location of the score file
    src_xml_dir: Path to the XML source folder
    dest_dir: Path to the target folder
    type_quota: Configuration for the number of each type to be selected (True means select all with score > 0.65, a number means select the specified quantity)

    """
    import json
    import os
    import shutil
    from collections import defaultdict

    os.makedirs(dest_dir, exist_ok=True)

    json_list = get_file(score_path, ".json")

    def extract_json_values(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    if not json_list:
        print("not find any .json file。")
        return

    type_dict = defaultdict(list)


    for json_file in json_list:
        try:
            entries = extract_json_values(json_file)
            for entry in entries:
                required_fields = ["Type", "name", "score1"]
                if all(field in entry for field in required_fields):
                    base_name = os.path.splitext(entry["name"])[0]
                    type_dict[entry["Type"]].append((base_name, entry["score1"]))
        except Exception as e:
            print(f"corp {json_file} error: {str(e)}")


    for type_name in ['type2', 'type3s', 'type4', 'type5']:
        if type_name not in type_dict or len(type_dict[type_name]) == 0:
            print(f"\n{type_name} no file")
            continue


        sorted_items = sorted(type_dict[type_name], key=lambda x: x[1], reverse=True)


        quota = type_quota.get(type_name, 0)
        if isinstance(quota, bool) and quota:

            selected = [item for item in sorted_items if item[1] > 0.65]
        elif isinstance(quota, int):

            selected = sorted_items[:quota]
        else:
            print(f"Invalid quota configuration: {type_name} The quota should be an integer or a boolean value.")
            continue

        print(f"\n{'=' * 30}")
        print(f"Processing {type_name}，select {len(selected)} files")


        success_count = 0
        for name, score in selected:  # 修正为遍历selected列表
            xml_file = f"{name}.xml"
            src_path = os.path.join(src_xml_dir, xml_file)
            dest_path = os.path.join(dest_dir, xml_file)

            try:
                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                    success_count += 1
                else:
                    print(f"文件不存在：{src_path}")
            except Exception as e:
                print(f"迁移 {xml_file} 失败：{str(e)}")

        print(f"Successfully migrated {success_count}/{len(selected)} files")

    # Finally, merge the additional files in the source directory (retain according to requirements).
    # shutil.copytree("E:\\coco2voc\\VOC\\dataset_chage1\\train\\xml - source",
    #                 dest_dir,
    #                 dirs_exist_ok=True)

if __name__ == '__main__':

    filter("","","")

    pass