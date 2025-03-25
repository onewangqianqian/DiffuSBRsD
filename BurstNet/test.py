# from diffusers import AutoencoderKL
#
# url = "data/ckpt/vae/sdxl.vae.safetensors"  # can also be a local file
# model = AutoencoderKL.from_single_file(url)
# print(model)

# from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
# print(is_wandb_available())
import os
from torch.utils.data import DataLoader,TensorDataset
import torch
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
import numpy as np
def count_all_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

def dataset():
    from datasets import load_dataset
    dataset = load_dataset("imagefolder", data_dir="data/BurstData")

    dataloader=DataLoader(dataset['train'],batch_size=2,shuffle=True)
    for x,y in dataloader:
        print(x)
        break

def coll_fn():
    class SimpleCustomBatch:
        def __init__(self, data):
            transposed_data = list(zip(*data))
            self.inp = torch.stack(transposed_data[0], 0)
            self.tgt = torch.stack(transposed_data[1], 0)

        # custom memory pinning method on custom type
        def pin_memory(self):
            self.inp = self.inp.pin_memory()
            self.tgt = self.tgt.pin_memory()
            return self
        def test(self):
            return self.inp


    def collate_wrapper(batch):
        return SimpleCustomBatch(batch)


    inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    dataset = TensorDataset(inps, tgts)

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                        pin_memory=True)

    for batch_ndx, sample in enumerate(loader):
        print(sample.test())
        # print(sample.tgt)

def create_brustdat_json():

    data_all={
        'data':[]
    }
    for root, dirs, files in os.walk('data/BurstData/cls3'):
        for name in files:
            caption='Type IV Solar radio bursts'
            # bytes_data =caption.encode('utf-8')
            height='410'
            # height=height.encode('utf-8')
            width= '620'
            # width=width.encode('utf-8')
            image=name
            # image=image.encode('utf-8')
            data = {
                'caption': caption,
                'height': height,
                'width': width,
                'image': image

            }
            data_all['data'].append(data)
    print(data_all)
    file_path = "data/BurstData/cls3_cropping.json"
    with open(file_path, 'w') as json_file:
        json.dump(data_all, json_file, indent=4)  # indent=4 for pretty printing


    print(f"Data has been written to {file_path}")
    #
    # with open(file_path, 'r') as json_file:
    #     loaded_data = json.load(json_file)
    #
    # print("Loaded data:")
    # print(loaded_data)

    pass

def sem_show():
    import numpy as np
    import matplotlib.pyplot as plt

    # Assuming the mask is for a 2D array (image)
    mask_values = [1, 187, 189, 2, 193, 2, 196, 3, 200, 7, 233, 3, 237, 2538, 2778, 233, 3017, 230, 3248, 5, 3254, 228, 3484, 5, 3492, 227, 3720, 7, 3730, 225, 3956, 9, 3968, 236, 4205, 236, 4442, 236, 4681, 235, 4918, 235, 5155, 235, 5393, 233, 5630, 233, 5864, 1, 5866, 233, 6101, 234, 6337, 234, 6572, 235, 6808, 1004, 7813, 1, 7818, 166, 7985, 66, 8052, 640, 8693, 69, 8763, 2, 8766, 159, 8931, 229, 9169, 58, 9228, 166, 9406, 223, 9644, 220, 9882, 46, 9929, 170, 10120, 44, 10165, 168, 10357, 39, 10397, 24, 10424, 9, 10434, 135, 10595, 37, 10633, 23, 10660, 144, 10833, 35, 10870, 2, 10873, 19, 10896, 1, 10898, 137, 11056, 1, 11058, 1, 11069, 35, 11105, 23, 11129, 1, 11131, 139, 11294, 1, 11307, 27, 11335, 1, 11344, 17, 11362, 2, 11366, 142, 11523, 2, 11526, 1, 11528, 1, 11530, 1, 11544, 28, 11574, 169, 11755, 2, 11758, 3, 11762, 2, 11766, 2, 11781, 24, 11806, 2, 11816, 17, 11834, 2, 11838, 140, 11991, 13, 12006, 1, 12018, 21, 12040, 1, 12042, 31, 12074, 140, 12227, 13, 12241, 2, 12256, 18, 12278, 172, 12463, 17, 12493, 16, 12516, 170, 12697, 19, 12730, 15, 12753, 169, 12933, 19, 12966, 15, 12985, 2, 12988, 128, 13121, 67, 13203, 14, 13221, 96, 13328, 21, 13361, 12, 13377, 47, 13439, 14, 13460, 81, 13569, 14, 13608, 1, 13615, 46, 13675, 14, 13694, 80, 13805, 13, 13853, 44, 13910, 15, 13929, 2, 13932, 3, 13936, 41, 13980, 10, 13995, 2, 14003, 1, 14045, 9, 14078, 2, 14092, 40, 14146, 15, 14165, 48, 14218, 1, 14281, 9, 14333, 35, 14382, 15, 14404, 38, 14570, 1, 14575, 5, 14581, 23, 14617, 16, 14637, 2, 14640, 37, 14812, 2, 14817, 23, 14852, 17, 14870, 1, 14873, 35, 15026, 2, 15044, 1, 15048, 1, 15053, 22, 15088, 17, 15112, 30, 15290, 2, 15295, 1, 15297, 15, 15323, 18, 15348, 6, 15355, 8, 15364, 13, 15501, 5, 15526, 1, 15533, 15, 15559, 18, 15581, 2, 15584, 3, 15588, 3, 15592, 3, 15596, 2, 15599, 10, 15739, 3, 15769, 15, 15795, 18, 15814, 1, 15817, 2, 15820, 10, 15835, 9, 16006, 13, 16030, 19, 16057, 2, 16075, 2, 16241, 15, 16266, 19, 16292, 2, 16295, 1, 16297, 1, 16308, 1, 16311, 1, 16480, 11, 16501, 20, 16526, 1, 16528, 3, 16533, 1, 16536, 2, 16540, 2, 16543, 3, 16547, 1, 16718, 1, 16720, 7, 16737, 20, 16764, 4, 16772, 10, 16958, 1, 16972, 21, 17015, 2, 17189, 2, 17194, 1, 17207, 22, 17237, 1, 17241, 1, 17244, 6, 17251, 1, 17430, 1, 17443, 22, 17466, 1, 17469, 2, 17473, 1, 17477, 2, 17480, 6, 17625, 2, 17678, 23, 17713, 1, 17715, 6, 17862, 1, 17913, 24, 17951, 3, 17955, 2, 17958, 1, 18149, 24, 18184, 6, 18334, 1, 18384, 25, 18410, 1, 18421, 5, 18570, 1, 18619, 26, 18647, 1, 18660, 1, 18855, 26, 18882, 2, 18886, 2, 18889, 5, 19074, 7, 19090, 27, 19128, 2, 19309, 6, 19316, 1, 19325, 28, 19364, 2, 19544, 7, 19561, 28, 19600, 1, 19751, 1, 19755, 1, 19770, 1, 19773, 3, 19777, 13, 19792, 1, 19796, 29, 19836, 1, 19991, 1, 20010, 2, 20013, 2, 20017, 9, 20028, 33, 20072, 1, 20223, 10, 20234, 3, 20238, 1, 20242, 2, 20245, 3, 20249, 3, 20253, 8, 20262, 1, 20264, 33, 20465, 1, 20489, 6, 20496, 1, 20500, 33, 20721, 1, 20725, 6, 20732, 1, 20734, 1, 20736, 33, 20946, 1, 20948, 6, 20957, 3, 20961, 8, 20970, 35, 21184, 2, 21187, 2, 21197, 7, 21208, 33, 21433, 8, 21444, 33, 21669, 8, 21680, 33, 21905, 44, 22143, 1, 22145, 2, 22148, 1, 22150, 35, 22380, 3, 22384, 37, 22617, 40, 22854, 39, 23091, 38, 23328, 37, 23564, 37, 23801, 36, 24038, 35, 24272, 3, 24276, 33, 24512, 33, 24743, 1, 24750, 8, 24759, 22, 24986, 2, 25001, 16, 25239, 14, 25478, 11, 25714, 11, 25950, 11, 26186, 11, 26422, 11, 26658, 11, 26894, 11, 27130, 11, 27366, 11, 27602, 11, 27837, 12, 28073, 12, 28308, 13, 28544, 13, 28779, 14, 29015, 14, 29250, 15, 29486, 15, 29721, 16, 29957, 16, 30192, 17, 30428, 17, 30664, 17, 30900, 17, 31135, 18, 31371, 18, 31608, 17, 31844, 17, 32080, 19, 32100, 2, 32317, 19, 32553, 17, 32789, 16, 33026, 16, 33261, 16, 33281, 3, 33497, 16, 33518, 2, 33734, 15, 33750, 1, 33752, 4, 33757, 1, 33970, 25, 34206, 26, 34442, 26, 34469, 1, 34678, 1, 34680, 1, 34682, 1, 34686, 18, 34705, 1, 34918, 1, 34922, 20, 35158, 21, 35394, 11, 35406, 6, 35413, 3, 35629, 11, 35641, 6, 35648, 3, 35652, 2, 35864, 27, 36098, 14, 36124, 2, 36334, 11, 36346, 3, 36350, 2, 36353, 2, 36357, 6, 36364, 1, 36570, 35, 36609, 1, 36734, 42, 36778, 3, 36806, 36, 36968, 45, 37014, 3, 37042, 36, 37202, 51, 37278, 37, 37436, 53, 37514, 13, 37529, 26, 37669, 57, 37750, 24, 37775, 16, 37902, 60, 37986, 41, 38135, 64, 38219, 48, 38268, 1, 38366, 70, 38456, 50, 38600, 73, 38691, 50, 38744, 1, 38836, 74, 38926, 52, 38979, 3, 39072, 74, 39162, 60, 39307, 76, 39395, 66, 39542, 77, 39630, 67, 39778, 78, 39866, 69, 39957, 1, 39961, 2, 40014, 83, 40102, 33, 40139, 34, 40175, 1, 40178, 13, 40192, 8, 40250, 83, 40338, 33, 40379, 30, 40414, 7, 40423, 4, 40433, 3, 40484, 85, 40574, 32, 40617, 28, 40654, 1, 40659, 3, 40669, 2, 40720, 85, 40806, 1, 40810, 32, 40855, 27, 40955, 86, 41046, 31, 41093, 27, 41190, 87, 41278, 35, 41332, 26, 41425, 89, 41515, 33, 41570, 26, 41617, 1, 41619, 1, 41621, 1, 41658, 126, 41810, 59, 41884, 136, 42048, 57, 42108, 2, 42120, 135, 42287, 65, 42353, 137, 42525, 201, 42766, 43, 42811, 2, 42817, 145, 43001, 44, 43047, 2, 43053, 1, 43055, 1, 43057, 1, 43060, 2, 43063, 134, 43238, 43, 43283, 2, 43302, 131, 43475, 46, 43525, 5, 43532, 137, 43714, 6, 43722, 2, 43725, 30, 43756, 1, 43760, 146, 43960, 33, 43997, 145, 44193, 36, 44233, 147, 44431, 33, 44465, 2, 44469, 148, 44670, 33, 44704, 151, 44911, 1, 44915, 24, 44941, 152, 45148, 25, 45174, 1, 45177, 155, 45385, 24, 45415, 155, 45623, 22, 45650, 158, 45860, 19, 45880, 3, 45885, 162, 46096, 17, 46115, 171, 46332, 17, 46351, 173, 46548, 3, 46568, 194, 46782, 8, 46804, 198, 47015, 16, 47039, 205, 47249, 21, 47273, 8424]

    # Define the image dimensions (for example, 100x100)
    image_dim = (236, 236)

    # Create an empty mask
    mask = np.zeros(image_dim)

    # Apply the mask values (assuming they are positions in the flattened array)
    for val in mask_values:
        row = val // image_dim[1]
        col = val % image_dim[1]
        mask[row, col] = 1

    # Visualize the mask
    plt.imshow(mask, cmap='gray')
    plt.savefig('mask_make/plot.png')
    pass


def polygon_to_mask(image_shape, polygons, fill=255):
    list = []
    for index in polygons:
        temp = (index[0], index[1])
        list.append(temp)
    polygons = []
    polygons.append(list)
    image = Image.new('L', image_shape, 0)  # 创建一个全黑的灰度图像
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        # polygon = list(map(tuple, polygon))  # 确保多边形坐标是元组形式
        draw.polygon(polygon, outline=1, fill=fill)  # 绘制多边形，这里outline参数实际未使用
    return np.array(image)


def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def cropping(path,dir):
    temp_path=dir+'/'+path
    image = cv2.imread(temp_path)

    # Check if the image was successfully loaded
    if image is None:
        raise FileNotFoundError(f"Unable to load image at path: {temp_path}")

    # Define the coordinates for cropping (example coordinates)
    # Coordinates are in the format (x, y)
    top_left = (100, 70)
    bottom_right = (720, 480)

    # Crop the image using the coordinates
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


    # Save the cropped image
    cropped_image_path = dir+'/'+path
    cv2.imwrite(cropped_image_path, cropped_image)
def batch_cropping():

    from pathlib import Path
    source='data/BurstData/cls4'
    # Load the image
    dir_path = Path(source)

    file_names = [f.name for f in dir_path.iterdir() if f.is_file()]
    for index in file_names:
        cropping(index,source)


def bbox_to_points(bbox):
    """
    Convert a bounding box to its coordinate points.

    Parameters:
    bbox (tuple): A tuple of (x, y, width, height) representing the bounding box.

    Returns:
    list: A list of coordinate points (x, y) of the four corners of the bounding box.
    """
    x, y, width, height = bbox

    # Calculate the coordinates of the four corners
    top_left = (x, y)
    top_right = (x + width, y)
    bottom_left = (x, y + height)
    bottom_right = (x + width, y + height)

    return [top_left, top_right, bottom_left, bottom_right]
def cls_cropping_json():
    train_path='data/BurstData/instances_train2017.json'
    data_all={
        'data':[]
    }
    with open(train_path,'r') as file:
        train_data=json.load(file)
    def find_imagename(id):
        for index in train_data['images']:
            if index['id']== id:
                return index['file_name']
    syms=['Ⅱ','Ⅲ','Ⅲs','Ⅳ','Ⅴ']
    for index in train_data['annotations']:
        img_name=find_imagename(index['image_id'])
        caption="Type "+syms[index['category_id']-1]+ " Solar radio bursts"
        temp_box=[index['bbox'][0]-100,index['bbox'][1]-70,index['bbox'][2],index['bbox'][3]]
        data = {
            'caption': caption,
            'height': 620,
            'width': 410,
            'img_name':img_name,
            'box':temp_box,
            'segmentation':bbox_to_points(temp_box),
            'category_id':index['category_id']


        }
        data_all['data'].append(data)
    print(len(train_data['annotations']),len(data_all['data']))
    file_path='data/BurstData/train_cropping.json'
    # Save the data to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data_all, json_file, indent=4)

    print(f"Data successfully saved to {file_path}")
    pass

if __name__ == '__main__':
    # create_brustdat_json()
    # from datasets import load_dataset
    # dataset = load_dataset("json", data_files="data/BurstData/cls3/data.json")
    # print(dataset)
    # current_directory = os.getcwd()
    # print(current_directory)


    # Display the cropped image
    # segmentation=[
    #     [
    #         322,
    #         175
    #     ],
    #     [
    #         618,
    #         175
    #     ],
    #     [
    #         618,
    #         332
    #     ],
    #     [
    #         322,
    #         332
    #     ]
    # ]
    # mask=polygon_to_mask((620,410),segmentation)
    # cv2.imwrite('output/maks.png',mask)
    # cropping('image1mask.png','examples/brushnet/test-image')
    cls_cropping_json()
    pass