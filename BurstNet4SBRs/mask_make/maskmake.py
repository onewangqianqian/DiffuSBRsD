import cv2
import numpy as np

# Define the image size (height, width)
import os
import glob
import shutil
import json
# Define the four standpoints (coordinates)

def gather_jpg_files(source_folder, destination_folder=None):
    # Get all JPEG files in the source folder
    jpg_files = glob.glob(os.path.join(source_folder, '*.jpg')) + glob.glob(os.path.join(source_folder, '*.jpeg'))
    jpg_file_names = [os.path.basename(file) for file in jpg_files]

    if destination_folder:
        # Create destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Move or copy the files to the destination folder
        for file in jpg_files:
            shutil.copy(file, destination_folder)  # Use shutil.move() if you want to move instead of copy

    return jpg_file_names

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def bbox_to_standpoints(bbox):
    x, y, width, height = bbox
    top_left = [x, y]
    top_right = [x + width, y]
    bottom_right = [x + width, y + height]
    bottom_left = [x, y + height]
    return [top_left, top_right, bottom_right, bottom_left]

if __name__ == '__main__':
    path = 'data/BurstData/cls3'
    json_path = 'mask_make/instances_train2017.json'
    data = read_json_file(json_path)
    # test = gather_jpg_files(path)


    pass

    # print(len(data['images']))
    # print(len(data['annotations']))
    #
    # for ann in data['annotations']:
    #     if ann['image_id']== 502022033017300801:
    #         points=np.array(bbox_to_standpoints(ann['bbox']))
    #         mask = np.zeros((600, 800), dtype=np.uint8)
    #         cv2.fillPoly(mask, [points], 255)
    #         cv2.imwrite('mask_make/'+str(ann['image_id'])+'.png', mask)
    #         print(ann )




    # points = np.array([[100, 100], [400, 100], [400, 400], [100, 400]])
    #
    # # Create a black image
    # mask = np.zeros((img_height, img_width), dtype=np.uint8)
    #
    # # Fill the defined polygon with white color
    # cv2.fillPoly(mask, [points], 255)
    #
    # # Save the mask image
    # cv2.imwrite('mask.png', mask)
    #
    # print("Mask created and saved as 'mask.png'")
