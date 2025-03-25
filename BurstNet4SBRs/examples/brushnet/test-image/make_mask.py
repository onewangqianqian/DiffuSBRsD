import cv2
import numpy as np
import json
# 假设这是您的多边形坐标（注意：应为整数坐标，并构成闭合多边形）
json_file='./type5.json'
with open(json_file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
polygon_points = np.array(data["shapes"][0]["points"], dtype=np.int32)
polygon_points = polygon_points.reshape((-1, 1, 2))  # opencv fillPoly需要的形状
# polygon_points1 = np.array(data["shapes"][1]["points"], dtype=np.int32)
# polygon_points1 = polygon_points1.reshape((-1, 1, 2))
# 创建一个空白图像（例如，全为1的图像）
image_height, image_width = data['imageHeight'], data["imageWidth"]
image = np.zeros((image_height, image_width), dtype=np.uint8) * 255  # opencv默认使用uint8，且255为白色

# 使用opencv的fillPoly函数将多边形内部填充为0（黑色）
# 注意：fillPoly期望的像素值为[0, 255]范围（对于uint8），因此我们用0表示黑色
cv2.fillPoly(image, [polygon_points], 255)
# cv2.fillPoly(image, [polygon_points1], 255)
# 此时，image已根据多边形坐标进行了修改，黑色区域表示多边形内部
# 如果您需要，可以将其保存为图像文件或进行其他处理
cv2.imwrite('type5_output_image.png', image)  # 保存为PNG文件