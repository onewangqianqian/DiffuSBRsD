import numpy as np
from PIL import Image
import os

# 配置参数
image_path = "output/jiazhao/Arecibo-Observatory_20220929_114500_63.jpg"  # 输入图像路径
output_dir = "output/jiazhao"  # 输出文件夹
# noise_scale = 25  # 噪声强度系数（控制标准差为25）

total_steps = 5  # 加噪次数
final_noise_std = 700 # 最终噪声标准差（覆盖0-255全范围）

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载图像并转为浮点矩阵（保留原始值范围）
original = Image.open(image_path)
img_float = np.array(original, dtype=np.float32)

# 计算每步噪声标准差（五次叠加后总方差为 final_std^2）
step_std = final_noise_std / np.sqrt(total_steps)  # 根据方差可加性计算

for step in range(total_steps):
    # 生成当前步骤噪声（均值为0，标准差按步数递增）
    noise = np.random.normal(
        loc=0,
        scale=step_std * (step + 1),  # 逐步增强噪声
        size=img_float.shape
    )

    # 叠加噪声（保持浮点运算）
    noisy_img = img_float + noise

    # 保存中间结果（带归一化处理）
    temp_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    Image.fromarray(temp_img).save(
        os.path.join(output_dir, f"step_{step + 1}.png")  # 使用PNG避免压缩损失
    )

# 最终结果标准化处理（强制转为标准正态分布）
final_img = (noisy_img - noisy_img.mean()) / noisy_img.std()  # 标准化
final_img = np.clip(final_img * 127.5 + 127.5, 0, 255).astype(np.uint8)  # 映射到0-255

# 保存最终状态
Image.fromarray(final_img).save(os.path.join(output_dir, "final_normalized.png"))

print(f"强噪声生成完成，过程图像保存至 {output_dir}")