import os

import numpy as np
import matplotlib.pyplot as plt

def load_and_visualize_npy(file_path):
    """
    读取 .npy 文件保存的张量，并对所有通道取均值，以热力图的形式展示。

    参数:
        file_path (str): .npy 文件的路径。
    """
    # 加载 .npy 文件
    tensor = np.load(file_path, allow_pickle=True)  # 允许加载 pickled 数据
    print(f"加载的张量形状: {tensor.shape}")
    print("数据最小值:", np.min(tensor))
    print("数据最大值:", np.max(tensor))
    print("数据均值:", np.mean(tensor))

    # 检查张量形状
    if tensor.ndim != 4:
        raise ValueError(f"输入张量的形状应为 (Batch, channel, W, H)，但实际形状为 {tensor.shape}")
    file_name_with_ext = os.path.basename(file_path)
    # 对所有通道取均值并可视化
    batch_size, num_channels, width, height = tensor.shape
    for batch_idx in range(batch_size):
        for channel_idx in range(num_channels):
            # 对当前批次的所有通道取均值
            # mean_data = np.mean(tensor[batch_idx, :, :, :], axis=0)
            data = tensor[batch_idx, channel_idx, :, :]
            # 可视化热力图
            # plt.imshow(mean_data,cmap="hot")  # 使用热力图颜色映射
            plt.imshow(data, cmap='hot')  # 使用热力图颜色映射
            plt.title(f"{file_name_with_ext}")
            plt.colorbar(label='Intensity')
            plt.axis('off')  # 关闭坐标轴
            plt.show()

# 示例：读取并可视化本地的.npy文件
# file_path = "/mnt/sda/Disk_B/jinjie/Tracking/YOLOv11/saved_npy_UNet/x8_after_decoder_1.npy"  # 替换为你的.npy文件路径
# file_path = "/mnt/sda/Disk_B/jinjie/Tracking/YOLOv11/saved_npy_UNet/x_rgb_base_1.npy"  # 替换为你的.npy文件路径
file_path = "/mnt/sda/Disk_B/jinjie/Tracking/YOLOv11_two_stream/saved_npy/x_ir_fused_1.npy"  # 替换为你的.npy文件路径
# file_path = "/mnt/sda/Disk_B/jinjie/Tracking/YOLOv11/saved_npy/x_ir_output_1.npy"  # 替换为你的.npy文件路径
# file_path = "/mnt/sda/Disk_B/jinjie/Tracking/YOLOv11/saved_npy/x_output_1.npy"

load_and_visualize_npy(file_path)