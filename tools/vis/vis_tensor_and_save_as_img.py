import os
import numpy as np
import matplotlib.pyplot as plt


def load_and_visualize_npy(file_path, output_dir):
    """
    读取 .npy 文件保存的张量，并对所有通道取均值，以热力图的形式展示并保存。

    参数:
        file_path (str): .npy 文件的路径。
        output_dir (str): 输出图片的目录。
    """
    # 加载 .npy 文件
    tensor = np.load(file_path, allow_pickle=True)
    print(file_path, ":")
    print(f"加载的张量形状: {tensor.shape}")
    print("数据最小值:", np.min(tensor))
    print("数据最大值:", np.max(tensor))
    print("数据均值:", np.mean(tensor))

    if tensor.ndim == 3:
        tensor = tensor.reshape((tensor.shape[0], 1, *tensor.shape[1:]))  # 扩展为 (B,1,W,H)
    elif tensor.ndim == 2:
        tensor = tensor.reshape((1, 1, *tensor.shape))  # 处理二维数据 (1,1,W,H)
    elif tensor.ndim != 4:
        raise ValueError(f"无效维度 {tensor.ndim}，仅支持2D/3D/4D张量")

    file_name_with_ext = os.path.basename(file_path)
    file_name = os.path.splitext(file_name_with_ext)[0]

    batch_size, num_channels, width, height = tensor.shape
    for batch_idx in range(batch_size):
        mean_data = np.mean(tensor[batch_idx, :, :, :], axis=0)

        plt.figure()
        plt.imshow(mean_data, cmap="hot")
        plt.title(f"{file_name} - Batch {batch_idx}")  # 在标题中添加批次信息
        plt.colorbar(label='Intensity')
        plt.axis('off')

        # 生成包含批次索引的文件名
        output_path = os.path.join(output_dir, f"{file_name}.pdf")
        plt.savefig(output_path, bbox_inches='tight',pad_inches=0.05)
        plt.show()  # 显示图像
        plt.close()
        print(f"已保存: {output_path}")


def process_all_npy_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(input_dir, file_name)
            print(f"\n正在处理文件: {file_path}")
            try:
                load_and_visualize_npy(file_path, output_dir)
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {str(e)}")


# 示例用法
input_dir = "/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/saved_npy/"
output_dir = "/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/vis_tensors_and_save_as_img-0001/"

process_all_npy_files(input_dir, output_dir)