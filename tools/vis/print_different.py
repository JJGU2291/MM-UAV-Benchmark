import os
from collections import defaultdict


def read_mot_gt(gt_path):
    """读取MOT格式的标注文件，返回{(frame, id)}集合"""
    mot_set = set()
    if not os.path.exists(gt_path):
        return mot_set

    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # 跳过空行和注释行
                continue
            parts = line.split(',')  # MOT格式通常用逗号分隔，也可能是空格，根据实际情况调整
            if len(parts) < 2:
                continue  # 无效行
            try:
                frame = int(parts[0])  # 第1列是帧号
                obj_id = int(parts[1])  # 第2列是目标ID
                mot_set.add((frame, obj_id))
            except (ValueError, IndexError):
                continue  # 跳过格式错误的行
    return mot_set


def main():
    root_dir = "/mnt/sda/Disk_D/MMMUAV/test/"
    result = defaultdict(lambda: defaultdict(list))  # 结构: {序列名: {帧号: [目标ID列表]}}

    # 遍历所有序列文件夹
    for seq_name in os.listdir(root_dir):
        seq_path = os.path.join(root_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue  # 只处理文件夹

        # 定义两种模态的标注文件路径
        rgb_gt_path = os.path.join(seq_path, "gt_rgb", "gt.txt")
        ir_gt_path = os.path.join(seq_path, "gt_ir", "gt.txt")

        # 检查文件是否存在
        if not os.path.exists(rgb_gt_path):
            print(f"警告: {seq_name} 缺少RGB标注文件 {rgb_gt_path}，跳过该序列")
            continue
        if not os.path.exists(ir_gt_path):
            print(f"警告: {seq_name} 缺少IR标注文件 {ir_gt_path}，跳过该序列")
            continue

        # 读取标注数据
        rgb_set = read_mot_gt(rgb_gt_path)
        ir_set = read_mot_gt(ir_gt_path)

        # 找出RGB存在但IR不存在的目标(帧, ID)对
        missing_in_ir = rgb_set - ir_set

        # 整理结果
        for (frame, obj_id) in missing_in_ir:
            result[seq_name][frame].append(obj_id)

    # 输出统计结果
    print("统计结果：RGB存在但IR不存在的目标情况")
    print("=" * 80)
    for seq_name in sorted(result.keys()):
        frame_info = result[seq_name]
        if not frame_info:
            continue
        print(f"序列: {seq_name}")
        print("-" * 40)
        for frame in sorted(frame_info.keys()):
            obj_ids = sorted(frame_info[frame])
            print(f"  帧 {frame}: 目标ID {obj_ids}")
        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()