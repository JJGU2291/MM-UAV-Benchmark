import cv2
import numpy as np
import os
from os.path import join
import glob
import time


def visualize_tracking_results(txt_path, img_dir, output_path, duration_seconds=None, fps=60):
    """
    可视化跟踪结果并保存为视频，同时显示帧率
    :param txt_path: 跟踪结果文件路径（MOT格式）
    :param img_dir: 图片文件夹路径
    :param output_path: 输出视频路径
    :param duration_seconds: 要保存的视频时长（秒），None表示保存全部
    :param fps: 视频帧率，默认为25
    """
    # 读取跟踪结果
    tracks = {}
    with open(txt_path, 'r') as f:
        for line in f:
            # 解析MOT格式的行: <frame_id> <id> <x> <y> <w> <h> <score> <...>
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])

            # 将坐标存储为整数
            x, y, w, h = int(x), int(y), int(w), int(h)

            if frame_id not in tracks:
                tracks[frame_id] = []
            tracks[frame_id].append((track_id, x, y, w, h))

    # 获取所有图片路径
    img_files = sorted(glob.glob(join(img_dir, '*')))
    img_files.sort()

    # 获取视频参数
    img = cv2.imread(img_files[0])
    height, width, _ = img.shape

    # 计算要处理的帧数
    if duration_seconds is not None:
        total_frames = min(int(duration_seconds * fps), len(img_files))
    else:
        total_frames = len(img_files)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 颜色映射，为每个目标分配固定颜色
    color_map = {}

    # 计算帧率相关变量
    start_time = time.time()
    frame_count = 0

    # 处理每一帧
    for frame_idx in range(1, total_frames + 1):
        # 获取当前帧的图片
        img_path = img_files[frame_idx - 1]
        img = cv2.imread(img_path)

        # 如果有跟踪结果，绘制边界框
        if frame_idx in tracks:
            for track_id, x, y, w, h in tracks[frame_idx]:
                # 为每个ID分配颜色
                if track_id not in color_map:
                    color_map[track_id] = np.random.randint(0, 255, size=3).tolist()
                color = color_map[track_id]

                # 绘制边界框
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # 绘制ID文本
                cv2.putText(img, f'ID: {track_id}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 计算并显示当前帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # 显示当前帧率在右上角
        cv2.putText(img, f'FPS: {30:.2f}', (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 写入视频
        video_writer.write(img)

    # 释放资源
    video_writer.release()
    print(f"跟踪可视化视频已保存至: {output_path}")


if __name__ == "__main__":
    # 文件路径
    txt_path = "/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/YOLOX_outputs/All-results-demo/track_results_ir/0001.txt"
    img_dir = "/mnt/sda/Disk_D/MMMUAV/test/0001/ir_frame/"
    output_path = "/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/Track_Demo/0001_ir.MP4"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 只保存前10秒的视频
    desired_duration = 10  # 秒

    # 执行可视化
    visualize_tracking_results(txt_path, img_dir, output_path,desired_duration)