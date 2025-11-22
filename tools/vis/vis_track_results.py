import os
import cv2
import numpy as np
import random
from pathlib import Path


def read_mot_results(results_path, conf_thresh=0.0):
    """读取MOT格式的跟踪结果文件"""
    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0

            if conf < conf_thresh:
                continue

            if frame_id not in results:
                results[frame_id] = []
            results[frame_id].append({
                'track_id': track_id,
                'bbox': (x, y, w, h),
                'confidence': conf
            })
    return results


def visualize_bbox(image, bbox, track_id, confidence=None, color=None, thickness=2):
    """在图像上绘制边界框和跟踪ID"""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)

    # 随机颜色或指定颜色
    if color is None:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # 绘制边界框
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    # 准备标签文本
    label = f"ID: {track_id}"
    if confidence is not None:
        label += f", Conf: {confidence:.2f}"

    # 绘制标签背景和文本
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, y - label_size[1] - baseline), (x + label_size[0], y), color, -1)
    cv2.putText(image, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def main():
    # 设置路径和参数
    track_path = "/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/YOLOX_outputs/masort-stn_tuning-event-2-03conf-09iou-Recent/track_results_rgb/0292.txt"
    img_dir = "/mnt/sda/Disk_D/MMMUAV/test/0292/rgb_frame"
    output_dir = "/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/YOLOX_outputs/masort-stn_tuning-event-2-03conf-09iou-Recent-vis-results/292-rgb"
    show_visualization = False  # 是否显示可视化结果
    conf_thresh = 0.0  # 置信度阈值

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取跟踪结果
    print(f"Reading tracking results from {track_path}")
    mot_results = read_mot_results(track_path, conf_thresh)

    # 获取所有有跟踪结果的帧ID
    frame_ids = sorted(mot_results.keys())
    if not frame_ids:
        print("No tracking results found!")
        return

    print(f"Found {len(frame_ids)} frames with tracking results")

    # 为每个跟踪ID分配固定颜色
    track_colors = {}

    # 处理每一帧
    for frame_id in frame_ids:
        # 构建图像路径
        img_name = f"{frame_id:04d}.jpg"  # 假设图像文件名格式为4位数字
        img_path = os.path.join(img_dir, img_name)

        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping frame {frame_id}")
            continue

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}, skipping frame {frame_id}")
            continue

        # 获取当前帧的所有跟踪结果
        tracks = mot_results[frame_id]

        # 在图像上绘制所有边界框
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            conf = track['confidence']

            # 为该跟踪ID分配颜色（如果还没有）
            if track_id not in track_colors:
                track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # 可视化边界框
            image = visualize_bbox(
                image, bbox, track_id, conf,
                color=track_colors[track_id]
            )

        # 保存可视化结果
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, image)

        # 显示图像（如果需要）
        if show_visualization:
            cv2.imshow('Tracking Visualization', image)
            if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
                break

    if show_visualization:
        cv2.destroyAllWindows()

    print(f"Visualization completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()