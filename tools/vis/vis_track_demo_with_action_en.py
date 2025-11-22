import cv2
import numpy as np
import os

# 跟踪结果文件路径
tracking_results_path = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/YOLOX_outputs/track_demos/track_results/multi-qifei.txt'

# 图片路径
frames_dir = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/videos-for-demo/multi-qifei/'

# 输出视频路径
output_video_dir = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/videos-for-demo/multi-qifei/results/'

# 创建输出目录（如果不存在）
if not os.path.exists(output_video_dir):
    os.makedirs(output_video_dir)


# 读取跟踪结果
def read_tracking_results(file_path):
    tracking_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            # 假设每行格式为：frame_id, track_id, x, y, w, h, confidence, class_id, ...
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            confidence = float(parts[6])
            class_id = int(parts[7]) if len(parts) > 7 else 0

            if frame_id not in tracking_data:
                tracking_data[frame_id] = []
            tracking_data[frame_id].append({
                'track_id': track_id,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'confidence': confidence,
                'class_id': class_id
            })
    return tracking_data


tracking_data = read_tracking_results(tracking_results_path)

# 获取所有图片文件
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

# 读取第一张图片以获取尺寸
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width, layers = first_frame.shape

# 定义视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_video_dir, 'multi-qifei.mp4')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# 设置帧范围
start_frame = 140  # 从第70帧开始
end_frame = None  # 到第140帧结束

# 设置置信度阈值
confidence_threshold = 0.7  # 只显示置信度高于0.6的目标

# 遍历每一帧图片
for frame_file in frame_files:
    frame_id = int(frame_file.split('.')[0])

    # 检查是否在指定帧范围内
    if end_frame is not None and frame_id > end_frame:
        break
    if frame_id < start_frame:
        continue

    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)

    if frame_id in tracking_data:
        tracks = tracking_data[frame_id]

        # 过滤置信度低于阈值的目标
        filtered_tracks = [track for track in tracks if track['confidence'] >= confidence_threshold]
        drone_count = len(filtered_tracks)  # 当前帧检测到的高置信度无人机数量

        # 在左上角显示无人机数量
        text = f"Detected {drone_count} drones"
        text_color = (0, 255, 0)  # 绿色
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # 绘制每个跟踪目标的边界框和标签
        for track in filtered_tracks:
            x, y, w, h = track['x'], track['y'], track['w'], track['h']
            track_id = track['track_id']
            confidence = track['confidence']

            # 将坐标转换为整数
            x, y, w, h = int(x), int(y), int(w), int(h)

            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 显示“Approaching”提示在边界框正上方，字体大小根据框宽度自适应
            approaching_text = "Up"
            font_scale = max(w / 90.0, 0.2)  # 根据框宽度调整字体大小，至少为0.5
            text_size = cv2.getTextSize(approaching_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x + (w - text_size[0]) // 2  # 居中对齐
            text_y = y - 10  # 放在边界框上方
            cv2.putText(frame, approaching_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

    # 将处理后的帧写入视频
    out.write(frame)

# 释放资源
out.release()

print(f"Tracking visualization video has been saved to {output_video_path}")