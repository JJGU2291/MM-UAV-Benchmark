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
output_video_path = os.path.join(output_video_dir, 'multi-qifei-behavior.mp4')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# 设置帧范围
start_frame = 70  # 从第70帧开始
end_frame = None  # 到第140帧结束

# 设置置信度阈值
confidence_threshold = 0.79  # 只显示置信度高于0.6的目标

# 用于保存目标的历史位置和宽高（用于计算行为和速度）
track_history = {}

# 手动设置的行为列表
predefined_behavior = {
    "Static": [0, 30],  # 0-20帧静止
    "Move Away": [31, end_frame]  # 21帧到最后为远离
}

# 是否显示未来位置预测的标志
show_prediction = False  # 设置为True显示预测，False不显示

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

        # 设置无人机数量文本的颜色
        text_color = (0, 255, 0)  # "Detected" 和 "drones" 保持绿色

        # 在左上角显示无人机数量
        text = f"Detected {drone_count} drones"
        # 绘制文本的两个部分："Detected " 和 " drones" 用绿色
        cv2.putText(frame, "Detected ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, " drones", (10 + 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # 绘制无人机数量，根据数量改变颜色
        if drone_count > 0:
            count_color = (0, 0, 255)  # 数字用红色
        else:
            count_color = (0, 255, 0)  # 数字用绿色
        count_text = str(drone_count)
        count_text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        count_text_x = 15 + 100  # 调整位置以确保数字显示在 "Detected" 和 "drones" 之间
        cv2.putText(frame, count_text, (count_text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, count_color, 2)

        # 绘制每个跟踪目标的边界框和标签
        for track in filtered_tracks:
            track_id = track['track_id']
            x, y, w, h = track['x'], track['y'], track['w'], track['h']
            confidence = track['confidence']

            # 将坐标转换为整数
            x, y, w, h = int(x), int(y), int(w), int(h)

            # 记录目标的历史位置和宽高
            if track_id not in track_history:
                track_history[track_id] = {
                    'positions': [],
                    'sizes': [],
                    'speeds': []
                }
            center_x, center_y = x + w // 2, y + h // 2
            track_history[track_id]['positions'].append((center_x, center_y))  # 中心点
            track_history[track_id]['sizes'].append((w, h))

            # 保留最近5帧的历史记录
            if len(track_history[track_id]['positions']) > 5:
                track_history[track_id]['positions'].pop(0)
            if len(track_history[track_id]['sizes']) > 5:
                track_history[track_id]['sizes'].pop(0)

            # 计算速度（单位像素/帧）
            speed = 0.0
            if len(track_history[track_id]['positions']) >= 2:
                prev_pos = track_history[track_id]['positions'][-2]
                curr_pos = track_history[track_id]['positions'][-1]
                speed = np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)
                track_history[track_id]['speeds'].append(speed)
                if len(track_history[track_id]['speeds']) > 5:
                    track_history[track_id]['speeds'].pop(0)
                speed = np.mean(track_history[track_id]['speeds'])  # 使用最近几帧的平均速度

            # 手动设置行为信息
            if predefined_behavior["Static"][0] <= frame_id <= predefined_behavior["Static"][1]:
                behavior = "Static"
            elif predefined_behavior["Move Away"][0] <= frame_id <= predefined_behavior["Move Away"][1]:
                behavior = "Move Away"
            else:
                behavior = "Moving"

            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 显示行为信息在边界框上方
            font_scale = max(w / 90.0, 0.5)  # 根据框宽度调整字体大小
            text_size = cv2.getTextSize(behavior, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x + (w - text_size[0]) // 2  # 居中对齐
            text_y = y - 10  # 放在边界框上方

            if behavior == "Approaching":
                text_color_behavior = (0, 0, 255)  # 红色
            elif behavior == "Move Away":
                text_color_behavior = (255, 0, 0)  # 蓝色
            else:
                text_color_behavior = (0, 255, 0)  # 绿色

            cv2.putText(frame, behavior, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color_behavior, 2)

            # 显示速度信息在边界框下方
            speed_text = f"{speed:.2f} m/s"
            speed_text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            speed_text_x = x + (w - speed_text_size[0]) // 2
            speed_text_y = y + h + 20
            cv2.putText(frame, speed_text, (speed_text_x, speed_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 绘制未来位置预测点（直接从标注中读取未来位置）
            if show_prediction:
                future_frame_id = frame_id + 10
                if future_frame_id in tracking_data:
                    future_tracks = tracking_data[future_frame_id]
                    future_track = next((ft for ft in future_tracks if ft['track_id'] == track_id), None)
                    if future_track is not None and frame_id % 10 == 0:  # 每10帧显示一次预测位置
                        future_x, future_y, future_w, future_h = int(future_track['x']), int(future_track['y']), int(
                            future_track['w']), int(future_track['h'])
                        future_center_x = future_x + future_w // 2
                        future_center_y = future_y + future_h // 2
                        cv2.circle(frame, (future_center_x, future_center_y), 5, (0, 255, 255), -1)  # 绘制实心圆点

    # 将处理后的帧写入视频
    out.write(frame)

# 释放资源
out.release()

print(
    f"Tracking visualization video with behavior, speed, and future position prediction has been saved to {output_video_path}")