import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# 跟踪结果文件路径
tracking_results_path = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/YOLOX_outputs/track_demos/track_results/kaojing.txt'

# 图片路径
frames_dir = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/videos-for-demo/kaojing/'

# 输出视频路径
output_video_dir = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/videos-for-demo/kaojing/results/'

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
output_video_path = os.path.join(output_video_dir, 'tracking_result.mp4')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# 加载中文字体
# 尝试使用 Noto Sans CJK 字体
font_path = '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc'
try:
    font = ImageFont.truetype(font_path, 25)
except IOError:
    # 如果字体文件不存在，输出错误信息并退出
    print("Error: The specified Chinese font file does not exist.")
    print("Please install the Noto CJK fonts or update the font path in the code.")
    exit()

# 设置帧范围
start_frame = 70  # 从第1帧开始
end_frame = 140  # 到最后一帧结束

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
        drone_count = len(tracks)  # 当前帧检测到的无人机数量

        # 将OpenCV图像转换为Pillow图像
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # 在左上角显示无人机数量
        text = f"检测到 {drone_count} 个无人机"
        if drone_count > 0:
            text_color = (255, 0, 0)  # 红色
        else:
            text_color = (0, 255, 0)  # 绿色

        draw.text((10, 10), text, fill=text_color, font=font)

        # 绘制每个跟踪目标的边界框和标签
        for track in tracks:
            x, y, w, h = track['x'], track['y'], track['w'], track['h']
            track_id = track['track_id']
            confidence = track['confidence']
            class_id = track['class_id']

            # 将坐标转换为整数
            x, y, w, h = int(x), int(y), int(w), int(h)

            # 绘制边界框
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)

            # 显示“靠近”提示在边界框正上方
            approach_text = "靠近"
            approach_text_width = draw.textlength(approach_text, font=font)
            text_x = x + (w - approach_text_width) // 2  # 居中对齐
            text_y = y - 35  # 放在边界框上方
            draw.text((text_x, text_y), approach_text, fill=(255, 0, 0), font=font)

        # 将Pillow图像转换回OpenCV图像
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 将处理后的帧写入视频
    out.write(frame)

# 释放资源
out.release()

print(f"Tracking visualization video has been saved to {output_video_path}")