import cv2
import os

# 视频路径
video_path = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/videos-for-demo/zaiwu-kaojing.mp4'
# 保存帧图片的目录
output_dir = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/videos-for-demo/zaiwu-kaojing/'

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否打开成功
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 初始化帧计数器
frame_count = 1

# 读取视频的每一帧
while True:
    ret, frame = cap.read()
    # 如果正确读取帧，ret 为 True
    if not ret:
        break

    # 构建保存的文件名
    frame_filename = f"{output_dir}{str(frame_count).zfill(4)}.jpg"

    # 保存帧为图片文件
    cv2.imwrite(frame_filename, frame)

    # 更新帧计数器
    frame_count += 1

# 释放视频捕获对象
cap.release()

print(f"Successfully saved {frame_count - 1} frames to {output_dir}")