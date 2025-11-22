import os
import numpy as np
import json
import cv2

# Use the same script for MOT16

DATA_PATH = "/mnt/sda/Disk_D/MMMUAV"
OUT_PATH = os.path.join(DATA_PATH, "annotations")
SPLITS = [
    "train",
    "val"
]
HALF_VIDEO = False
CREATE_SPLITTED_ANN = False
CREATE_SPLITTED_DET = False

if __name__ == "__main__":

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        if split == "val":
            data_path = os.path.join(DATA_PATH, "test")
        else:
            data_path = os.path.join(DATA_PATH, "train")

        # 创建两个模态的输出路径
        out_path_rgb = os.path.join(OUT_PATH, "{}-rgb.json".format(split))
        out_path_ir = os.path.join(OUT_PATH, "{}-ir.json".format(split))

        # 初始化两个模态的输出字典
        out_rgb = {
            "images": [],
            "annotations": [],
            "videos": [],
            "categories": [{"id": 1, "name": "drone"}],
        }
        out_ir = {
            "images": [],
            "annotations": [],
            "videos": [],
            "categories": [{"id": 1, "name": "drone"}],
        }

        seqs = os.listdir(data_path)
        image_cnt_rgb = 0
        image_cnt_ir = 0
        ann_cnt_rgb = 0
        ann_cnt_ir = 0
        video_cnt = 0
        tid_curr_rgb = 0
        tid_last_rgb = -1
        tid_curr_ir = 0
        tid_last_ir = -1

        for seq in sorted(seqs):
            if ".DS_Store" in seq:
                continue
            video_cnt += 1  # video sequence number.
            # 添加视频信息到两个模态
            out_rgb["videos"].append({"id": video_cnt, "file_name": seq})
            out_ir["videos"].append({"id": video_cnt, "file_name": seq})

            seq_path = os.path.join(data_path, seq)
            img_path_rgb = os.path.join(seq_path, "rgb_frame")
            img_path_ir = os.path.join(seq_path, "ir_frame")
            ann_path_rgb = os.path.join(seq_path, "gt_rgb/gt.txt")
            ann_path_ir = os.path.join(seq_path, "gt_ir/gt.txt")

            # 检查两个模态的标注文件是否存在
            if not (os.path.exists(ann_path_rgb) and os.path.exists(ann_path_ir)):
                continue

            # 读取两个模态的标注数据
            anns_rgb = np.loadtxt(ann_path_rgb, dtype=np.float32, delimiter=",") if os.path.exists(ann_path_rgb) else np.zeros((0, 8))

            anns_ir = np.loadtxt(ann_path_ir, dtype=np.float32, delimiter=",") if os.path.exists(ann_path_ir) else np.zeros((0, 8))

            # 获取两个模态的有效帧ID
            valid_frame_ids_rgb = np.unique(anns_rgb[:, 0]).astype(int) if anns_rgb.size > 0 else np.array([])
            valid_frame_ids_ir = np.unique(anns_ir[:, 0]).astype(int) if anns_ir.size > 0 else np.array([])

            # 找出两个模态共有的帧ID
            common_frame_ids = sorted(set(valid_frame_ids_rgb) & set(valid_frame_ids_ir))

            # 获取图像数量（使用RGB模态的图像数量）
            images = os.listdir(img_path_rgb)
            num_images = len([image for image in images if "jpg" in image])

            if HALF_VIDEO and ("half" in split):
                image_range = [0, num_images // 2] if "train" in split else [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            # 创建映射：帧ID -> 图像信息（两个模态分别存储）
            frame_id_to_image_info_rgb = {}
            frame_id_to_image_info_ir = {}

            # 只处理两个模态共有的帧
            for frame_id in common_frame_ids:
                # 检查帧ID是否在范围内
                frame_index = frame_id - 1
                if frame_index < image_range[0] or frame_index > image_range[1]:
                    continue

                # ===== 处理RGB模态 =====
                img_file_rgb = os.path.join(img_path_rgb, f"{frame_id:04d}.jpg")
                if not os.path.exists(img_file_rgb):
                    continue
                img_rgb = cv2.imread(img_file_rgb)
                height_rgb, width_rgb = img_rgb.shape[:2]

                # 创建RGB图像信息
                image_info_rgb = {
                    "file_name": f"{split}/{seq}/rgb_frame/{frame_id:04d}.jpg" if split == "train" else f"test/{seq}/rgb_frame/{frame_id:04d}.jpg",
                    "id": image_cnt_rgb + 1,
                    "frame_id": int(frame_id - image_range[0]),
                    "prev_image_id": -1,
                    "next_image_id": -1,
                    "video_id": video_cnt,
                    "height": height_rgb,
                    "width": width_rgb,
                }
                out_rgb["images"].append(image_info_rgb)
                frame_id_to_image_info_rgb[frame_id] = image_info_rgb
                image_cnt_rgb += 1

                # ===== 处理IR模态 =====
                img_file_ir = os.path.join(img_path_ir, f"{frame_id:04d}.jpg")
                if not os.path.exists(img_file_ir):
                    continue
                img_ir = cv2.imread(img_file_ir)
                height_ir, width_ir = img_ir.shape[:2]

                # 创建IR图像信息
                image_info_ir = {
                    "file_name": f"{split}/{seq}/ir_frame/{frame_id:04d}.jpg" if split == "train" else f"test/{seq}/ir_frame/{frame_id:04d}.jpg",
                    "id": image_cnt_ir + 1,
                    "frame_id": int(frame_id - image_range[0]),
                    "prev_image_id": -1,
                    "next_image_id": -1,
                    "video_id": video_cnt,
                    "height": height_ir,
                    "width": width_ir,
                }
                out_ir["images"].append(image_info_ir)
                frame_id_to_image_info_ir[frame_id] = image_info_ir
                image_cnt_ir += 1

            # 处理标注（如果不是测试集）
            if split != "test":
                # ===== 处理RGB模态的标注 =====
                for i in range(anns_rgb.shape[0]):
                    frame_id = int(anns_rgb[i][0])

                    # 只处理共有的帧
                    if frame_id not in frame_id_to_image_info_rgb:
                        continue

                    track_id = int(anns_rgb[i][1])
                    ann_cnt_rgb += 1

                    # 类别处理逻辑
                    if not ("15" in DATA_PATH):
                        if not (int(anns_rgb[i][6]) == 1):  # whether ignore.
                            continue
                        if int(anns_rgb[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                            continue
                        if int(anns_rgb[i][7]) in [2, 7, 8, 12]:  # Ignored person
                            continue
                        else:
                            category_id = 1  # pedestrian(non-static)
                            if not track_id == tid_last_rgb:
                                tid_curr_rgb += 1
                                tid_last_rgb = track_id
                    else:
                        category_id = 1

                    # 获取对应的图像信息
                    image_info = frame_id_to_image_info_rgb[frame_id]
                    bbox = anns_rgb[i][2:6].tolist()

                    ann = {
                        "id": ann_cnt_rgb,
                        "category_id": category_id,
                        "image_id": image_info["id"],
                        "track_id": tid_curr_rgb,
                        "bbox": bbox,
                        "conf": float(anns_rgb[i][6]),
                        "iscrowd": 0,
                        "area": float(bbox[2] * bbox[3]),
                    }
                    out_rgb["annotations"].append(ann)

                # ===== 处理IR模态的标注 =====
                for i in range(anns_ir.shape[0]):
                    frame_id = int(anns_ir[i][0])

                    # 只处理共有的帧
                    if frame_id not in frame_id_to_image_info_ir:
                        continue

                    track_id = int(anns_ir[i][1])
                    ann_cnt_ir += 1

                    # 类别处理逻辑
                    if not ("15" in DATA_PATH):
                        if not (int(anns_ir[i][6]) == 1):  # whether ignore.
                            continue
                        if int(anns_ir[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                            continue
                        if int(anns_ir[i][7]) in [2, 7, 8, 12]:  # Ignored person
                            continue
                        else:
                            category_id = 1  # pedestrian(non-static)
                            if not track_id == tid_last_ir:
                                tid_curr_ir += 1
                                tid_last_ir = track_id
                    else:
                        category_id = 1

                    # 获取对应的图像信息
                    image_info = frame_id_to_image_info_ir[frame_id]
                    bbox = anns_ir[i][2:6].tolist()

                    ann = {
                        "id": ann_cnt_ir,
                        "category_id": category_id,
                        "image_id": image_info["id"],
                        "track_id": tid_curr_ir,
                        "bbox": bbox,
                        "conf": float(anns_ir[i][6]),
                        "iscrowd": 0,
                        "area": float(bbox[2] * bbox[3]),
                    }
                    out_ir["annotations"].append(ann)

            print(f"{seq}: RGB-{len(frame_id_to_image_info_rgb)} frames, IR-{len(frame_id_to_image_info_ir)} frames")

        # 保存两个模态的JSON文件
        json.dump(out_rgb, open(out_path_rgb, "w"))
        json.dump(out_ir, open(out_path_ir, "w"))

        print("RGB: loaded {} for {} images and {} samples".format(split, len(out_rgb["images"]),
                                                                   len(out_rgb["annotations"])))
        print("IR: loaded {} for {} images and {} samples".format(split, len(out_ir["images"]),
                                                                  len(out_ir["annotations"])))
        print("=" * 50)