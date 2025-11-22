import argparse
import multiprocessing
import os
import sys
import os.path as osp
import time

import cv2
import numpy as np
import torch

sys.path.append('.')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

from tracker.tracking_utils.timer import Timer
from tracker.ma_sort_event import MASORT
from tracker.ma_sort2_event import MASORT2

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("MMA-SORT Tracks For Evaluation!")

    parser.add_argument("path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")

    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MMMUAV', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

    parser.add_argument("--p", default=4, type=int, help="number for processing")

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=240, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=0, help='filter out tiny boxes')

    # CMC
    parser.add_argument("--cmc-method", default="none", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    parser.add_argument('--event_thresh', type=float, default=0.4, help='threshold for rejecting low event similarity reid matches')

    parser.add_argument('--use_recent', type=bool, default=False, help='use_recent_embedding')

    parser.add_argument("--use_event_1", default=False, action="store_true", help="use event in first stage")
    parser.add_argument("--use_event_2", default=False, action="store_true", help="use event in second stage")
    parser.add_argument("--use_event_3", default=False, action="store_true", help="use event in third stage")

    parser.add_argument("--use_iou_3", default=False, action="store_true", help="use iou in third stage")

    parser.add_argument("--use_app_3", default=False, action="store_true", help="use app in third stage")

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img_rgb, img_ir, timer):
        img_info_rgb = {"id": 0}
        img_info_ir = {"id": 0}
        if isinstance(img_rgb, str):
            img_info_rgb["file_name"] = osp.basename(img_rgb)
            img_rgb = cv2.imread(img_rgb)
        else:
            img_info_rgb["file_name"] = None

        if isinstance(img_ir, str):
            img_info_ir["file_name"] = osp.basename(img_ir)
            img_ir = cv2.imread(img_ir)
        else:
            img_info_ir["file_name"] = None

        if img_rgb is None:
            raise ValueError("Empty image: ", img_info_rgb["file_name"])
        if img_ir is None:
            raise ValueError("Empty image: ", img_info_ir["file_name"])

        height_rgb, width_rgb = img_rgb.shape[:2]
        img_info_rgb["height"] = height_rgb
        img_info_rgb["width"] = width_rgb
        img_info_rgb["raw_img"] = img_rgb

        height_ir, width_ir = img_ir.shape[:2]
        img_info_ir["height"] = height_ir
        img_info_ir["width"] = width_ir
        img_info_ir["raw_img"] = img_ir

        # img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_rgb, ratio_rgb = preproc(img_rgb, self.test_size)
        img_ir, ratio_ir = preproc(img_ir, self.test_size)

        img_info_rgb["ratio"] = ratio_rgb
        img_info_ir["ratio"] = ratio_ir

        img_rgb = torch.from_numpy(img_rgb).unsqueeze(0).float().to(self.device)
        img_ir = torch.from_numpy(img_ir).unsqueeze(0).float().to(self.device)

        if self.fp16:
            img_rgb = img_rgb.half()  # to FP16
            img_ir = img_ir.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs_rgb, outputs_ir = self.model(img_rgb,img_ir)

            outputs_rgb = postprocess(outputs_rgb, self.num_classes, self.confthre, self.nmsthre)
            outputs_ir = postprocess(outputs_ir, self.num_classes, self.confthre, self.nmsthre)
        return outputs_rgb, outputs_ir, img_info_rgb, img_info_ir


def image_track(predictor, vis_folder_rgb, vis_folder_ir, args, exp):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]

    files.sort()

    if args.ablation:
        files = files[len(files) // 2 + 1:]

    num_frames = len(files)


    # Tracker
    tracker_rgb = MASORT(args, frame_rate=args.fps)

    tracker_ir = MASORT2(args, frame_rate=args.fps)

    results_rgb = []
    results_ir = []

    total_time = 0

    frame_count = 0

    for frame_id, img_path in enumerate(files, 1):

        img_path2 = img_path.replace("rgb", "ir")
        img_path_event = img_path.replace("rgb", "event")
        img_event = cv2.imread(img_path_event)

        # Detect objects
        outputs_rgb, outputs_ir, img_info_rgb, img_info_ir = predictor.inference(img_path, img_path2, timer)

        scale_rgb = min(exp.test_size[0] / float(img_info_rgb['height'], ), exp.test_size[1] / float(img_info_rgb['width']))

        scale_ir = min(exp.test_size[0] / float(img_info_ir['height'], ), exp.test_size[1] / float(img_info_ir['width']))

        if outputs_rgb[0] is not None:
            outputs_rgb = outputs_rgb[0].cpu().numpy()
            detections_rgb = outputs_rgb[:, :7]
            detections_rgb[:, :4] /= scale_rgb

            trackerTimer.tic()
            online_targets_rgb = tracker_rgb.update(detections_rgb, img_info_rgb["raw_img"], img_event)
            trackerTimer.toc()

            online_tlwhs_rgb = []
            online_ids_rgb = []
            online_scores_rgb = []

            for t in online_targets_rgb:
                tlwh = t.tlwh
                tid = t.track_id
                # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                vertical = False
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs_rgb.append(tlwh)
                    online_ids_rgb.append(tid)
                    online_scores_rgb.append(t.score)

                    # save results
                    if frame_id % 20 ==1 :
                        results_rgb.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
            timer.toc()
            online_im_rgb = plot_tracking(
                img_info_rgb['raw_img'], online_tlwhs_rgb, online_ids_rgb, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im_rgb = img_info_rgb['raw_img']

        if outputs_ir[0] is not None:

            outputs_ir = outputs_ir[0].cpu().numpy()
            detections_ir = outputs_ir[:, :7]
            detections_ir[:, :4] /= scale_ir

            trackerTimer.tic()
            online_targets_ir = tracker_ir.update(detections_ir, img_info_ir["raw_img"], img_event)
            trackerTimer.toc()

            online_tlwhs_ir = []
            online_ids_ir = []
            online_scores_ir = []

            for t in online_targets_ir:
                tlwh = t.tlwh
                tid = t.track_id
                # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                vertical = False
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs_ir.append(tlwh)
                    online_ids_ir.append(tid)
                    online_scores_ir.append(t.score)

                    # save results
                    if frame_id % 20 ==1 :
                        results_ir.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
            timer.toc()
            online_im_ir = plot_tracking(
                img_info_ir['raw_img'], online_tlwhs_ir, online_ids_ir, frame_id=frame_id, fps=1. / timer.average_time
            )

        else:
            timer.toc()
            online_im_ir = img_info_ir['raw_img']

        if args.save_frames:
            save_folder_rgb = osp.join(vis_folder_rgb, args.name)
            os.makedirs(save_folder_rgb, exist_ok=True)
            cv2.imwrite(osp.join(save_folder_rgb, osp.basename(img_path)), online_im_rgb)

            save_folder_ir = osp.join(vis_folder_ir, args.name)
            os.makedirs(save_folder_ir, exist_ok=True)
            cv2.imwrite(osp.join(save_folder_ir, osp.basename(img_path2)), online_im_ir)

        if frame_id % 20 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 2. / max(1e-5, timer.average_time)))

    res_file_rgb = osp.join(vis_folder_rgb, args.name + ".txt")
    res_file_ir = osp.join(vis_folder_ir, args.name + ".txt")

    with open(res_file_rgb, 'w') as f:
        f.writelines(results_rgb)
    logger.info(f"save results to {res_file_rgb}")

    with open(res_file_ir, 'w') as f:
        f.writelines(results_ir)
    logger.info(f"save results to {res_file_ir}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder_rgb = osp.join(output_dir, "track_results_rgb")
    vis_folder_ir = osp.join(output_dir, "track_results_ir")

    os.makedirs(vis_folder_rgb, exist_ok=True)
    os.makedirs(vis_folder_ir, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if args.ckpt is None:
        ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")

    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info(f"loaded checkpoint from {ckpt_file} done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    predictor = Predictor(model, exp, args.device, args.fp16)

    image_track(predictor, vis_folder_rgb, vis_folder_ir, args, exp)


def process_sequences(chunk, data_path, fp16, device, ablation):
    for seq in chunk:
        args = make_parser().parse_args()
        args.name = seq

        args.ablation = ablation
        args.fps = 30
        args.device = device
        args.fp16 = fp16
        args.batch_size = 1
        args.trt = False

        split = 'test'

        if args.benchmark == 'MMMUAV':
            args.path = data_path + '/' + split + '/' + seq + '/' + 'rgb_frame'
        else:
            print("not support benchmark")
            raise NotImplementedError

        exp = get_exp(args.exp_file, args.name)

        exp.test_conf = 0.5

        main(exp, args)

if __name__ == "__main__":
    args = make_parser().parse_args()

    data_path = args.path
    fp16 = args.fp16
    device = args.device

    ablation = False

    mainTimer = Timer()
    mainTimer.tic()

    seqs = os.listdir(os.path.join(data_path, "test"))
    seqs.sort()

    # 将序列分成4份
    num_processes = args.p
    chunk_size = len(seqs) // num_processes
    chunks = [seqs[i:i+chunk_size] for i in range(0, len(seqs), chunk_size)]

    # 多进程处理
    with multiprocessing.Manager() as manager:
        processes = []
        for chunk in chunks:
            p = multiprocessing.Process(target=process_sequences, args=(chunk, data_path, fp16, device, ablation))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()




