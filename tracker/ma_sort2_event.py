import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import torch
from PIL import Image

from tracker import matching
from tracker.Extractor import Extractor
from tracker.Extractor_Event import EventExtractor
from tracker.gmc import GMC
from tracker.basetrack2 import BaseTrack2, TrackState2
from tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface

Vis = False

class STrack2(BaseTrack2):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_event=None, feat_history=50, use_recent=False):

        self.use_recent = use_recent

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None

        self.smooth_feat_event = None  ###
        self.curr_feat_event = None  ###

        if feat is not None:
            self.update_features(feat)

        if feat_event is not None:
            self.update_features_event(feat_event)

        self.features = deque([], maxlen=feat_history)
        self.features_event = deque([], maxlen=feat_history)  ###

        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat

        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_features_event(self, feat):
        """Update the feature vector and apply exponential moving average smoothing."""
        feat /= np.linalg.norm(feat)
        self.curr_feat_event = feat
        if self.smooth_feat_event is None:
            self.smooth_feat_event = feat
        else:
            if self.use_recent:
                # print("use recent")
                self.smooth_feat_event = feat
            else:
                self.smooth_feat_event =  self.alpha * self.smooth_feat_event + (1 - self.alpha) * feat
        self.features_event.append(feat)
        self.smooth_feat_event /= np.linalg.norm(self.smooth_feat_event)


    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState2.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState2.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack2.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState2.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState2.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState2.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class MASORT2(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack2]
        self.lost_stracks = []  # type: list[STrack2]
        self.removed_stracks = []  # type: list[STrack2]
        BaseTrack2.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh
        # self.new_track_thresh = 0.6

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.event_thresh = args.event_thresh

        self.use_recent = args.use_recent

        self.use_event_1 = args.use_event_1
        self.use_event_2 = args.use_event_2
        self.use_event_3 = args.use_event_3

        self.use_iou_3 = args.use_iou_3

        self.use_app_3 = args.use_app_3

        if args.with_reid:
            if args.benchmark=='MMMUAV':
                path = "pretrained/multi_uav-ir.t7"
                self.encoder = Extractor(model_path=path, device=self.args.device, num_classes=2913)  # 2913
                self.encoder_event = EventExtractor()
            else:
                raise NotImplementedError
                # self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        # self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def update(self, output_results, img, img_event=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            # lowest_inds = scores > self.track_low_thresh #过滤掉比low低的
            # bboxes = bboxes[lowest_inds]
            # scores = scores[lowest_inds]
            # classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh #找到比hign高的
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        if len(dets) > 0:

            # print(img)

            '''Extract embeddings '''
            if self.args.with_reid:
                # target_crops = get_crops(img_rgb=img, dets=dets)
                target_crops, target_crops_event = get_crops2(img_rgb=img, img_event=img_event, dets=dets)
                features_keep = self.encoder(target_crops)
                features_event = self.encoder_event(target_crops_event)

            '''Detections'''
            if self.args.with_reid:
                detections = [STrack2(STrack2.tlbr_to_tlwh(tlbr), s, f, f_e, use_recent=self.use_recent) for
                              (tlbr, s, f, f_e) in zip(dets, scores_keep, features_keep, features_event)]
            else:
                detections = [STrack2(STrack2.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack2]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack2.multi_predict(strack_pool)

        # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack2.multi_gmc(strack_pool, warp)
        # STrack2.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes

        ious_dists = matching.iou_distance(strack_pool, detections)

        ious_dists_mask = (ious_dists > self.proximity_thresh)

        ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections)

            #### motion
            if self.use_event_1:
                # print("using event 1")
                emb_dists_motion = matching.motion_distance(strack_pool, detections) / 4.0
                emb_dists = emb_dists * 0.5 + emb_dists_motion * 0.5

            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0

            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)


        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState2.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with unmatched detection boxes'''

        detections = [detections[i] for i in u_detection]  # 未匹配的检测框
        r_tracked_stracks = [strack_pool[i] for i in u_track]  # 提取未匹配的轨迹


        emb_dists = matching.embedding_distance(r_tracked_stracks, detections)


        #### motion
        if self.use_event_2:
            # print("using event 2")
            emb_dists_motion = matching.motion_distance(r_tracked_stracks, detections) / 4.0
            emb_dists = emb_dists * 0.5 + emb_dists_motion * 0.5

        emb_dists[emb_dists > self.appearance_thresh] = 1.0

        matches, u_track, u_detection = matching.linear_assignment(emb_dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState2.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState2.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''

        detections = [detections[i] for i in u_detection]

        emb_dists = matching.embedding_distance(unconfirmed, detections)

        # raw_emb_dists = emb_dists.copy()
        emb_dists[emb_dists > self.appearance_thresh] = 1.0

        #### motion
        if self.use_event_3:
            emb_dists_motion = matching.motion_distance(unconfirmed, detections)
            emb_dists_motion[emb_dists_motion > self.event_thresh] = 1.0

            if self.use_app_3:
                emb_dists = np.maximum(emb_dists, emb_dists_motion)
            else:
                emb_dists = emb_dists_motion

        if self.use_iou_3:
            ious_dists = matching.iou_distance(unconfirmed, detections)
            ious_dists_mask = (ious_dists > self.proximity_thresh)
            emb_dists[ious_dists_mask] = 1.0

        matches, u_unconfirmed, u_detection = matching.linear_assignment(emb_dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState2.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def get_crops(img_rgb, dets):
    #dets: [x1,y1,x2,y2] numpy.ndarray
    # print('raw dets=',dets)
    # dets_xyxy=xywh2xyxy(dets)
    # print('xyxy dets=', dets)

    return [crop_targets(xyxy[:4], img_rgb) for xyxy in dets]

def get_crops2(img_rgb, img_event, dets):
    crops_a = []  # 用于存储第一个结果
    crops_b = []  # 用于存储第二个结果
    for xyxy in dets:
        crop_result = crop_targets_2modal(xyxy=xyxy[:4], im=img_rgb, event_im=img_event, pad_event=40)
        # 假设 crop_targets 返回两个值，分别添加到对应的列表
        crops_a.append(crop_result[0])
        crops_b.append(crop_result[1])
    return crops_a, crops_b  # 返回两个列表

def crop_targets(xyxy, im,  gain=1.02, pad=10, square=False, BGR=False):
    b_xyxy=xyxy

    b = xyxy2xywh(xyxy.reshape(-1, 4))  # boxes

    # print('b_xyxy=',b_xyxy)
    # print('b_xywh=',b)

    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square

    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    # print('b_expend=',b)

    xyxy = xywh2xyxy(b).astype(np.int64)
    xyxy = clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    # Check if the cropped image is empty
    if crop.size == 0:
        print(b_xyxy)
        print(b)
        print(xyxy)
        raise ValueError("Cropped image is empty")
    return crop


def crop_targets_2modal(
        xyxy,
        im,
        event_im,
        gain_im=1.02,
        pad_im=10,
        gain_event=1.02,
        pad_event=40,
        square=False,
        BGR=False,
        save=False
):
    """
    对IR图像和事件图像分别按照独立的增益和填充参数裁剪目标区域。

    Args:
        xyxy (torch.Tensor | list): 边界框，格式为xyxy。
        im (numpy.ndarray): RGB输入图像。
        event_im (numpy.ndarray): 事件输入图像。
        file (Path): 裁剪后的RGB图像保存路径。
        gain_im (float): RGB图像的边界框增益系数。
        pad_im (int): RGB图像的边界框填充像素。
        gain_event (float): 事件图像的边界框增益系数。
        pad_event (int): 事件图像的边界框填充像素。
        square (bool): 是否强制输出方形裁剪区域。
        BGR (bool): 是否以BGR格式保存RGB图像。
        save (bool): 是否保存RGB裁剪图像。

    Returns:
        (tuple[numpy.ndarray, numpy.ndarray]): RGB裁剪图像和事件裁剪图像。
    """
    # 确保输入为numpy数组
    xyxy = np.asarray(xyxy)

    # 处理RGB图像裁剪
    b_im = xyxy2xywh(xyxy.reshape(-1, 4))
    if square:
        b_im[:, 2:] = b_im[:, 2:].max(axis=1, keepdims=True)  # 转换为方形
    b_im[:, 2:] = b_im[:, 2:] * gain_im + pad_im  # 应用独立增益和填充
    xyxy_im = xywh2xyxy(b_im).astype(np.int64)
    xyxy_im = clip_boxes(xyxy_im, im.shape)
    crop_im = im[
              int(xyxy_im[0, 1]): int(xyxy_im[0, 3]),
              int(xyxy_im[0, 0]): int(xyxy_im[0, 2]),
              :: (1 if BGR else -1),
              ]

    # 处理事件图像裁剪
    h_im, w_im = im.shape[:2]
    h_event, w_event = event_im.shape[:2]
    scale_x, scale_y = w_event / w_im, h_event / h_im  # 计算比例因子

    # 将原始边界框缩放到事件图像坐标系
    xyxy_event_orig = xyxy * np.array([scale_x, scale_y, scale_x, scale_y])

    # 独立处理事件图像的增益和填充
    b_event = xyxy2xywh(xyxy_event_orig.reshape(-1, 4))
    if square:
        b_event[:, 2:] = b_event[:, 2:].max(axis=1, keepdims=True)
    b_event[:, 2:] = b_event[:, 2:] * gain_event + pad_event  # 事件专用参数
    xyxy_event = xywh2xyxy(b_event).astype(np.int64)
    xyxy_event = clip_boxes(xyxy_event, event_im.shape)
    crop_event = event_im[
                 int(xyxy_event[0, 1]): int(xyxy_event[0, 3]),
                 int(xyxy_event[0, 0]): int(xyxy_event[0, 2]),
                 ]
    if save:
        # 定义保存文件夹路径
        rgb_folder = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/YOLOX_outputs/crops-rgb'
        event_folder = '/mnt/sda/Disk_B/jinjie/Tracking/MOTs/BoT-SORT-2stream/YOLOX_outputs/crops-event'

        # 确保文件夹存在
        os.makedirs(rgb_folder, exist_ok=True)
        os.makedirs(event_folder, exist_ok=True)

        # 生成基于当前时间的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒级时间戳

        # 构建完整文件路径
        rgb_path = os.path.join(rgb_folder, f"rgb_{timestamp}.jpg")
        event_path = os.path.join(event_folder, f"event_{timestamp}.jpg")

        # 保存图像
        Image.fromarray(crop_im[..., ::-1]).save(rgb_path, quality=95, subsampling=0)
        Image.fromarray(crop_event[..., ::-1]).save(event_path, quality=95, subsampling=0)

    # print(crop_im.shape, crop_event.shape)

    return crop_im, crop_event

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] >= 4, f"input shape last dimension expected >=4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:4] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:4] = xy + wh  # bottom right xy
    if x.shape[-1] > 4:  # 如果有额外值（如 id）
        y[..., 4:] = x[..., 4:]  # 直接复制额外值
    return y

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): The bounding boxes to clip.
        shape (tuple): The shape of the image.

    Returns:
        (torch.Tensor | numpy.ndarray): The clipped boxes.
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

def empty_like(x):
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )