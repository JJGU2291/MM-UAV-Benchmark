CUDA_VISIBLE_DEVICES=0 \
python3 tools/masort-multi2-event.py "/mnt/sda/Disk_D/MMMUAV/" \
--experiment-name 'MMA-SORT-Def' \
--benchmark "MMMUAV" \
--eval "test" \
--p 2 \
--track_high_thresh 0.1 \
--track_low_thresh 0.1 \
--new_track_thresh 0.7 \
--exp_file 'yolox/exps/example/custom/yolox_s_2_def-tuning-fusion-head.py' \
--ckpt 'YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/best_ckpt.pth.tar' \
--conf 0.3 \
--appearance_thresh 0.20 \
--proximity_thresh 0.9 \
--use_recent True \
--with-reid \
--fuse \
--use_iou_3 \
--use_event_3 \
--use_app_3