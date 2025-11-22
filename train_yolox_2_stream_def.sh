CUDA_VISIBLE_DEVICES=1 \
python -m yolox.train \
-f yolox/exps/example/custom/yolox_s_2_def-tuning-fusion-head.py \
--experiment-name yolox_s_2_stream_def_tuning_fusion_head \
-d 1 \
-b 1 \
--fp16