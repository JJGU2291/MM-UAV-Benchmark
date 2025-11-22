CUDA_VISIBLE_DEVICES=1 \
python -m yolox.train \
-f yolox/exps/example/custom/yolox_s_2_stn-tuning-fusion-head.py \
--experiment-name yolox_s_2_stream_stn-tuning-fusion-head \
-d 1 \
-b 16 \
--fp16