CUDA_VISIBLE_DEVICES=3 \
python -m yolox.train \
-f yolox/exps/example/custom/yolox_s_2_pretrain.py \
--experiment-name yolox_s_2_stream \
-d 1 \
-b 1 \
--fp16