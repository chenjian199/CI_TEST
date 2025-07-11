#!/bin/bash
python run_infer_mode.py \
 --backend PYTORCH \
 --model retinanet \
 --scenario Offline \
 --data_dir /mnt/Datasets/open-images-v6/validation/data/ \
 --images_processed_path /mnt/Datasets/npy/OpenImages \
 --batch_size 16 \
 --duration 60 \
 --precision fp16 \
 --sample_per_query 30002 \
 --querys 1 \
 --accuracy True
