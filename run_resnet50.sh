#!/bin/bash
python run_infer_mode.py \
 --backend PYTORCH \
 --model resnet50 \
 --scenario Offline \
 --data_dir /mnt/Datasets/ImageNet/inference \
 --images_processed_path /mnt/Datasets/npy/ImageNet_RN50 \
 --batch_size 1024 \
 --duration 60 \
 --sample_per_query 100000 \
 --querys 1 \
 --precision fp16 \
 --accuracy True \
