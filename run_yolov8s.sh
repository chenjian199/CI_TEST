#!/bin/bash
python run_infer_mode.py \
 --backend ONNX \
 --model yolov8s \
 --scenario Offline \
 --data_dir /mnt/Datasets/COCO2017/images \
 --images_processed_path /mnt/Datasets/npy/COCO2017 \
 --batch_size 1 \
 --duration 60 \
 --sample_per_query 100000 \
 --querys 1 \
 --precision fp16 \
 --accuracy True \
