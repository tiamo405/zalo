#!/bin/sh
python predict.py --replicate 11 --name_model mobilenet_v2 --public public2 --dataset_dir dataset \
--test_dir dataset/test --checkpoint_dir checkpoints --save_dir results