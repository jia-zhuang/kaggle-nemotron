#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RESUME_FROM_CHECKPOINT=auto
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# accelerate launch --config_file accelerate_zero2_sp.yaml train_sp.py  > train.log 2>&1
# accelerate launch --config_file accelerate_zero2.yaml train.py  > train.log 2>&1
accelerate launch --config_file accelerate_zero2.yaml train.py 


# python3.11 train.py

