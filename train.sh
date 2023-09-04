#!/bin/bash

#SBATCH -c4 --mem=32g
#SBATCH --gpus 2
#SBATCH -p cs -q cspg

####source /usr2/share/gpu.sbatch

torchrun train.py --dataset_dir ~/dissertation-msc/datasets/nyuv2 --fusion_module SE --context_module 1 --rgb_encoder convnext_t --depth_encoder convnext_t

# torchrun train.py --dataset_dir ~/dissertation-msc/datasets/sunrgbd --dataset sunrgbd