#!/bin/bash

#SBATCH -c4 --mem=32g
#SBATCH --gpus 2
#SBATCH -p cs -q cspg

####source /usr2/share/gpu.sbatch

# torchrun inference.py --ckpt_path ~/dissertation-msc/trained_models/ckp_nyuv2_bst_0.pth \
#                       --dataset_dir ~/dissertation-msc/datasets/nyuv2 \
#                       --dataset nyuv2 \
#                       --num_samples 700

torchrun inference.py --ckpt_path ~/dissertation-msc/trained_models/ckp_sunrgbd_50.pth \
                      --dataset_dir ~/dissertation-msc/datasets/sunrgbd \
                      --dataset sunrgbd \
                      --num_samples 1