#!/bin/bash

#SBATCH -c4 --mem=32g
#SBATCH --gpus 2
#SBATCH -p cs -q cspg

####source /usr2/share/gpu.sbatch

# torchrun inference.py --ckpt_path ~/dissertation-msc/trained_models/ckp_ECA_convnext_t_nyuv2_50.pth \
#                       --dataset_dir ~/dissertation-msc/datasets/nyuv2 \
#                       --dataset nyuv2 \
#                       --num_samples 3 \
#                       --context_module 1 \
#                       --rgb_encoder convnext_b --depth_encoder convnext_b

torchrun inference.py --ckpt_path ~/dissertation-msc/trained_models/ckp_sunrgbd_bst_5.pth \
                      --dataset_dir ~/dissertation-msc/datasets/sunrgbd \
                      --dataset sunrgbd \
                      --num_samples 3
                      --context_module 1 \
                      --rgb_encoder convnext_b --depth_encoder convnext_b