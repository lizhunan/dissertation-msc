#!/bin/bash

#SBATCH -c4 --mem=32g
#SBATCH --gpus 2
#SBATCH -p cs -q cspg

####source /usr2/share/gpu.sbatch

# torchrun train.py --dataset_dir ~/dissertation-msc/datasets/nyuv2

torchrun train.py --dataset_dir ~/dissertation-msc/datasets/sunrgbd --dataset sunrgbd