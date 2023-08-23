#!/bin/bash

#SBATCH -c4 --mem=16g
#SBATCH --gpus 1
#SBATCH -p cs -q cspg

####source /usr2/share/gpu.sbatch

torchrun train.py --dataset_dir ~/dissertation-msc/datasets/nyuv2