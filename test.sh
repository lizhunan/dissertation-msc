#!/bin/bash

#SBATCH -c4 --mem=16g
#SBATCH --gpus 1
#SBATCH -p cs -q cspg

####source /usr2/share/gpu.sbatch

# torchrun download_data.py  --nyuv2-output-path datasets/nyuv2 --download True