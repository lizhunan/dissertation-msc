#!/bin/bash

#SBATCH -c4 --mem=32g
#SBATCH --gpus 2
#SBATCH -p cs -q cspg

####source /usr2/share/gpu.sbatch

torchrun torch2onnx.py --ckpt_path ~/dissertation-msc/trained_models/ckp_nyuv2_bst_0.pth