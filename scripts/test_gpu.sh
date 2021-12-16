#!/usr/bin/bash
# use the bash shell

#SBATCH -p GPU-shared # partition (queue)
#SBATCH --gres=gpu:v100-16:2 # partition (queue)
#SBATCH -c 8 # number of cores
module spider cuda
mudule spider cudnn

module load cuda
modele load cudnn
nvidia-smi
