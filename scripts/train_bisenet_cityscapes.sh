#!/usr/bin/bash
# use the bash shell

#SBATCH -p GPU-shared # partition (queue)
#SBATCH --gres=gpu:v100-16:2 # partition (queue)
#SBATCH -c 8 # number of cores
#SBATCH -t 12:00:00 # Upper bound on time
cd /jet/home/russelld/dev/BiSeNet
source ~/anaconda3/etc/profile.d/conda.sh

conda activate pytorch181
export CUDA_VISIBLE_DEVICES=0,1
cfg_file=configs/bisenetv1_city.py
NGPUS=2
python -c 'import torch; print(torch.cuda.is_available())'
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file
cd -
