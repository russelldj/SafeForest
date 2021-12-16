#!/usr/bin/bash
# use the bash shell

#SBATCH -p GPU-shared # partition (queue)
#SBATCH --gres=gpu:v100-16:2 # partition (queue)
#SBATCH -c 8 # number of cores
#SBATCH -t 1:00:00 # number of cores
cd /jet/home/russelld/dev/BiSeNet
source ~/anaconda3/etc/profile.d/conda.sh

conda activate pytorch181
cd /jet/home/russelld/dev/BiSeNet
python tools/demo_video.py --config configs/bisenetv2_city.py --weight-path models/model_final_v2_city.pth --input ./video.mp4 --output res.mp4
