#!/usr/bin/bash
# use the bash shell

#SBATCH -p GPU-shared # partition (queue)
#SBATCH --gres=gpu:v100-16:2 # partition (queue)
#SBATCH -c 8 # number of cores
#SBATCH -t 1:00:00 # number of cores
conda init bash
conda activate bisenet2
cd /jet/home/russelld/dev/BiSeNet
python tools/demo_video.py --config configs/bisenetv2_city.py --weight-path models/model_final_v2_city.pth --input ./video.mp4 --output res.mp4
