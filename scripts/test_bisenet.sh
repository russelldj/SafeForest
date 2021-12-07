#!/usr/bin/bash
# use the bash shell

#SBATCH -p GPU-shared # partition (queue)
#SBATCH --gres=gpu:v100-16:1 # partition (queue)
#SBATCH -t 10:00 # time limit
cd /jet/home/russelld/dev/BiSeNet
source ~/anaconda3/etc/profile.d/conda.sh

conda activate pytorch181
export CUDA_VISIBLE_DEVICES=0,1
cfg_file=configs/bisenetv1_synthetic.py
echo "cuda is available: "
python -c 'import torch; print(torch.cuda.is_available())'
echo $CONDA_PREFIX

python tools/demo_video.py --config configs/bisenetv1_synthetic.py \
    --weight-path /ocean/projects/tra190016p/russelld/SafeForestData/models/synthetic/bisenetv1-synthetic-train-2021-11-23-11-54-23/model_final.pth \
    --input /ocean/projects/tra190016p/russelld/SafeForestData/datasets/first_synthetic_delivery_derived/test.mp4 \
    --output /ocean/projects/tra190016p/russelld/SafeForestData/models/synthetic/bisenetv1-synthetic-train-2021-11-23-11-54-23/test_pred.mp4
cd -
