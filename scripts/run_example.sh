#!/usr/bin/bash
# use the bash shell
set -x 
# echo each command to standard out before running it
date
# run the Unix 'date' command
echo "Hello world, from Bridges-2!"
# run the Unix 'echo' command
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate bisenet2
#python -c 'import torch; print(torch.cuda.is_available())'
nvidia-smi
