#!/bin/bash
cd ..

#python dev/model_evaluation/evaluate_model.py with \
#    "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_80_finetune/segformer_mit-b5_512x512_10k_sete_80_finetune.py" \
#    "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_80_finetune/latest.pth" \
#    "palette=semfire-ros" "log_preds=True" --comment "finetune 80" \

python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels.py" \
    "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/iter_112000.pth" \
    "images_dir=/home/frc-ag-1/data/SafeForestData/datasets/first_synthetic_delivery_derived/training/rui_semfire_label/img_dir/val" \
    "palette=semfire-ros" "log_preds=True" --comment "Rui's model"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels.py" \
#    "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/iter_112000.pth" \
#    "palette=semfire-ros" "log_preds=True" --comment "Rui's model"
#python dev/model_evaluation/evaluate_model.py with \
#    "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_80_finetune/segformer_mit-b5_512x512_10k_rui_with_sete_80_finetune.py" \
#    "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_80_finetune/latest.pth" \
#    "palette=semfire-ros" "log_preds=True" --comment "finetune 80" \

# Rui's model
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/iter_112000.pth" --comment "Rui's model"

# Finetuning
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_20_finetune/segformer_mit-b5_512x512_10k_rui_with_sete_20_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_20_finetune/latest.pth" --comment "finetune 20"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_40_finetune/segformer_mit-b5_512x512_10k_rui_with_sete_40_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_40_finetune/latest.pth" --comment "finetune 40"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_60_finetune/segformer_mit-b5_512x512_10k_rui_with_sete_60_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_60_finetune/latest.pth" --comment "finetune 60"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_80_finetune/segformer_mit-b5_512x512_10k_rui_with_sete_80_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_rui_with_sete_80_finetune/latest.pth" --comment "finetune 80"

# Trained from scratch
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_05_finetune/segformer_mit-b5_512x512_10k_sete_05_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_05_finetune/latest.pth" --comment "train 05"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_10_finetune/segformer_mit-b5_512x512_10k_sete_10_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_10_finetune/latest.pth" --comment "train 10"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_15_finetune/segformer_mit-b5_512x512_10k_sete_15_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_15_finetune/latest.pth" --comment "train 15"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_20_finetune/segformer_mit-b5_512x512_10k_sete_20_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_20_finetune/latest.pth" --comment "train 20"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_40_finetune/segformer_mit-b5_512x512_10k_sete_40_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_40_finetune/latest.pth" --comment "train 40"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_60_finetune/segformer_mit-b5_512x512_10k_sete_60_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_60_finetune/latest.pth" --comment "train 60"
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_80_finetune/segformer_mit-b5_512x512_10k_sete_80_finetune.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_sete_80_finetune/latest.pth" --comment "train 80"

# Rui's model on synthetic val
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/iter_112000.pth" "images_dir=/home/frc-ag-1/data/SafeForestData/datasets/first_synthetic_delivery_derived/training/rui_semfire_label/img_dir/val" "groundtruth_dir=/home/frc-ag-1/data/SafeForestData/datasets/first_synthetic_delivery_derived/training/rui_semfire_label/ann_dir/val"
# Rui's model on synthetic train
#python dev/model_evaluation/evaluate_model.py with "cfg_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels.py" "model_path=/home/frc-ag-1/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_160k_rui_with_semfire_sete_labels/iter_112000.pth" "images_dir=/home/frc-ag-1/data/SafeForestData/datasets/first_synthetic_delivery_derived/training/rui_semfire_label/img_dir/train" "groundtruth_dir=/home/frc-ag-1/data/SafeForestData/datasets/first_synthetic_delivery_derived/training/rui_semfire_label/ann_dir/train"
cd -