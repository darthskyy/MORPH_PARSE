#!/bin/bash
#PBS -N XH_XLMRL
#PBS -q gpu_1
#PBS -l select=1:ncpus=2:ngpus=1
#PBS -P CSCI1674
#PBS -l walltime=12:00:00
#PBS -m abe
#PBS -M mwrsim003@myuct.ac.za
ulimit -s unlimited

echo "Starting script at $(date)"
cd /mnt/lustre/users/smawere/MORPH_PARSE
module purge
module load chpc/python/anaconda/3-2021.05
module load chpc/cuda/11.2/PCIe/11.2

echo "Installing requirements"
pip3 install -r requirements.txt

echo "Training PLM"
data=data
checkpoint=xlm-roberta-large
lang=XH
output_dir=plm/models/${checkpoint}_${lang}
epochs=10
batch_size=16
evaluation_strategy=steps
learning_rate=2e-5
validation_split=0.1
save_steps=2500
save_total_limit=2
load_best_model_at_end=True
metric_for_best_model=loss
greater_is_better=False
resume_from_checkpoint=True

python3 plm/train.py \
    --data $data \
    --checkpoint $checkpoint \
    --output $output_dir \
    --epochs $epochs \
    --batch_size $batch_size \
    --evaluation_strategy $evaluation_strategy \
    --lang $lang \
    --learning_rate $learning_rate \
    --validation_split $validation_split \
    --save_steps $save_steps \
    --save_total_limit $save_total_limit \
    --load_best_model_at_end \
    --metric_for_best_model $metric_for_best_model \
    --greater_is_better $greater_is_better \
    --resume_from_checkpoint
