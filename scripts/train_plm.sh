#!/bin/bash
#PBS -N train_plm
#PBS -q gpu_1
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -P CSCI1674
#PBS -l walltime=6:00:00
#PBS -m abe
#PBS -M mwrsim003@myuct.ac.za
ulimit -s unlimited

source /mnt/lustre/users/smawere/MORPH_PARSE/myenv/bin/activate
cd /mnt/lustre/users/smawere/MORPH_PARSE

data=data
checkpoint=xlm-roberta-base
output_dir=plm/models/$checkpoint
epochs=10
batch_size=16
evaluation_strategy=epoch
lang=NR
learning_rate=2e-5
validation_split=0.1
save_steps=10
save_total_limit=2

python3 plm/train_plm.py \
    --data $data \
    --checkpoint $checkpoint \
    --output_dir $output_dir \
    --epochs $epochs \
    --batch_size $batch_size \
    --evaluation_strategy $evaluation_strategy \
    --lang $lang \
    --learning_rate $learning_rate \
    --validation_split $validation_split \
    --save_steps $save_steps \
    --save_total_limit $save_total_limit \
    > $output_dir/train.txt

echo "Evaluating model" > $output_dir/results.txt

python3 plm/train.py \
    --test $data/TEST/$lang_TEST.tsv
    --model $checkpoint \
    --tokenizeer $checkpoint \
    --lang $lang \
    --metric all \
    > $output_dir/results.txt
