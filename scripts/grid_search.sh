#!/bin/bash
#PBS -N ZU_grid_X
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

language=ZU
output_dir=plm/models/grid/${language}_search
data_dir=data
model_dir=xlm-roberta-large
python3 plm/utils.py --language $language --output_dir $output_dir \
    --logging_steps 200 --disable_tqdm --data_dir $data_dir --model_dir $model_dir \
    --log_file ${language}_search_X.txt --overwrite_output_dir
