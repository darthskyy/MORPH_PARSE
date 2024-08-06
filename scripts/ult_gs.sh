#!/bin/bash
#PBS -N GSU_ZU_X2
#PBS -q gpu_1
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -P CSCI1674
#PBS -l walltime=12:00:00
#PBS -m abe
#PBS -M mwrsim003@myuct.ac.za
ulimit -s unlimited

echo "Starting script at $(date)"
cd /mnt/lustre/users/smawere/MORPH_PARSE
module purge
module load chpc/python/anaconda/3-2021.05
module load chpc/cuda/11.5.1/PCIe/11.5.1
module load gcc/9.2.0

echo "Installing requirements"
pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install -r requirements.txt

language=ZU
output_dir=plm/models/grid/${language}_$(date +%Y%m%d%H%M%S)
data_dir=data
model_dir=xlm-roberta-large
python3 plm/grid_search.py --language $language --output_dir $output_dir \
    --disable_tqdm --data_dir $data_dir --model_dir $model_dir \
    --logging_steps 5000 --eval_steps 5000 --save_steps 5000 \
    --log_file results/${language}_final.csv --overwrite_output_dir \
    --epoch_list 5  --learning_rate_list 1e-5 3e-5 5e-5  --batch_size_list 8 16 32

rm output_dir -rf