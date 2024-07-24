#!/bin/bash
#PBS -N auto_ZU_n
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
pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html -q
pip3 install -r requirements.txt -q

python3 plm/demo_script.py