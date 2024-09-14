echo "Starting script at $(date)"
cd /mnt/lustre/users/smawere/MORPH_PARSE
echo "Loading modules"
module purge
module load chpc/python/anaconda/3-2021.05
module load chpc/cuda/11.5.1/PCIe/11.5.1
module load gcc/9.2.0

echo "Installing requirements"
pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html -q
pip3 install -r requirements.txt -q