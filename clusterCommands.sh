module load miniconda
conda create -yn notebook_env anaconda python=3

srun --pty -p interactive bash
module load miniconda
source activate notebook_env
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
