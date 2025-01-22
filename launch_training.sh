#!/bin/bash
#SBATCH --job-name=md4
#SBATCH --output=slurm_out/logs_%j.log
#SBATCH --error=slurm_out/errs_%j.log
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=94g
#SBATCH --time=0:30:00

module load gcc/13.2.0 tmux/2.9a python/3.11.7 cuda/12.4.1 cudnn/9.2.1.18-12 openmpi/5.0.3-cuda py-mpi4py/4.0.0
source /home/sclocchi/venvs_kuma/md4_venv/bin/activate

python md4/main.py --config=md4/configs/md4/custom_openwebtext.py --sharded=false --workdir=./expt