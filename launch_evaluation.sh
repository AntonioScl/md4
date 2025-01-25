#!/bin/bash
#SBATCH --job-name=md4_evaluation
#SBATCH --array=0-10
#SBATCH --output=slurm_out/slurm_%A_%a.out
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=94g
#SBATCH --time=12:00:00

module load gcc/13.2.0 tmux/2.9a python/3.11.7 cuda/12.4.1 cudnn/9.2.1.18-12 openmpi/5.0.3-cuda py-mpi4py/4.0.0
source /home/sclocchi/venvs_kuma/md4_venv/bin/activate
export PYTHONPATH="$PYTHONPATH:/scratch/sclocchi/md4"

export JAX_TRACEBACK_FILTERING=off

# Selct the checkpoint_step from a list according to the array index
checkpoint_steps=(16 64 256 1024 4096 8192 16384 32768 65536 131072)
checkpoint_step=${checkpoint_steps[$SLURM_ARRAY_TASK_ID]}
echo "Checkpoint step: $checkpoint_step"

# Run the evaluation script
srun python md4/evaluate_main.py --config=md4/configs/md4/evaluate_openwebtext.py --sharded=false --workdir=/scratch/sclocchi/md4/expt2 --checkpoint_step=$checkpoint_step --num_batches=64
