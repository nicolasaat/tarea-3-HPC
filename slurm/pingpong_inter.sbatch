#!/bin/bash
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="Tarea 3 IIC3533 (e,g): ping-pong entre 2 procesos en nodos distintos para estimar alpha y beta inter-nodo"
#SBATCH --job-name=pp_inter
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3800M
#SBATCH --time=00:08:00
#SBATCH --output=logs/pp_inter_%j.out
#SBATCH --error=logs/pp_inter_%j.err
# Enviar desde la raiz del repo: sbatch slurm/pingpong_inter.sbatch

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results logs

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "Job $SLURM_JOB_ID en nodos: $SLURM_JOB_NODELIST"

mpirun --bind-to none -n 2 python pingpong.py --tag inter --csv results/pingpong.csv
