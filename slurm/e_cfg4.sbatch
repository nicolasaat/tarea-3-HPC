#!/bin/bash
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --job-name=e_cfg4
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3800M
#SBATCH --time=00:14:30
#SBATCH --output=logs/e_cfg4_%j.out
#SBATCH --error=logs/e_cfg4_%j.err
# Inciso (e), configuracion 4: grid 2x4, T=1, 2 nodos x 4 procesos. Enviar desde la raiz del repo:
#   sbatch slurm/e_cfg4.sbatch

BS="512 2048"   # b_a y b_c (b_c asumido 2048: confirmar con results_c.csv y editar si es otro)
REPS=3          # repeticiones por b
N=8192          # 2^13
PR=2; PC=4
TAG=e_cfg4

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results logs

# Mismo entorno que los incisos (c) y (d): conda base (Open MPI 4.1.6, mpi4py, numpy, scipy).
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job $SLURM_JOB_ID en nodos: $SLURM_JOB_NODELIST"

for b in $BS; do
  for rep in $(seq 1 $REPS); do
    mpirun --bind-to none -n $SLURM_NTASKS python lu_distributed.py --N $N --b $b --pr $PR --pc $PC \
      --csv results/e_times.csv --tag ${TAG}_b${b}_rep${rep}
  done
done
