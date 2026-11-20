#!/bin/bash
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="Tarea 3 IIC3533: prueba del entorno (conda, mpi4py) y de lu_distributed.py en el cluster"
#SBATCH --job-name=test_env
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3800M
#SBATCH --time=00:05:00
#SBATCH --output=logs/test_env_%j.out
#SBATCH --error=logs/test_env_%j.err
# Prueba rapida del entorno (1 nodo, 4 procesos). Enviar desde la raiz del repo:
#   sbatch slurm/test_env.sbatch
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
which python mpirun
python -c "import numpy, scipy, mpi4py; print(numpy.__version__, scipy.__version__, mpi4py.__version__)"

echo "== Ranks y hosts =="
mpirun --bind-to none -n $SLURM_NTASKS python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.Get_processor_name())"

echo "== LU 2x2 con --check (N chico) =="
mpirun --bind-to none -n $SLURM_NTASKS python lu_distributed.py --N 1024 --b 128 --pr 2 --pc 2 --check
