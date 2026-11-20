#!/bin/bash
# Inciso (b) en el cluster: correctitud de lu_distributed.py contra lu_serial.py y
# tiempos de computo/comunicacion por proceso. Un nodo, hasta 8 procesos x 1 thread.
# Uso (en kraken, desde la raiz del repo, no desde slurm/; la carpeta logs/ debe existir):
#   mkdir -p logs && sbatch slurm/b_cluster.sbatch
# Salida: results/cluster/b_check.csv, results/cluster/b_times.csv y logs/lu_b_<jobid>.out
# (la ejecucion reportada en el informe uso la version anterior, con log b_cluster_41158.out)
#
#SBATCH --job-name=lu_b
#SBATCH --comment="IIC3533 Tarea 3 inciso (b): verificacion y tiempos de computo/comunicacion de LU distribuida con MPI"
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/lu_b_%j.out
#SBATCH --error=logs/lu_b_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Mismo entorno que los incisos (c)-(e): conda base (Open MPI 4.1.6, mpi4py, numpy, scipy).
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

OUT=results/cluster
mkdir -p "$OUT"
rm -f "$OUT/b_check.csv" "$OUT/b_times.csv"

echo "Nodo: $(hostname)  tareas: $SLURM_NTASKS  threads/tarea: $SLURM_CPUS_PER_TASK"
python -c "import numpy, scipy, mpi4py; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'mpi4py', mpi4py.__version__)"

GRIDS="1x1 1x2 2x1 2x2 2x4 4x2"

echo "== Correctitud =="
for Nb in 2048:1024 2048:256 2560:512 2560:128 2048:2048; do
  N=${Nb%:*}; b=${Nb#*:}
  for g in $GRIDS; do
    pr=${g%x*}; pc=${g#*x}
    mpirun --bind-to none -n $((pr * pc)) python lu_distributed.py --N "$N" --b "$b" \
      --pr "$pr" --pc "$pc" --check --check-csv "$OUT/b_check.csv" | tail -n 1
  done
done

echo "== Tiempos (N=4096, b=512) =="
for g in $GRIDS; do
  pr=${g%x*}; pc=${g#*x}
  for rep in 1 2 3; do
    mpirun --bind-to none -n $((pr * pc)) python lu_distributed.py --N 4096 --b 512 \
      --pr "$pr" --pc "$pc" --csv "$OUT/b_times.csv" --tag "rep$rep" | head -n 1
  done
done
echo "Listo: $OUT/b_check.csv y $OUT/b_times.csv"
