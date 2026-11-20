#!/bin/bash
#SBATCH --job-name=lu_c
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="Tarea 3 IIC3533 inciso c: barrido de tamano de bloque b en LU distribuida"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:14:00
#SBATCH --output=logs/lu_c_%j.out
#SBATCH --error=logs/lu_c_%j.err
# Inciso (c): barrido de b con N = 2^13, grid 2x2, T = 2. Tambien provee los datos del
# inciso (f) (b = N/2 = 4096 y b = N/16 = 512). Enviar desde la raiz del repo:
#   sbatch slurm/c_barrido_b.sbatch

mkdir -p logs results

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

CSV_OUT="results/results_c.csv"
rm -f "$CSV_OUT"

N=8192
REPS=5

mpirun --bind-to none -n $SLURM_NTASKS python lu_distributed.py --N 1024 --b 256 --pr 2 --pc 2 > /dev/null

for b in 4096 2048 1024 512 256 128 64; do
    for rep in $(seq 1 $REPS); do
        echo "Ejecutando N=$N b=$b rep=$rep/$REPS..."
        mpirun --bind-to none -n $SLURM_NTASKS python lu_distributed.py \
            --N $N --b $b --pr 2 --pc 2 \
            --csv "$CSV_OUT" --tag "c_b${b}_r${rep}"
    done
done
