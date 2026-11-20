#!/bin/bash
#SBATCH --job-name=lu_d
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="Tarea 3 IIC3533 inciso d: escalamiento configuraciones 0 a 3"
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=3800M
#SBATCH --time=00:14:30
#SBATCH --output=logs/lu_d_%j.out
#SBATCH --error=logs/lu_d_%j.err
# Inciso (d): configuraciones 0 a 3 con N = 2^13. El valor de b se pasa con B_VAL
# (por defecto 512). Enviar desde la raiz del repo, una vez por cada b:
#   sbatch slurm/d_cfg0a3.sbatch
#   B_VAL=<b del inciso (c)> sbatch slurm/d_cfg0a3.sbatch

mkdir -p logs results

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

B_VAL=${B_VAL:-512}
N=8192
REPS=3
CSV_OUT="results/results_d_b${B_VAL}.csv"
rm -f "$CSV_OUT"

unset SLURM_NTASKS SLURM_NPROCS

run_case() {
    local cfg=$1
    local pr=$2
    local pc=$3
    local T=$4
    local P=$((pr * pc))

    export OMP_NUM_THREADS=$T
    export OPENBLAS_NUM_THREADS=$T
    export MKL_NUM_THREADS=$T

    for rep in $(seq 1 $REPS); do
        echo "[b=$B_VAL] Config $cfg | grid=${pr}x${pc} | T=$T | rep=$rep/$REPS"
        mpirun --bind-to none -n $P python lu_distributed.py \
            --N $N --b $B_VAL --pr $pr --pc $pc \
            --csv "$CSV_OUT" --tag "cfg${cfg}_${pr}x${pc}_T${T}_b${B_VAL}_r${rep}"
    done
}

# Calentamiento
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    mpirun --bind-to none -n 2 python lu_distributed.py --N 1024 --b 256 --pr 1 --pc 2 > /dev/null

# Configuracion 0: 1x1, T in {1, 2, 4, 8}
for T in 1 2 4 8; do
    run_case 0 1 1 $T
done

# Configuracion 1: 1x2 y 2x1, T in {1, 2, 4}
for T in 1 2 4; do
    run_case 1 1 2 $T
    run_case 1 2 1 $T
done

# Configuracion 2: 2x2, T in {1, 2}
for T in 1 2; do
    run_case 2 2 2 $T
done

# Configuracion 3: 2x4 y 4x2, T = 1
run_case 3 2 4 1
run_case 3 4 2 1
