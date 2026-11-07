#!/bin/bash
# Inciso (b), pruebas locales de lu_distributed.py (maximo 4 procesos x 1 thread,
# para no sobresuscribir los 4 nucleos de rendimiento del M2).
#   1) Correctitud: compara contra lu_serial.py en varios grids y tamanos, incluyendo
#      casos donde N/b no es multiplo de pr o pc, y b = N (un solo bloque).
#   2) Tiempos de computo y comunicacion por proceso, N = 4096, b = 512.
# Uso: ./run_b.sh [repeticiones]  -> results/b_check.csv, results/b_times.csv

set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
REPS=${1:-3}
CHECK=results/b_check.csv
TIMES=results/b_times.csv
mkdir -p results
rm -f "$CHECK" "$TIMES"

GRIDS="1x1 1x2 2x1 2x2 1x4 4x1"

echo "== Correctitud =="
for Nb in 2048:1024 2048:256 2560:512 2560:128 2048:2048; do
  N=${Nb%:*}; b=${Nb#*:}
  for g in $GRIDS; do
    pr=${g%x*}; pc=${g#*x}
    mpirun -n $((pr * pc)) python3 lu_distributed.py --N "$N" --b "$b" \
      --pr "$pr" --pc "$pc" --check --check-csv "$CHECK" | tail -n 1
  done
done

echo "== Tiempos (N=4096, b=512) =="
for g in 1x1 1x2 2x1 2x2; do
  pr=${g%x*}; pc=${g#*x}
  for rep in $(seq 1 "$REPS"); do
    mpirun -n $((pr * pc)) python3 lu_distributed.py --N 4096 --b 512 \
      --pr "$pr" --pc "$pc" --csv "$TIMES" --tag "rep$rep" | head -n 1
  done
done
echo "Resultados en $CHECK y $TIMES"
