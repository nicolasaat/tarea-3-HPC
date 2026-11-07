#!/bin/bash
# Inciso (a): lu_serial.py con N = 2^12 y un thread, variando b.
# Uso: ./run_a.sh [repeticiones]   -> escribe results/a_serial.csv

set -euo pipefail
cd "$(dirname "$0")"

# Un thread de BLAS (Accelerate en macOS usa VECLIB_MAXIMUM_THREADS).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

N=4096
REPS=${1:-3}
OUT=results/a_serial.csv
mkdir -p results
echo "N,b,rep,tiempo_s,gflops" > "$OUT"

for b in 4096 2048 1024 512 256 128 64; do
  for rep in $(seq 1 "$REPS"); do
    line=$(python3 lu_serial.py --N "$N" --b "$b")
    echo "$line"
    t=$(echo "$line" | sed -E 's/.*tiempo=([0-9.]+).*/\1/')
    g=$(echo "$line" | sed -E 's/.*GFLOP\/s=([0-9.]+).*/\1/')
    echo "$N,$b,$rep,$t,$g" >> "$OUT"
  done
done
echo "Resultados en $OUT"
