#!/bin/bash
#SBATCH --job-name=info_cluster
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --comment="Tarea 3 IIC3533: caracteristicas del hardware y software de los nodos"
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3800M
#SBATCH --time=00:05:00
#SBATCH --output=logs/info_cluster_%j.out
#SBATCH --error=logs/info_cluster_%j.err
# Enviar desde la raiz del repo: sbatch slurm/info_cluster.sbatch
# Resultado: logs/info_cluster_<jobid>.out (una seccion por nodo)

mkdir -p logs

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

echo "Job $SLURM_JOB_ID en nodos: $SLURM_JOB_NODELIST"

# Se ejecuta una vez en cada nodo
srun --ntasks-per-node=1 bash -c '
echo "=================== $(hostname) ==================="
echo "--- Sistema operativo y kernel"
grep PRETTY_NAME /etc/os-release; uname -srm
echo "--- CPU (modelo, sockets, nucleos, frecuencia)"
lscpu | grep -E "Model name|Architecture|Socket|Core|Thread|^CPU\(s\)|MHz|max MHz"
echo "--- Caches"
lscpu | grep -E "L1d|L1i|L2|L3"
echo "--- Memoria del nodo y limite de este job"
free -h | head -2
echo "CPUs visibles para el job: $(nproc)"
echo "--- Red (interfaces y velocidad en Mbit/s)"
ip -br link 2>/dev/null | grep -v " lo "
for f in /sys/class/net/*/speed; do echo "$f: $(cat $f 2>/dev/null)"; done
echo "--- CMA (aviso de Open MPI en el ping-pong)"
cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null
echo "--- Software"
which python mpirun
python --version
mpirun --version | head -1
python - <<PY
import numpy, scipy, mpi4py
print("NumPy", numpy.__version__, "| SciPy", scipy.__version__, "| mpi4py", mpi4py.__version__)
numpy.show_config()
PY
'

echo "=================== SLURM ==================="
scontrol show node ventress yodaxico | grep -E "NodeName|CPUTot|CPUAlloc|RealMemory|CoresPerSocket|ThreadsPerCore|Sockets|Gres"
sinfo -N -l
