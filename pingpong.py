"""Microbenchmark ping-pong entre 2 procesos MPI: estima alpha (latencia) y beta (1/ancho de banda).

Mide el tiempo de ida (mitad del ida y vuelta) para mensajes de 1 B a 128 MiB, y ajusta
    t(s) = alpha + beta * s     (s en bytes)
con los mensajes grandes. Se usa para comparar comunicacion dentro de un nodo (2 procesos en el
mismo nodo) y entre nodos (2 procesos en nodos distintos), segun como se lance el job.

Uso: mpirun --bind-to none -n 2 python pingpong.py --tag intra [--csv results/pingpong.csv]
"""

import argparse
import csv
import statistics as st
from pathlib import Path

import numpy as np
from mpi4py import MPI


def one_way_time(comm, rank, buf, nreps):
    """Tiempo de ida (s): mitad del ida y vuelta, promediado sobre nreps."""
    comm.Barrier()
    t0 = MPI.Wtime()
    for _ in range(nreps):
        if rank == 0:
            comm.Send(buf, dest=1, tag=7)
            comm.Recv(buf, source=1, tag=7)
        else:
            comm.Recv(buf, source=0, tag=7)
            comm.Send(buf, dest=0, tag=7)
    return (MPI.Wtime() - t0) / (2 * nreps)


def run(comm, tag, trials=5, kmax=27):
    rank = comm.Get_rank()
    if comm.Get_size() != 2:
        raise SystemExit("pingpong.py necesita exactamente 2 procesos")
    host = MPI.Get_processor_name()
    hosts = comm.gather(host, root=0)
    rows = []
    for k in range(0, kmax + 1):
        size = 2 ** k
        buf = np.zeros(size, dtype=np.uint8)
        nreps = max(5, min(2000, int(2e8 // size)))
        one_way_time(comm, rank, buf, max(2, nreps // 10))      # calentamiento
        ts = [one_way_time(comm, rank, buf, nreps) for _ in range(trials)]
        if rank == 0:
            t = st.median(ts)
            rows.append(dict(tag=tag, bytes=size, t_oneway_s=t, bw_GBs=size / t / 1e9,
                             host0=hosts[0], host1=hosts[1]))
    return rows


def fit_alpha_beta(rows, min_bytes=2 ** 20):
    """alpha = tiempo del mensaje mas pequeno; beta = pendiente de t vs bytes (mensajes grandes)."""
    alpha = min(rows, key=lambda r: r["bytes"])["t_oneway_s"]
    big = [r for r in rows if r["bytes"] >= min_bytes]
    s = np.array([r["bytes"] for r in big], dtype=float)
    t = np.array([r["t_oneway_s"] for r in big], dtype=float)
    beta, _ = np.polyfit(s, t, 1)
    return alpha, beta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="pingpong", help="etiqueta, p. ej. intra o inter")
    p.add_argument("--csv", default=None, help="archivo donde se guardan los puntos")
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    rows = run(comm, args.tag)
    if comm.Get_rank() != 0:
        return
    alpha, beta = fit_alpha_beta(rows)
    print(f"[{args.tag}] hosts: {rows[0]['host0']} <-> {rows[0]['host1']}")
    print(f"{'bytes':>12} {'t ida [us]':>12} {'GB/s':>8}")
    for r in rows:
        print(f"{r['bytes']:>12} {r['t_oneway_s'] * 1e6:12.2f} {r['bw_GBs']:8.3f}")
    print(f"alpha ~ {alpha * 1e6:.2f} us   beta ~ {beta * 1e9:.3f} ns/B "
          f"(= {beta * 8 * 1e9:.3f} ns/palabra de 8 B, ancho de banda ~ {1 / beta / 1e9:.2f} GB/s)")
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
