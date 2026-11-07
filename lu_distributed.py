# Factorizacion LU por bloques distribuida en un grid 2D pr x pc de procesos MPI,
# con distribucion bloque-ciclica: el bloque A_IJ vive en el proceso (I mod pr, J mod pc).
# Cada proceso guarda sus bloques en un arreglo local ordenado por indice global, de modo
# que sus bloques de la submatriz restante (I, J > k) son un sub-arreglo contiguo.
#
# Uso: mpirun -n P python lu_distributed.py --N N --b b --pr pr --pc pc
#             [--check] [--csv tiempos.csv] [--check-csv check.csv] [--tag etiqueta]
#
# Mide por proceso el tiempo de computo (lu_factor, solve_triangular, @) y de
# comunicacion (Bcast). El tiempo de comunicacion incluye la espera a que la raiz
# termine su computo (desbalance), porque Bcast bloquea hasta tener los datos.

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from scipy.linalg import solve_triangular

from lu_serial import lu_blocked, lu_diag_block, matrix_entries


def first_after(my, k, p):
    """Indice local del primer bloque (de fila o columna) con indice global > k.

    Los bloques locales tienen indices globales my, my + p, my + 2p, ...
    """
    return 0 if my > k else (k - my) // p + 1


def global_indices(my, p, nb, b):
    """Indices globales (de filas o columnas) de los bloques locales, en orden."""
    blocks = np.arange(my, nb, p)
    return (blocks[:, None] * b + np.arange(b)[None, :]).ravel()


def lu_distributed(Aloc, b, nb, pr, pc, myrow, mycol, row_comm, col_comm):
    """LU distribuida en el mismo arreglo local. Devuelve el diccionario de tiempos."""
    mloc, nloc = Aloc.shape[0] // b, Aloc.shape[1] // b
    T = dict(t_factor=0.0, t_trsm=0.0, t_gemm=0.0,
             t_bcast_diag=0.0, t_bcast_L=0.0, t_bcast_U=0.0)

    # Buffers reutilizados (contiguos) para lo que se recibe.
    Dkk = np.empty((b, b))
    Lbuf = np.empty((mloc * b, b))
    Ubuf = np.empty(b * nloc * b)

    for k in range(nb):
        r, c = k % pr, k % pc           # dueno de A_kk
        in_row, in_col = myrow == r, mycol == c
        li = first_after(myrow, k, pr)  # mis filas de bloque con I > k: li .. mloc-1
        lj = first_after(mycol, k, pc)  # mis columnas de bloque con J > k: lj .. nloc-1
        m, n = (mloc - li) * b, (nloc - lj) * b
        kr = (k // pr) * b              # fila local de A_kk (si in_row)
        kc = (k // pc) * b              # columna local de A_kk (si in_col)

        # 1. Bloque diagonal: (r, c) factoriza y difunde en su fila y su columna.
        if in_row and in_col:
            t = time.perf_counter()
            Akk = Aloc[kr:kr + b, kc:kc + b]
            lu_diag_block(Akk)
            Dkk[:] = Akk
            T["t_factor"] += time.perf_counter() - t
        t = time.perf_counter()
        if in_row:
            row_comm.Bcast(Dkk, root=c)
        if in_col:
            col_comm.Bcast(Dkk, root=r)
        T["t_bcast_diag"] += time.perf_counter() - t

        # 2-3. Paneles: fila r calcula U_kj (j > k); columna c calcula L_ik (i > k).
        t = time.perf_counter()
        if in_row and n:
            Aloc[kr:kr + b, lj * b:] = solve_triangular(
                Dkk, Aloc[kr:kr + b, lj * b:], lower=True, unit_diagonal=True,
                check_finite=False)
        if in_col and m:
            Aloc[li * b:, kc:kc + b] = solve_triangular(
                Dkk, Aloc[li * b:, kc:kc + b].T, trans="T", lower=False,
                check_finite=False).T
        T["t_trsm"] += time.perf_counter() - t

        # 4. Envio de paneles. Todos los procesos de una fila (columna) del grid tienen
        #    las mismas filas (columnas) de bloque, asi que m (n) es igual en todo el
        #    comunicador y el "if" es consistente entre sus miembros.
        Lp = Lbuf[:m]
        Up = Ubuf[:b * n].reshape(b, n)
        t = time.perf_counter()
        if m:
            if in_col:
                Lp[:] = Aloc[li * b:, kc:kc + b]
            row_comm.Bcast(Lp, root=c)
        T["t_bcast_L"] += time.perf_counter() - t
        t = time.perf_counter()
        if n:
            if in_row:
                Up[:] = Aloc[kr:kr + b, lj * b:]
            col_comm.Bcast(Up, root=r)
        T["t_bcast_U"] += time.perf_counter() - t

        # 5. Actualizacion: una sola multiplicacion sobre el sub-arreglo contiguo.
        if m and n:
            t = time.perf_counter()
            Aloc[li * b:, lj * b:] -= Lp @ Up
            T["t_gemm"] += time.perf_counter() - t

    T["t_comp"] = T["t_factor"] + T["t_trsm"] + T["t_gemm"]
    T["t_comm"] = T["t_bcast_diag"] + T["t_bcast_L"] + T["t_bcast_U"]
    return T


def append_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--b", type=int, default=512)
    p.add_argument("--pr", type=int, default=None)
    p.add_argument("--pc", type=int, default=None)
    p.add_argument("--check", action="store_true",
                   help="compara con lu_serial.py y calcula ||A - LU|| / ||A||")
    p.add_argument("--csv", default=None, help="agrega los tiempos por proceso")
    p.add_argument("--check-csv", default=None, help="agrega el resultado de --check")
    p.add_argument("--tag", default="", help="etiqueta libre para el CSV")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if args.pr is None or args.pc is None:
        pr, pc = MPI.Compute_dims(size, 2)
    else:
        pr, pc = args.pr, args.pc
    N, b = args.N, args.b
    if pr * pc != size:
        raise SystemExit(f"pr*pc = {pr * pc} != {size} procesos")
    if N % b:
        raise SystemExit(f"b = {b} debe dividir a N = {N}")
    nb = N // b

    myrow, mycol = rank // pc, rank % pc
    row_comm = comm.Split(myrow, mycol)   # rank en row_comm = mycol
    col_comm = comm.Split(mycol, myrow)   # rank en col_comm = myrow

    # Cada proceso genera solo sus bloques, a partir de los indices globales.
    rows = global_indices(myrow, pr, nb, b)
    cols = global_indices(mycol, pc, nb, b)
    Aloc = matrix_entries(rows, cols, N)

    comm.Barrier()
    t0 = time.perf_counter()
    T = lu_distributed(Aloc, b, nb, pr, pc, myrow, mycol, row_comm, col_comm)
    t1 = time.perf_counter()
    comm.Barrier()
    t_wall = time.perf_counter() - t0
    t_wall = comm.reduce(t_wall, op=MPI.MAX, root=0)

    threads = os.environ.get("OMP_NUM_THREADS", "")
    info = dict(tag=args.tag, N=N, b=b, pr=pr, pc=pc, P=size, T=threads,
                rank=rank, myrow=myrow, mycol=mycol, host=MPI.Get_processor_name(),
                t_total=t1 - t0, **T)
    all_info = comm.gather(info, root=0)

    if rank == 0:
        gflops = 2 * N**3 / 3 / t_wall / 1e9
        print(f"N={N} b={b} grid={pr}x{pc} T={threads or '?'} tiempo={t_wall:.3f} s "
              f"GFLOP/s={gflops:.1f}")
        print(f"{'rank':>4} {'(fila,col)':>10} {'comp [s]':>9} {'comm [s]':>9} "
              f"{'total [s]':>9}  host")
        for d in all_info:
            print(f"{d['rank']:>4} {str((d['myrow'], d['mycol'])):>10} "
                  f"{d['t_comp']:9.3f} {d['t_comm']:9.3f} {d['t_total']:9.3f}  {d['host']}")
        if args.csv:
            for d in all_info:
                d["t_wall"] = t_wall
            append_csv(args.csv, all_info)

    if args.check:
        locals_ = comm.gather((rows, cols, Aloc), root=0)
        if rank == 0:
            LU = np.empty((N, N))
            for r_, c_, a_ in locals_:
                LU[np.ix_(r_, c_)] = a_
            idx = np.arange(N)
            A = matrix_entries(idx, idx, N)
            ref = A.copy()
            lu_blocked(ref, b)
            diff = np.linalg.norm(LU - ref) / np.linalg.norm(ref)
            maxdiff = np.abs(LU - ref).max()
            Lm = np.tril(LU, -1) + np.eye(N)
            res = np.linalg.norm(A - Lm @ np.triu(LU)) / np.linalg.norm(A)
            print(f"||LU_dist - LU_serial||_F / ||LU_serial||_F = {diff:.3e}  "
                  f"max|dif| = {maxdiff:.3e}  ||A - LU||_F / ||A||_F = {res:.3e}")
            if args.check_csv:
                append_csv(args.check_csv, [dict(
                    tag=args.tag, N=N, b=b, pr=pr, pc=pc, P=size, nb=nb,
                    rel_diff_serial=diff, max_abs_diff=maxdiff, residual=res)])


if __name__ == "__main__":
    main()
