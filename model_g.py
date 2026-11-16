"""Inciso (g): modelo de costo alpha-beta-gamma frente a las mediciones del barrido de b (inciso c).

Modelo (broadcast en arbol binomial; L = log2(pr) + log2(pc); p = pr*pc; word = 8 bytes):
  cómputo   gamma * [ 2N^3/(3p)  +  (N^2 b / 2)(1/pr + 1/pc)  +  (2/3) N b^2 ]
              actualización      paneles (camino crítico)      LU de bloques diagonales
  latencia  alpha * 2 (N/b) L
  ancho de banda  beta_w * [ (N^2/2)(log2(pc)/pr + log2(pr)/pc)  +  N b L ]     (beta_w = beta por palabra)
  T(N,b,pr,pc) = cómputo + latencia + ancho de banda

Parametros: gamma sale de T(1x1, T) de la config 0 del inciso (d) (--t-ref); alpha y beta del ping-pong.
Compara con results/results_c.csv (grid 2x2, N = 8192, T = 2) y estima la espera como
(comunicación medida) - (alpha + beta del modelo).

Lee:  results/results_c.csv, results/pingpong_fit.csv
Escribe: results/g_model.csv/.tex, figures/g_model.png/.pdf
Uso: python3 model_g.py [--t-ref 8.107] [--tag intra] [--N 8192] [--pr 2] [--pc 2]
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

COMP, COMM, WAIT = "#2a78d6", "#eb6834", "#b9b7b0"
INK, GRID = "#52514e", "#e4e3df"
RES, FIG = Path("results"), Path("figures")


def model_terms(N, b, pr, pc, gamma, alpha, beta_w):
    p, L = pr * pc, math.log2(pr) + math.log2(pc)
    flops = 2 * N**3 / (3 * p) + N**2 * b / 2 * (1 / pr + 1 / pc) + 2 * N * b**2 / 3
    lat = alpha * 2 * (N / b) * L
    bw = beta_w * ((N**2 / 2) * (math.log2(pc) / pr + math.log2(pr) / pc) + N * b * L)
    return dict(flops=flops, comp=gamma * flops, lat=lat, bw=bw, total=gamma * flops + lat + bw)


def b_star(N, pr, pc, gamma, alpha):
    """Minimo de  alpha*2(N/b)L + gamma*(N^2 b/2)(1/pr+1/pc):  b* = sqrt(4 alpha L / (gamma N (1/pr+1/pc)))."""
    L = math.log2(pr) + math.log2(pc)
    return math.sqrt(4 * alpha * L / (gamma * N * (1 / pr + 1 / pc))) if L > 0 else float("nan")


def load_measured(path, pr, pc):
    runs = defaultdict(lambda: defaultdict(list))        # (b, tag) -> columnas por rank
    wall = {}
    for r in csv.DictReader(open(path)):
        if (int(r["pr"]), int(r["pc"])) != (pr, pc):
            continue
        key = (int(r["b"]), r["tag"])
        runs[key]["comp"].append(float(r["t_comp"]))
        runs[key]["comm"].append(float(r["t_comm"]))
        wall[key] = float(r["t_wall"])
    by_b = defaultdict(lambda: defaultdict(list))
    for (b, tag), d in runs.items():
        by_b[b]["wall"].append(wall[(b, tag)])
        by_b[b]["comp"].append(np.mean(d["comp"]))       # promedio entre procesos
        by_b[b]["comm"].append(np.mean(d["comm"]))
    return {b: {k: float(np.median(v)) for k, v in d.items()} for b, d in by_b.items()}


def load_pingpong(path, tag):
    for r in csv.DictReader(open(path)):
        if r["tag"] == tag:
            return float(r["alpha_us"]) * 1e-6, float(r["beta_ns_per_word8"]) * 1e-9
    raise SystemExit(f"No hay tag '{tag}' en {path}")


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, labelsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-c", default=str(RES / "results_c.csv"))
    ap.add_argument("--pingpong", default=str(RES / "pingpong_fit.csv"))
    ap.add_argument("--tag", default="intra", help="fila del ping-pong a usar (intra o inter)")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--pr", type=int, default=2)
    ap.add_argument("--pc", type=int, default=2)
    ap.add_argument("--t-ref", type=float, default=8.107,
                    help="T(1x1) de la config 0 del inciso (d) con el mismo T que (c): T=2, b=2048 -> 8.107 s")
    a = ap.parse_args()
    N, pr, pc = a.N, a.pr, a.pc

    gamma = a.t_ref / (2 * N**3 / 3)
    alpha, beta_w = load_pingpong(a.pingpong, a.tag)
    meas = load_measured(a.results_c, pr, pc)
    bs = sorted(meas)
    print(f"gamma = {gamma:.3e} s/flop ({1 / gamma / 1e9:.1f} GFLOP/s)   alpha = {alpha * 1e6:.2f} us   "
          f"beta = {beta_w * 1e9:.2f} ns/palabra (8 B) [{a.tag}]")

    rows = []
    for b in bs:
        m, t = meas[b], model_terms(N, b, pr, pc, gamma, alpha, beta_w)
        comm_model = t["lat"] + t["bw"]
        rows.append(dict(
            b=b, steps=N // b, t_meas=m["wall"], t_model=t["total"],
            comp_meas=m["comp"], comm_meas=m["comm"],
            comp_model=t["comp"], lat_model=t["lat"], bw_model=t["bw"],
            wait_est=max(m["comm"] - comm_model, 0.0),
            gamma_eff_ratio=(m["comp"] / t["flops"]) / gamma))
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    with open(RES / "g_model.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [r"\begin{tabular}{cccccccc}", r"\toprule",
             r"$b$ & $N/b$ & $T$ medido [s] & $T$ modelo [s] & Cómputo medido [s] & Com. medida [s] & "
             r"Com. modelo [s] & Espera est. [s] \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join([str(r["b"]), str(r["steps"]), f"{r['t_meas']:.2f}", f"{r['t_model']:.2f}",
                                 f"{r['comp_meas']:.2f}", f"{r['comm_meas']:.2f}",
                                 f"{r['lat_model'] + r['bw_model']:.3f}", f"{r['wait_est']:.2f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (RES / "g_model.tex").write_text("\n".join(lines))

    bstar = b_star(N, pr, pc, gamma, alpha)
    grid_b = np.unique(np.round(np.geomspace(32, N, 200)).astype(int))
    tot_model = [model_terms(N, int(b), pr, pc, gamma, alpha, beta_w)["total"] for b in grid_b]
    best_meas = min(rows, key=lambda r: r["t_meas"])["b"]
    best_model = int(grid_b[int(np.argmin(tot_model))])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    a1.plot(bs, [r["t_meas"] for r in rows], "o-", color=COMP, lw=1.8, ms=5, label="Medido", zorder=3)
    a1.plot(grid_b, tot_model, "--", color=COMM, lw=1.8, label="Modelo ($\\alpha$-$\\beta$-$\\gamma$)", zorder=2)
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_yticks([2, 3, 5, 8, 12, 16])
    a1.yaxis.set_major_formatter(ScalarFormatter())
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.set_xlabel("Tamaño de bloque $b$", fontsize=12)
    a1.set_ylabel("Tiempo de factorización [s]", fontsize=12)
    a1.set_title("Medido vs modelo", fontsize=12, loc="left")
    a1.legend(frameon=False, fontsize=11)
    style(a1)
    a1.grid(True, axis="x", color=GRID, linewidth=0.8)

    x = np.arange(len(rows))
    comp = np.array([r["comp_meas"] for r in rows])
    tr = np.array([r["lat_model"] + r["bw_model"] for r in rows])
    wt = np.array([r["wait_est"] for r in rows])
    a2.bar(x, comp, 0.7, color=COMP, edgecolor="white", linewidth=1.5, label="Cómputo (medido)", zorder=3)
    a2.bar(x, tr, 0.7, bottom=comp, color=COMM, edgecolor="white", linewidth=1.5,
           label="Transferencia ($\\alpha+\\beta$, modelo)", zorder=3)
    a2.bar(x, wt, 0.7, bottom=comp + tr, color=WAIT, edgecolor="white", linewidth=1.5,
           label="Espera / sincronización (estimada)", zorder=3)
    a2.set_xticks(x)
    a2.set_xticklabels([str(r["b"]) for r in rows], fontsize=11)
    a2.set_xlabel("Tamaño de bloque $b$", fontsize=12)
    a2.set_ylabel("Tiempo promedio por proceso [s]", fontsize=12)
    a2.set_title("Dónde se va el tiempo medido", fontsize=12, loc="left")
    a2.legend(frameon=False, fontsize=10, loc="upper center")
    a2.set_ylim(0, max(comp + tr + wt) * 1.35)
    style(a2)
    fig.suptitle(f"Modelo de costo, grid {pr}×{pc}, $N={N}$", fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"g_model.{ext}", dpi=200, bbox_inches="tight")

    print(f"\n{'b':>5} {'T med':>7} {'T mod':>7} {'comp med':>9} {'comm med':>9} {'comm mod':>9} "
          f"{'espera':>7} {'g_ef/g':>7}")
    for r in rows:
        print(f"{r['b']:>5} {r['t_meas']:7.2f} {r['t_model']:7.2f} {r['comp_meas']:9.2f} {r['comm_meas']:9.2f} "
              f"{r['lat_model'] + r['bw_model']:9.3f} {r['wait_est']:7.2f} {r['gamma_eff_ratio']:7.2f}")
    print("\n(g_ef/g = gamma efectivo / gamma de referencia; solo es interpretable donde el computo esta balanceado, "
          "b <~ 1024: con b grande el modelo usa el camino critico y lo medido es un promedio entre procesos.)")
    print(f"\nb* analitico (latencia vs camino critico) = {bstar:.1f};  minimo numerico del modelo = {best_model};  "
          f"mejor b medido = {best_meas}")
    print("Escrito: results/g_model.{csv,tex}, figures/g_model.{png,pdf}")


if __name__ == "__main__":
    main()