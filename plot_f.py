"""Inciso (f): grid 2x2, b = N/2 vs b = N/16, computo y comunicacion por proceso.

Lee results/results_c.csv (barrido de b del inciso (c): N = 2^13, grid 2x2) y escribe:
  results/f_times_summary.csv/.tex  mediana por proceso de computo y comunicacion
  results/f_breakdown.csv           desglose: factor, trsm, gemm, Bcast diag/L/U
  figures/f_times.png/.pdf          barras apiladas computo/comunicacion por proceso
  figures/f_breakdown.png/.pdf      desglose por operacion
Uso: python3 plot_f.py [ruta_csv]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COMP = "#2a78d6"
COMM = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"
RES, FIG = Path("results"), Path("figures")
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "results_c.csv"
COLS = ("t_comp", "t_comm", "t_total", "t_wall", "t_factor", "t_trsm", "t_gemm",
        "t_bcast_diag", "t_bcast_L", "t_bcast_U")


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_tex(path, cols, header, body):
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path(path).write_text("\n".join(lines))


def summarize():
    raw = [r for r in read(SRC) if (int(r["pr"]), int(r["pc"])) == (2, 2)]
    if not raw:
        raise SystemExit("No hay filas de grid 2x2 en " + str(SRC))
    N = int(raw[0]["N"])
    bs = {"N/2": N // 2, "N/16": N // 16}
    by = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in raw:
        b = int(r["b"])
        if b not in bs.values():
            continue
        key = (b, int(r["rank"]))
        meta[key] = r
        for c in COLS:
            by[key][c].append(float(r[c]))
    if not by:
        raise SystemExit(f"No hay corridas con b = {bs['N/2']} ni b = {bs['N/16']}")
    rows = []
    for (b, rank) in sorted(by, key=lambda k: (-k[0], k[1])):
        m = meta[(b, rank)]
        d = {c: float(np.median(v)) for c, v in by[(b, rank)].items()}
        rows.append(dict(N=N, b=b, b_rel="N/2" if b == N // 2 else "N/16",
                         pr=2, pc=2, T=m["T"], rank=rank,
                         myrow=int(m["myrow"]), mycol=int(m["mycol"]),
                         reps=len(by[(b, rank)]["t_comp"]), **d))
    write_csv(RES / "f_times_summary.csv", rows)
    write_csv(RES / "f_breakdown.csv", [
        dict(b=r["b"], rank=r["rank"], proc=f"({r['myrow']},{r['mycol']})",
             factor=r["t_factor"], trsm=r["t_trsm"], gemm=r["t_gemm"],
             bcast_diag=r["t_bcast_diag"], bcast_L=r["t_bcast_L"], bcast_U=r["t_bcast_U"])
        for r in rows])

    body = []
    for r in rows:
        body.append([f"{r['b']} ({r['b_rel']})", f"({r['myrow']}, {r['mycol']})",
                     f"{r['t_comp']:.2f}", f"{r['t_comm']:.2f}", f"{r['t_total']:.2f}",
                     f"{100 * r['t_comm'] / (r['t_comp'] + r['t_comm']):.0f}"])
    write_tex(RES / "f_times_summary.tex", "lcrrrr",
              r"$b$ & Proceso & Cómputo [s] & Comunicación [s] & Total [s] & Com. [\%]", body)
    return rows


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, labelsize=11)


def plot_times(rows):
    groups = sorted({r["b"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(groups), figsize=(10, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    r0 = rows[0]
    for ax, b in zip(axes, groups):
        rs = [r for r in rows if r["b"] == b]
        xs = np.arange(len(rs))
        comp = np.array([r["t_comp"] for r in rs])
        comm = np.array([r["t_comm"] for r in rs])
        ax.bar(xs, comp, width=0.7, color=COMP, edgecolor="white", linewidth=2,
               label="Cómputo", zorder=3)
        ax.bar(xs, comm, bottom=comp, width=0.7, color=COMM, edgecolor="white",
               linewidth=2, label="Comunicación", zorder=3)
        for x, c1, c2 in zip(xs, comp, comm):
            ax.text(x, c1 + c2 + 0.08, f"{100 * c2 / (c1 + c2):.0f}% com.",
                    ha="center", fontsize=9.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"({r['myrow']},{r['mycol']})" for r in rs], fontsize=11)
        ax.set_xlabel("Proceso (fila, col)", fontsize=12)
        ax.set_title(f"$b={b}$ ($N/{r0['N'] // b}$), {r0['N'] // b}×{r0['N'] // b} bloques\n"
                     f"tiempo total {rs[0]['t_wall']:.2f} s", fontsize=12)
        style(ax)
    axes[0].set_ylabel("Tiempo por proceso [s]", fontsize=12)
    axes[0].legend(frameon=False, fontsize=11, loc="upper left")
    fig.suptitle(f"LU distribuida, grid 2×2, $N={r0['N']}$, T={r0['T']} "
                 f"(cluster, mediana de {r0['reps']} corridas)", fontsize=12.5, x=0.01,
                 ha="left")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"f_times.{ext}", dpi=200, bbox_inches="tight")


def plot_breakdown(rows):
    parts = [("t_factor", "LU bloque diagonal", "#1f4e99"),
             ("t_trsm", "Paneles (trsm)", "#2a78d6"),
             ("t_gemm", "Actualización (gemm)", "#8db8ee"),
             ("t_bcast_diag", "Bcast diagonal", "#b8431a"),
             ("t_bcast_L", "Bcast panel L", "#eb6834"),
             ("t_bcast_U", "Bcast panel U", "#f3a98a")]
    groups = sorted({r["b"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(groups), figsize=(11, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    r0 = rows[0]
    for ax, b in zip(axes, groups):
        rs = [r for r in rows if r["b"] == b]
        xs = np.arange(len(rs))
        bottom = np.zeros(len(rs))
        for key, label, color in parts:
            v = np.array([r[key] for r in rs])
            ax.bar(xs, v, bottom=bottom, width=0.7, color=color, edgecolor="white",
                   linewidth=1.5, label=label, zorder=3)
            bottom += v
        ax.set_xticks(xs)
        ax.set_xticklabels([f"({r['myrow']},{r['mycol']})" for r in rs], fontsize=11)
        ax.set_xlabel("Proceso (fila, col)", fontsize=12)
        ax.set_title(f"$b={b}$", fontsize=12)
        style(ax)
    axes[0].set_ylabel("Tiempo por proceso [s]", fontsize=12)
    axes[-1].legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"Desglose por operación, grid 2×2, $N={r0['N']}$, T={r0['T']}",
                 fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"f_breakdown.{ext}", dpi=200, bbox_inches="tight")


def main():
    RES.mkdir(exist_ok=True)
    rows = summarize()
    plot_times(rows)
    plot_breakdown(rows)
    print(f"{'b':>5} {'proc':>7} {'comp[s]':>8} {'comm[s]':>8} {'total[s]':>8} {'comm%':>6}")
    for r in rows:
        print(f"{r['b']:>5} ({r['myrow']},{r['mycol']}) {r['t_comp']:8.2f} {r['t_comm']:8.2f} "
              f"{r['t_total']:8.2f} {100 * r['t_comm'] / (r['t_comp'] + r['t_comm']):6.0f}")
    print("Escrito: results/f_times_summary.{csv,tex}, results/f_breakdown.csv, "
          "figures/f_times.{png,pdf}, figures/f_breakdown.{png,pdf}")


if __name__ == "__main__":
    main()