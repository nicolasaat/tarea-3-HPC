"""Inciso (a): grafica tiempo y GFLOP/s de lu_serial.py vs tamano de bloque b.

Lee results/a_serial.csv (generado por run_a.sh), imprime una tabla resumen
(mediana, min y max por b) y guarda figures/a_serial.png y .pdf.
Uso: python3 plot_a.py [--csv results/a_serial.csv]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLOR = "#2a78d6"   # serie unica
BEST = "#1c5cab"    # marca del b optimo (mismo tono, mas oscuro)
INK = "#52514e"     # texto secundario / ejes
GRID = "#e4e3df"


def load(path):
    data = defaultdict(lambda: {"t": [], "g": []})
    N = None
    with open(path) as f:
        for row in csv.DictReader(f):
            N = int(row["N"])
            b = int(row["b"])
            data[b]["t"].append(float(row["tiempo_s"]))
            data[b]["g"].append(float(row["gflops"]))
    bs = sorted(data)
    t = [np.array(data[b]["t"]) for b in bs]
    g = [np.array(data[b]["g"]) for b in bs]
    return N, np.array(bs), t, g


def style(ax):
    ax.set_xscale("log", base=2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, labelsize=11)
    ax.set_xlabel("Tamaño de bloque $b$", fontsize=12)


def panel(ax, bs, samples, ylabel, title, best_idx, fmt):
    med = np.array([np.median(s) for s in samples])
    lo = med - np.array([s.min() for s in samples])
    hi = np.array([s.max() for s in samples]) - med
    ax.errorbar(bs, med, yerr=[lo, hi], color=COLOR, linewidth=2,
                marker="o", markersize=8, capsize=4, elinewidth=1.2,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.plot(bs[best_idx], med[best_idx], "o", markersize=13, color=BEST,
            markeredgecolor="white", markeredgewidth=2, zorder=4)
    ax.annotate(f"$b={bs[best_idx]}$\n{fmt.format(med[best_idx])}",
                (bs[best_idx], med[best_idx]), textcoords="offset points",
                xytext=(0, 16 if "GFLOP" in ylabel else -38), ha="center",
                fontsize=11, color="#0b0b0b")
    ax.set_xticks(bs)
    ax.set_xticklabels([f"$2^{{{int(np.log2(b))}}}$" for b in bs])
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_ylim(0, max(med + hi) * 1.18)
    style(ax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/a_serial.csv")
    p.add_argument("--out", default="figures/a_serial")
    args = p.parse_args()

    N, bs, t, g = load(args.csv)
    t_med = np.array([np.median(x) for x in t])
    best = int(np.argmin(t_med))

    # Tabla resumen (para copiar al informe).
    print(f"N = {N}, {len(t[0])} repeticiones por b (mediana [min, max])")
    print(f"{'b':>6} | {'tiempo [s]':>22} | {'GFLOP/s':>22}")
    print("-" * 56)
    for b, ti, gi in zip(bs, t, g):
        mark = "  <- mejor" if b == bs[best] else ""
        print(f"{b:>6} | {np.median(ti):7.3f} [{ti.min():.3f}, {ti.max():.3f}] | "
              f"{np.median(gi):7.1f} [{gi.min():6.1f}, {gi.max():6.1f}]{mark}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    panel(ax1, bs, t, "Tiempo [s]", "Tiempo de factorización", best, "{:.3f} s")
    panel(ax2, bs, g, "GFLOP/s", "Rendimiento", best, "{:.0f} GFLOP/s")
    fig.suptitle(f"LU serial por bloques, $N = 2^{{{int(np.log2(N))}}}$, 1 thread "
                 f"(mediana de {len(t[0])} corridas; barras = mín–máx)",
                 fontsize=14, x=0.01, ha="left")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    print(f"Figura guardada en {out}.png y {out}.pdf")

    tex = Path(args.csv).with_suffix(".tex")
    write_latex(tex, N, bs, t, g)
    print(f"Tabla LaTeX guardada en {tex}")


def write_latex(path, N, bs, t, g):
    """Tabla booktabs con los mismos estadisticos que la figura (mediana, min, max)."""
    best = int(np.argmin([np.median(x) for x in t]))
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$b$ & Pasos $N/b$ & Mediana [ms] & Mín. [ms] & Máx. [ms] & GFLOP/s \\",
        r"\midrule",
    ]
    for i in range(len(bs)):
        ti = t[i] * 1e3
        row = [str(bs[i]), str(N // bs[i]), f"{np.median(ti):.0f}",
               f"{ti.min():.0f}", f"{ti.max():.0f}", f"{np.median(g[i]):.1f}"]
        if i == best:
            row = [f"\\textbf{{{c}}}" for c in row]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
