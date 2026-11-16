"""Inciso (e): configuracion 3 (2x4, 1 nodo) vs 4 (2x4, 2 nodos) vs 5 (4x4, 2 nodos), T = 1.

Lee: results/results_d_b512.csv, results/results_d_b2048.csv (inciso d: config 0 para T(1) y config 3)
     results/e_times.csv (configs 4 y 5)
Escribe: results/e_summary.csv/.tex   tiempo, speedup, eficiencia, computo y comunicacion
         results/e_bcast.csv/.tex     desglose de la comunicacion (Bcast diagonal, L y U)
         figures/e_times.png/.pdf     barras apiladas computo + Bcast por configuracion
Computo y comunicacion = promedio entre procesos de cada corrida; se reporta la mediana de las corridas
(misma convencion que los incisos c y d). Speedup S = T(1)/T, eficiencia E = S/CPUs, con
T(1) = mediana de la config 0 (1x1, T=1) del inciso (d) con el mismo b.
Uso: python3 plot_e.py [--d512 ...] [--d2048 ...] [--e ...]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

COMP, DIAG, LPAN, UPAN = "#2a78d6", "#b8431a", "#eb6834", "#f3a98a"
INK, GRID = "#52514e", "#e4e3df"
RES, FIG = Path("results"), Path("figures")
COLS = ("t_comp", "t_comm", "t_bcast_diag", "t_bcast_L", "t_bcast_U")
# (numero de config, etiqueta, pr, pc, nodos)
CONFIGS = [(3, "Config. 3", 2, 4, 1), (4, "Config. 4", 2, 4, 2), (5, "Config. 5", 4, 4, 2)]


def read(path):
    return list(csv.DictReader(open(path)))


def summarize(rows):
    """Promedio entre procesos por corrida; mediana entre corridas."""
    runs = defaultdict(lambda: defaultdict(list))
    wall = {}
    for r in rows:
        for c in COLS:
            runs[r["tag"]][c].append(float(r[c]))
        wall[r["tag"]] = float(r["t_wall"])
    out = {c: median(np.mean(d[c]) for d in runs.values()) for c in COLS}
    out["wall"] = median(wall.values())
    out["reps"] = len(runs)
    return out


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, labelsize=11)


def write_tex(path, cols, header, body):
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d512", default=str(RES / "results_d_b512.csv"))
    ap.add_argument("--d2048", default=str(RES / "results_d_b2048.csv"))
    ap.add_argument("--e", default=str(RES / "e_times.csv"))
    a = ap.parse_args()
    d = {512: read(a.d512), 2048: read(a.d2048)}
    e = read(a.e)
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    rows = []
    for b in (512, 2048):
        t1 = median(float(r["t_wall"]) for r in d[b]
                    if (r["pr"], r["pc"], r["T"]) == ("1", "1", "1"))
        for cfg, label, pr, pc, nodes in CONFIGS:
            if cfg == 3:
                sel = [r for r in d[b] if (int(r["pr"]), int(r["pc"]), r["T"]) == (pr, pc, "1")]
            else:
                sel = [r for r in e if int(r["b"]) == b and (int(r["pr"]), int(r["pc"])) == (pr, pc)]
            if not sel:
                print(f"(sin datos para config {cfg}, b={b})")
                continue
            s = summarize(sel)
            cpus = pr * pc
            rows.append(dict(b=b, cfg=cfg, label=label, grid=f"{pr}x{pc}", nodes=nodes, cpus=cpus,
                             t1=t1, wall=s["wall"], speedup=t1 / s["wall"],
                             eff=t1 / s["wall"] / cpus, comp=s["t_comp"], comm=s["t_comm"],
                             comm_pct=100 * s["t_comm"] / (s["t_comp"] + s["t_comm"]),
                             bcast_diag=s["t_bcast_diag"], bcast_L=s["t_bcast_L"],
                             bcast_U=s["t_bcast_U"], reps=s["reps"]))
    with open(RES / "e_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_tex(RES / "e_summary.tex", "ccccccccccc",
              r"$b$ & Config. & Grid & Nodos & CPUs & Tiempo [s] & Speedup & Efic. & Comp. [s] & Com. [s] & Com. [\%]",
              [[str(r["b"]), str(r["cfg"]), r["grid"].replace("x", r"$\times$"), str(r["nodes"]),
                str(r["cpus"]), f"{r['wall']:.3f}", f"{r['speedup']:.2f}", f"{r['eff']:.2f}",
                f"{r['comp']:.3f}", f"{r['comm']:.3f}", f"{r['comm_pct']:.1f}"] for r in rows])
    write_tex(RES / "e_bcast.tex", "cccrrrr",
              r"$b$ & Config. & Grid & Bcast diag. [s] & Bcast $L$ [s] & Bcast $U$ [s] & Com. total [s]",
              [[str(r["b"]), str(r["cfg"]), r["grid"].replace("x", r"$\times$"),
                f"{r['bcast_diag']:.2f}", f"{r['bcast_L']:.2f}", f"{r['bcast_U']:.2f}",
                f"{r['comm']:.2f}"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, b in zip(axes, (512, 2048)):
        rs = [r for r in rows if r["b"] == b]
        xs = np.arange(len(rs))
        comp = np.array([r["comp"] for r in rs])
        parts = [("bcast_diag", DIAG, "Bcast diagonal"), ("bcast_L", LPAN, "Bcast panel $L$"),
                 ("bcast_U", UPAN, "Bcast panel $U$")]
        ax.bar(xs, comp, 0.65, color=COMP, edgecolor="white", linewidth=1.5, label="Cómputo", zorder=3)
        bottom = comp.copy()
        for key, color, label in parts:
            v = np.array([r[key] for r in rs])
            ax.bar(xs, v, 0.65, bottom=bottom, color=color, edgecolor="white", linewidth=1.5,
                   label=label, zorder=3)
            bottom += v
        for x, r, top in zip(xs, rs, bottom):
            ax.text(x, top + 0.12, f"$T$ = {r['wall']:.2f} s\n{r['comm_pct']:.0f}% com.",
                    ha="center", fontsize=9.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['label']}\n{r['grid'].replace('x', '×')}, {r['nodes']} "
                            f"{'nodo' if r['nodes'] == 1 else 'nodos'}" for r in rs], fontsize=10)
        ax.set_title(f"$b = {b}$", fontsize=12)
        ax.set_ylim(0, 12.5)
        style(ax)
    axes[0].set_ylabel("Tiempo promedio por proceso [s]", fontsize=12)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle("LU distribuida, $N=8192$, $T=1$: un nodo vs dos nodos", fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"e_times.{ext}", dpi=200, bbox_inches="tight")

    print(f"{'b':>5} {'cfg':>3} {'grid':>5} {'nodos':>5} {'T[s]':>7} {'S':>5} {'E':>5} {'comp':>6} {'comm':>6} "
          f"{'com%':>5} {'diag':>5} {'L':>5} {'U':>5}")
    for r in rows:
        print(f"{r['b']:>5} {r['cfg']:>3} {r['grid']:>5} {r['nodes']:>5} {r['wall']:7.3f} {r['speedup']:5.2f} "
              f"{r['eff']:5.2f} {r['comp']:6.2f} {r['comm']:6.2f} {r['comm_pct']:5.1f} "
              f"{r['bcast_diag']:5.2f} {r['bcast_L']:5.2f} {r['bcast_U']:5.2f}")
    print("Escrito: results/e_summary.{csv,tex}, results/e_bcast.{csv,tex}, figures/e_times.{png,pdf}")


if __name__ == "__main__":
    main()