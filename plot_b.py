"""Inciso (b): resume la verificacion y los tiempos por proceso de lu_distributed.py.

Lee results/b_check.csv y results/b_times.csv (generados por run_b.sh) y escribe:
  results/b_check_summary.csv/.tex  verificacion contra lu_serial.py
  results/b_times_summary.csv/.tex  computo y comunicacion por proceso (mediana)
  figures/b_times.png/.pdf          barras apiladas computo/comunicacion por proceso
Uso: python3 plot_b.py                                   (resultados locales)
     python3 plot_b.py --res results/cluster --label cluster
     (en ese caso la figura se guarda como figures/b_times_cluster.png/.pdf)
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COMP = "#2a78d6"
COMM = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"
RES, FIG = Path("results"), Path("figures")
LABEL = "local"


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


def sci(x):
    """Notacion cientifica para LaTeX; 0 exacto se escribe como 0."""
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return rf"${m} \times 10^{{{int(e)}}}$"


def check_summary():
    groups = defaultdict(list)
    for r in read(RES / "b_check.csv"):
        groups[(int(r["N"]), int(r["b"]))].append(r)
    rows = []
    for (N, b), rs in groups.items():
        rows.append(dict(
            N=N, b=b, nb=N // b,
            grids=" ".join(f"{r['pr']}x{r['pc']}" for r in rs),
            max_rel_diff_serial=max(float(r["rel_diff_serial"]) for r in rs),
            max_abs_diff=max(float(r["max_abs_diff"]) for r in rs),
            max_residual=max(float(r["residual"]) for r in rs)))
    write_csv(RES / "b_check_summary.csv", rows)
    body = [[str(r["N"]), str(r["b"]), str(r["nb"]),
             r["grids"].replace("x", r"$\times$").replace(" ", ", "),
             sci(r["max_rel_diff_serial"]), sci(r["max_residual"])] for r in rows]
    write_tex(RES / "b_check_summary.tex", "rrrlrr",
              r"$N$ & $b$ & $N/b$ & Grids $p_r \times p_c$ & "
              r"$\frac{\|LU_{\mathrm{dist}} - LU_{\mathrm{serial}}\|_F}{\|LU_{\mathrm{serial}}\|_F}$ & "
              r"$\frac{\|A - LU\|_F}{\|A\|_F}$", body)
    print(f"Verificacion: {len(rows)} casos (N, b), "
          f"max dif. vs serial = {max(r['max_rel_diff_serial'] for r in rows):.1e}, "
          f"max residuo = {max(r['max_residual'] for r in rows):.1e}")


def times_summary():
    raw = read(RES / "b_times.csv")
    # Por grid y proceso: mediana sobre repeticiones de cada tiempo.
    by = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in raw:
        key = (int(r["pr"]), int(r["pc"]), int(r["rank"]))
        meta[key] = r
        for c in ("t_comp", "t_comm", "t_total", "t_wall", "t_factor", "t_trsm",
                  "t_gemm", "t_bcast_diag", "t_bcast_L", "t_bcast_U"):
            by[key][c].append(float(r[c]))
    keys = sorted(by, key=lambda k: (k[0] * k[1], k[0], k[2]))
    rows = []
    for k in keys:
        m = meta[k]
        d = {c: float(np.median(v)) for c, v in by[k].items()}
        rows.append(dict(N=int(m["N"]), b=int(m["b"]), pr=k[0], pc=k[1], T=m["T"],
                         rank=k[2], myrow=int(m["myrow"]), mycol=int(m["mycol"]),
                         reps=len(by[k]["t_comp"]), **d))
    write_csv(RES / "b_times_summary.csv", rows)

    body = []
    for r in rows:
        body.append([f"{r['pr']}$\\times${r['pc']}", str(r["rank"]),
                     f"({r['myrow']}, {r['mycol']})",
                     f"{r['t_comp'] * 1e3:.0f}", f"{r['t_comm'] * 1e3:.0f}",
                     f"{r['t_total'] * 1e3:.0f}",
                     f"{100 * r['t_comm'] / r['t_total']:.0f}"])
    write_tex(RES / "b_times_summary.tex", "lrcrrrr",
              r"Grid & Rank & (fila, col) & Cómputo [ms] & Comunicación [ms] & "
              r"Total [ms] & Com. [\%]", body)
    return rows


def plot_times(rows):
    grids = []
    for r in rows:
        g = (r["pr"], r["pc"])
        if g not in grids:
            grids.append(g)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x, ticks, labels, centers = 0.0, [], [], []
    for g in grids:
        rs = [r for r in rows if (r["pr"], r["pc"]) == g]
        xs = x + np.arange(len(rs))
        comp = np.array([r["t_comp"] for r in rs]) * 1e3
        comm = np.array([r["t_comm"] for r in rs]) * 1e3
        ax.bar(xs, comp, width=0.7, color=COMP, edgecolor="white", linewidth=2,
               label="Cómputo" if g == grids[0] else None, zorder=3)
        ax.bar(xs, comm, bottom=comp, width=0.7, color=COMM, edgecolor="white",
               linewidth=2, label="Comunicación" if g == grids[0] else None, zorder=3)
        ticks += list(xs)
        labels += [f"({r['myrow']},{r['mycol']})" for r in rs]
        centers.append((xs.mean(), f"{g[0]}×{g[1]}  ({rs[0]['t_wall'] * 1e3:.0f} ms)"))
        x = xs[-1] + 1.8

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=10)
    for cx, txt in centers:
        ax.annotate(txt, (cx, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -36), textcoords="offset points", ha="center",
                    fontsize=11, color="#0b0b0b")
    ax.set_ylabel("Tiempo por proceso [ms]", fontsize=12)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, labelsize=11)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    r0 = rows[0]
    ax.set_title(f"LU distribuida, $N={r0['N']}$, $b={r0['b']}$, T={r0['T']} "
                 f"({LABEL}, mediana de {r0['reps']} corridas). "
                 "Proceso (fila, col); grid (tiempo total)",
                 fontsize=12.5, loc="left")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    name = "b_times" if LABEL == "local" else f"b_times_{LABEL}"
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def main():
    global RES, LABEL
    p = argparse.ArgumentParser()
    p.add_argument("--res", default="results", help="carpeta con b_check.csv y b_times.csv")
    p.add_argument("--label", default="local", help="'local' o 'cluster' (titulo y nombre de figura)")
    args = p.parse_args()
    RES, LABEL = Path(args.res), args.label

    check_summary()
    rows = times_summary()
    plot_times(rows)
    print(f"{'grid':>5} {'rank':>4} {'comp[ms]':>9} {'comm[ms]':>9} {'total[ms]':>9}")
    for r in rows:
        print(f"{r['pr']}x{r['pc']:<3} {r['rank']:>4} {r['t_comp'] * 1e3:9.1f} "
              f"{r['t_comm'] * 1e3:9.1f} {r['t_total'] * 1e3:9.1f}")
    print(f"Escrito en {RES}/: b_check_summary.{{csv,tex}}, b_times_summary.{{csv,tex}}; "
          f"figura en {FIG}/")


if __name__ == "__main__":
    main()
