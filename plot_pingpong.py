"""Ping-pong (incisos e y g): tiempo y ancho de banda vs tamano de mensaje, y ajuste de alpha y beta.

Lee results/pingpong.csv (generado por pingpong.py; tags 'intra' y/o 'inter') y escribe:
  results/pingpong_fit.csv/.tex   alpha, beta y ancho de banda por tag
  figures/pingpong.png/.pdf       tiempo de ida (izq.) y ancho de banda (der.) vs bytes
Uso: python3 plot_pingpong.py [ruta_csv]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#52514e"
GRID = "#e4e3df"
COLORS = {"intra": "#2a78d6", "inter": "#eb6834"}
LABELS = {"intra": "Mismo nodo", "inter": "Entre nodos"}
RES, FIG = Path("results"), Path("figures")
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "pingpong.csv"
MIN_FIT = 2 ** 20           # el ajuste de beta usa mensajes >= 1 MiB
# Tamano de un bloque b x b de doubles, para ubicar los b usados en los experimentos.
BLOCKS = [(512, "b=512"), (2048, "b=2048"), (4096, "b=4096")]


def load():
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(SRC)):
        by[r["tag"]][int(r["bytes"])].append(float(r["t_oneway_s"]))
    data = {}
    for tag, d in by.items():
        sizes = np.array(sorted(d))
        t = np.array([np.median(d[s]) for s in sizes])      # mediana si hay varias corridas
        data[tag] = (sizes.astype(float), t)
    return data


def fit(s, t):
    alpha = t[0]                                              # mensaje mas pequeno
    m = s >= MIN_FIT
    beta, _ = np.polyfit(s[m], t[m], 1)
    return alpha, beta


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, labelsize=11)


def size_fmt(x, _=None):
    for unit, f in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if x >= f:
            return f"{x / f:g} {unit}"
    return f"{x:g} B"


def main():
    data = load()
    tags = [t for t in ("intra", "inter") if t in data] + [t for t in data if t not in COLORS]
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    rows, body = [], []
    for tag in tags:
        s, t = data[tag]
        alpha, beta = fit(s, t)
        peak = (s / t / 1e9).max()
        big = (s / t / 1e9)[-1]
        rows.append(dict(tag=tag, alpha_us=alpha * 1e6, beta_ns_per_byte=beta * 1e9,
                         beta_ns_per_word8=beta * 8e9, bw_fit_GBs=1 / beta / 1e9,
                         bw_peak_GBs=peak, bw_largest_GBs=big))
        body.append([LABELS.get(tag, tag), f"{alpha * 1e6:.2f}", f"{beta * 1e9:.3f}",
                     f"{beta * 8e9:.2f}", f"{1 / beta / 1e9:.3g}", f"{peak:.3g}"])
    with open(RES / "pingpong_fit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Comunicación & $\alpha$ [$\mu$s] & $\beta$ [ns/B] & $\beta$ [ns/palabra] & "
             r"BW ajustado [GB/s] & BW máx. [GB/s] \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (RES / "pingpong_fit.tex").write_text("\n".join(lines))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for tag in tags:
        s, t = data[tag]
        alpha, beta = fit(s, t)
        c = COLORS.get(tag, INK)
        lab = LABELS.get(tag, tag)
        a1.loglog(s, t * 1e6, "o-", color=c, ms=4, lw=1.6, label=lab, zorder=3)
        xs = np.geomspace(s[0], s[-1], 100)
        a1.loglog(xs, (alpha + beta * xs) * 1e6, "--", color=c, lw=1.1, alpha=0.8,
                  label=f"{lab}: $\\alpha+\\beta s$ ($\\alpha$={alpha * 1e6:.2f} µs, "
                        f"$\\beta$={beta * 1e9:.3f} ns/B)")
        a2.plot(s, s / t / 1e9, "o-", color=c, ms=4, lw=1.6, label=lab, zorder=3)
    for b, name in BLOCKS:
        a2.axvline(b * b * 8, color=INK, lw=0.9, ls=":", zorder=1)
        a2.text(b * b * 8, 0.02, f" {name}", rotation=90, va="bottom", ha="right",
                fontsize=9, color=INK, transform=a2.get_xaxis_transform())
    a1.set_xlabel("Tamaño del mensaje [bytes]", fontsize=12)
    a1.set_ylabel("Tiempo de ida [µs]", fontsize=12)
    a2.set_xlabel("Tamaño del mensaje [bytes]", fontsize=12)
    a2.set_ylabel("Ancho de banda [GB/s]", fontsize=12)
    a2.set_xscale("log")
    a2.set_yscale("log")             # escala log: intra (~GB/s) e inter (~0.1 GB/s) se ven en el mismo grafico
    a2.set_ylim(1e-4, 30)
    for ax in (a1, a2):
        style(ax)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    a2.legend(frameon=False, fontsize=10, loc="lower right")
    a1.set_title("Tiempo de ida", fontsize=12, loc="left")
    a2.set_title("Ancho de banda (líneas punteadas: bloque $b\\times b$ de doubles)",
                 fontsize=12, loc="left")
    fig.suptitle("Ping-pong entre 2 procesos MPI", fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"pingpong.{ext}", dpi=200, bbox_inches="tight")

    print(f"{'tag':>6} {'alpha[us]':>10} {'beta[ns/B]':>11} {'BW fit[GB/s]':>13} {'BW max[GB/s]':>13}")
    for r in rows:
        print(f"{r['tag']:>6} {r['alpha_us']:10.2f} {r['beta_ns_per_byte']:11.3f} "
              f"{r['bw_fit_GBs']:13.3g} {r['bw_peak_GBs']:13.3g}")
    print("Escrito: results/pingpong_fit.{csv,tex}, figures/pingpong.{png,pdf}")


if __name__ == "__main__":
    main()