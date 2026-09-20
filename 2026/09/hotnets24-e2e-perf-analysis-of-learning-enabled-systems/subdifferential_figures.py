#!/usr/bin/env python3
"""Figures explaining piecewise sub-differentiability and subgradients.

Companion to Namyar et al., "End-to-End Performance Analysis of Learning-enabled
Systems" (HotNets '24), section 4 and footnote 1, which rests on the claim that
DNNs are *piecewise sub-differentiable*. Three figures, each answering one part
of that phrase:

  1. kink        - where differentiability fails, and what replaces it there
  2. pieces      - the same structure at network scale, in 1-D and 2-D
  3. ascent      - what the resulting gradient is actually used for

Run it:

    python3 subdifferential_figures.py                 # -> figures/*.png
    python3 subdifferential_figures.py --outdir /tmp --format pdf
    python3 subdifferential_figures.py --only kink

Each figure prints its caption, so the output doubles as a reading order.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Palette: slots 1/2/3 of the validated categorical order, which clears the
# all-pairs colorblind gate at three series. Never more than three per panel.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", BLUE])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.labelcolor": INK2, "text.color": INK,
    "axes.edgecolor": INK3, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.8, "lines.linewidth": 2.0,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.autolayout": False,
})


def style(ax, title=None, xlabel=None, ylabel=None):
    """Recessive frame: two spines, ink-secondary labels, grid behind marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


# ---------------------------------------------------------------------------
# 1. the kink: no derivative, but a whole interval of valid slopes
# ---------------------------------------------------------------------------
def fig_kink():
    z = np.linspace(-2, 2, 400)
    relu = np.maximum(0, z)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # (a) where it breaks
    ax = style(axes[0], "a.  ReLU is smooth everywhere but one point",
               "$z$", r"$\sigma(z)=\max(0,z)$")
    ax.plot(z, relu, color=BLUE)
    ax.plot([0], [0], "o", ms=8, color=SURFACE, mec=ORANGE, mew=2, zorder=5)
    ax.annotate("no tangent here:\nslope 0 on the left, 1 on the right",
                xy=(0, 0), xytext=(-1.92, 1.15), color=INK2, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=INK3, lw=1))
    ax.set_ylim(-0.35, 2.05)

    # (b) the set of slopes that still "works"
    ax = style(axes[1], "b.  Every slope in [0, 1] supports the graph",
               "$z$", None)
    # a line of slope g touches at 0 and stays below sigma iff 0 <= g <= 1;
    # the wedge those lines sweep is everything between y=0 and y=z
    zf = np.linspace(-1.35, 2, 200)          # trimmed so the wedge fits the frame
    ax.fill_between(zf, 0, zf, color=AQUA, alpha=0.13, lw=0)
    for g in np.linspace(0, 1, 6):
        ax.plot(zf, g * zf, color=AQUA, lw=1.1, alpha=0.75)
    ax.plot(z, relu, color=BLUE, zorder=4)
    ax.plot([0], [0], "o", ms=8, color=SURFACE, mec=ORANGE, mew=2, zorder=5)
    ax.text(-1.95, 1.45, r"$\partial\sigma(0)=[0,1]$", color=INK, fontsize=11)
    ax.text(-1.95, 0.95, "every line stays under the curve\nand touches it at $z=0$",
            color=INK2, fontsize=8.5)
    ax.annotate("slope 1", xy=(-1.3, -1.3), xytext=(-1.05, -1.28),
                color=INK2, fontsize=8)
    ax.annotate("slope 0", xy=(-1.3, 0), xytext=(-1.05, -0.28),
                color=INK2, fontsize=8)
    ax.set_ylim(-1.5, 2.05)

    # (c) the derivative jumps; the subdifferential bridges the jump
    ax = style(axes[2], "c.  The subdifferential fills the gap",
               "$z$", "slope")
    neg, pos = z[z < 0], z[z > 0]
    ax.plot(neg, np.zeros_like(neg), color=BLUE, label=r"$\sigma'(z)$")
    ax.plot(pos, np.ones_like(pos), color=BLUE)
    for y in (0, 1):
        ax.plot([0], [y], "o", ms=6, color=SURFACE, mec=BLUE, mew=1.8, zorder=5)
    ax.plot([0, 0], [0, 1], color=ORANGE, lw=3, solid_capstyle="round",
            label=r"$\partial\sigma(0)$", zorder=4)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(-0.35, 1.5)

    fig.tight_layout()
    return fig, (
        "1. THE KINK. ReLU has no derivative at z=0, but the set of slopes of "
        "lines that touch the graph there and stay below it is the whole "
        "interval [0,1]. Each is a valid subgradient; autodiff picks one "
        "(PyTorch returns 0) and carries on."
    )


# ---------------------------------------------------------------------------
# 2. the pieces: same structure, network scale
# ---------------------------------------------------------------------------
def relu_net_1d(x, rng, units=6):
    """A 1-hidden-layer ReLU net: piecewise linear, one kink per unit.

    The kinks are placed across the interval rather than drawn from the weights
    directly - random weights bunch them together and hide the structure the
    figure is about.
    """
    kinks = np.linspace(-2.3, 2.3, units) + rng.uniform(-0.2, 0.2, units)
    a = rng.choice([-1.0, 1.0], units) * rng.uniform(0.8, 2.0, units)
    b = -a * kinks                       # unit j switches exactly at kinks[j]
    w = rng.choice([-1.0, 1.0], units) * rng.uniform(1.2, 2.6, units)
    y = np.maximum(0, np.outer(x, a) + b) @ w
    return y, np.sort(kinks)


def fig_pieces(seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, 1200)
    y, breaks = relu_net_1d(x, rng)
    dy = np.gradient(y, x)

    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[2, 1],
                          hspace=0.12, wspace=0.28)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    # (a) the function: affine between kinks
    style(ax1, "a.  A ReLU network in 1-D is affine on each piece", None, "$H(x)$")
    for i, (lo, hi) in enumerate(zip(np.r_[x.min(), breaks],
                                     np.r_[breaks, x.max()])):
        if i % 2 == 0:
            ax1.axvspan(lo, hi, color=INK3, alpha=0.055, lw=0)
    ax1.plot(x, y, color=BLUE)
    ax1.plot(breaks, np.interp(breaks, x, y), "o", ms=5, color=SURFACE,
             mec=ORANGE, mew=1.8, zorder=5)
    ax1.tick_params(labelbottom=False)
    ax1.text(0.5, 0.92, f"{len(breaks)} kinks, one per unit -- "
                        "straight lines in between",
             transform=ax1.transAxes, color=INK2, fontsize=8.5,
             ha="center", va="top",
             bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=2))

    # (b) its gradient: constant on each piece, jumping at the kinks
    style(ax2, None, "$x$", r"$\nabla H$")
    ax2.plot(x, dy, color=BLUE)
    for bkt in breaks:
        ax2.axvline(bkt, color=ORANGE, lw=1, alpha=0.45)
    ax2.set_yticks([])
    ax2.text(0.5, 0.97, "constant on each piece, jumping only at the kinks",
             transform=ax2.transAxes, color=INK2, fontsize=8.5, ha="center",
             va="top", bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=2))

    # (c) in 2-D the pieces are convex polyhedra
    style(ax3, "b.  In 2-D the pieces are convex polyhedra", "$x_1$", "$x_2$")
    rng2 = np.random.default_rng(seed + 1)
    A, B = rng2.uniform(-1.6, 1.6, (7, 2)), rng2.uniform(-1.1, 1.1, 7)
    g = np.linspace(-2, 2, 500)
    X1, X2 = np.meshgrid(g, g)
    active = sum((A[i, 0] * X1 + A[i, 1] * X2 + B[i] > 0).astype(int)
                 for i in range(len(A)))
    ax3.imshow(active, extent=(-2, 2, -2, 2), origin="lower", cmap=SEQ,
               interpolation="nearest", alpha=0.85)
    for i in range(len(A)):
        if abs(A[i, 1]) > 1e-6:
            ax3.plot(g, -(A[i, 0] * g + B[i]) / A[i, 1], color=INK2, lw=1)
    ax3.set_xlim(-2, 2); ax3.set_ylim(-2, 2)
    ax3.grid(False)
    ax3.text(0.03, 0.03, "shade = units active;  each cell one affine map",
             transform=ax3.transAxes, color=INK2, fontsize=8.5,
             bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=2.5))

    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.14)
    return fig, (
        "2. THE PIECES. Each hidden unit's on/off boundary is a hyperplane; "
        "together they cut the input space into convex polyhedra. Inside one, "
        "the whole network is a single affine map, so the gradient is exact and "
        "constant. The kinks are only the boundaries -- a set of measure zero."
    )


# ---------------------------------------------------------------------------
# 3. the ascent: what the paper actually does with that gradient
# ---------------------------------------------------------------------------
BREAKS = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
SLOPES = np.array([0.8, 1.0, -1.2, 0.3, 1.5, -2.0])


def pw(x):
    """A piecewise-linear objective with a local and a global maximum."""
    knots = np.concatenate([[0.0], np.cumsum(SLOPES * np.diff(BREAKS))])
    return np.interp(x, BREAKS, knots)


def pw_grad(x):
    """Slope of the piece x sits in; at a knot, take the left piece -- the
    arbitrary-but-valid choice autodiff makes at a kink."""
    return SLOPES[np.clip(np.searchsorted(BREAKS, x, side="left") - 1,
                          0, len(SLOPES) - 1)]


def ascend(x0, alpha=0.5, steps=14):
    xs = [x0]
    for _ in range(steps):
        xs.append(np.clip(xs[-1] + alpha * pw_grad(xs[-1]), -3, 3))
    return np.array(xs)


def fig_ascent():
    x = np.linspace(-3, 3, 1000)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    style(ax, r"c.  Subgradient ascent:  $x^{(i+1)} \leftarrow x^{(i)} + "
              r"\alpha\,\nabla_{\!x}\,\mathcal{M}_{adv}(H(x^{(i)}))$",
          "$x$  (the input being searched for)", r"$\mathcal{M}_{adv}(H(x))$")
    ax.plot(x, pw(x), color=INK3, lw=2.4, zorder=1)

    for x0, color, label in ((-2.6, ORANGE, "start A  $\\rightarrow$ local optimum"),
                             (0.4, BLUE, "start B  $\\rightarrow$ global optimum")):
        xs = ascend(x0)
        # the path is a hint at ordering; the objective stays the loudest mark
        ax.plot(xs, pw(xs), "-", color=color, lw=0.9, alpha=0.45, zorder=2)
        ax.plot(xs, pw(xs), "o", color=color, ms=5, mec=SURFACE, mew=0.8,
                label=label, zorder=3)
        ax.plot([x0], [pw(x0)], "o", ms=10, color=SURFACE, mec=color, mew=2.2,
                zorder=4)

    ax.annotate("iterates jitter at the peak:\na kink has no zero gradient",
                xy=(2.0, pw(2.0)), xytext=(0.15, 3.0), color=INK2, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=INK3, lw=1))
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_ylim(-0.6, 3.6)
    fig.tight_layout()
    return fig, (
        "3. THE ASCENT. This is Eq. (1). The gradient exists piece by piece, so "
        "the search always has a direction to move in -- but it only ever finds "
        "a LOCAL optimum unless the objective is convex, and at a maximum that "
        "sits on a kink the gradient never vanishes, so a constant step size "
        "oscillates instead of settling."
    )


FIGURES = {"kink": fig_kink, "pieces": fig_pieces, "ascent": fig_ascent}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--only", choices=sorted(FIGURES), help="render one figure")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    names = [args.only] if args.only else list(FIGURES)
    for name in names:
        fig, caption = FIGURES[name]()
        path = out / f"{name}.{args.format}"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"{path}\n   {caption}\n")


if __name__ == "__main__":
    main()
