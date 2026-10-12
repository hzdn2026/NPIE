# -*- coding: utf-8 -*-
"""Render the NPIE architecture diagram as a publication-ready vector PDF + PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT_PDF = "figs/fig_arch.pdf"
OUT_PNG = "figs/fig_arch.png"

SIG = "#2b6cb0"      # signal path (blue)
PHY = "#c05621"      # physical path (orange)
INV = "#2f855a"      # invariance (green)
FUS = "#6b46c1"      # fusion (purple)
GREY = "#718096"
INK = "#1a202c"

fig, ax = plt.subplots(figsize=(7.6, 5.3), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.0)
ax.axis("off")

def box(x, y, w, h, text, fc, ec, tc="white", fs=8.0, lw=1.4, dashed=False):
    r = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=lw, edgecolor=ec, facecolor=fc,
                       linestyle="--" if dashed else "solid", zorder=3)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, weight="bold")

def arrow(x1, y1, x2, y2, color=INK, ls="-", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=12,
                 linewidth=lw, color=color, linestyle=ls,
                 connectionstyle=("arc3,rad=%.2f" % rad) if rad else "arc3,rad=0",
                 zorder=2))

def label(x, y, text, color=INK, fs=6.7, ha="center", weight="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, weight=weight)

label(5.0, 6.72, "NPIE: resolution-preserving complex frontend + physical-impairment estimator",
      color=INK, fs=9.6, weight="bold")

# ===== Top path (center y = 5.975) =====
Y, H = 5.55, 0.85
box(0.25, Y, 1.35, H, "IQ window\n$x[n]$  (2,L)", SIG, SIG, fs=8.0)
box(1.85, Y, 1.30, H, "Phase aug.\n$e^{j\\phi}$ (train)", SIG, SIG, fs=7.5, dashed=True)
box(3.45, Y, 1.55, H, "ComplexConv1d\n(C, no downsample)", SIG, SIG, fs=7.3)
box(5.30, Y, 1.15, H, "$z_b$\n(512)", SIG, SIG, fs=8.3)
box(6.75, Y, 1.25, H, "ParamNet\n(regress)", PHY, PHY, fs=7.9)
box(8.30, Y, 1.45, H, "$\\theta_{Tx} \\in \\mathbb{R}^{32}$\nimpairment est.", PHY, PHY, fs=7.7)
arrow(1.60, 5.975, 1.85, 5.975, SIG)
arrow(3.15, 5.975, 3.45, 5.975, SIG)
arrow(5.00, 5.975, 5.30, 5.975, SIG)
arrow(6.45, 5.975, 6.75, 5.975, SIG)
arrow(8.00, 5.975, 8.30, 5.975, PHY)
label(1.50, 5.28, "rotation-invariant prior", color=SIG, fs=6.5)
label(7.37, 5.28, "recon.+ortho keep $\\theta_{Tx}$ Tx-only", color=PHY, fs=6.5)

# ===== Invariance stack =====
box(8.30, 4.05, 1.45, 0.70, "SupCon + GRL\ninvariance", INV, INV, fs=7.3)
box(8.65, 3.05, 1.00, 0.62, "Rx / day\ndiscrim.", GREY, GREY, fs=6.5)
box(6.55, 3.05, 1.05, 0.62, "same-Tx\npull / push", GREY, GREY, fs=6.5)
arrow(9.02, 5.55, 9.02, 4.75, PHY)
arrow(9.02, 4.05, 9.02, 3.67, INV)
arrow(8.30, 4.40, 7.35, 3.67, GREY, ls="--", rad=-0.2)
label(9.30, 2.78, "fool capture condition", color="#4a5568", fs=6.3)

# ===== Bottom signal-energy branch (center y = 1.75) =====
Y2, H2 = 1.35, 0.80
arrow(1.40, 5.55, 2.10, 2.15, SIG, ls="--", rad=0.15)
label(0.82, 3.60, "fork\nfrom $x[n]$", color="#4a5568", fs=6.4)
box(1.85, Y2, 1.25, H2, "Flatten\nIQ", FUS, FUS, fs=7.8)
box(3.45, Y2, 1.50, H2, "MLP head\nsignal-energy", FUS, FUS, fs=7.3)
box(5.30, Y2, 1.35, H2, "$f_{energy}$\n(512)", FUS, FUS, fs=7.9)
arrow(3.10, 1.75, 3.45, 1.75, FUS)
arrow(4.95, 1.75, 5.30, 1.75, FUS)

# ===== Fusion -> classifier -> output =====
box(6.95, Y2, 1.35, H2, "concat\n$\\theta_{Tx} \\oplus f_{energy}$", FUS, FUS, fs=6.9)
box(8.50, Y2, 1.40, H2, "Classifier\n(softmax)", FUS, FUS, fs=7.7)
box(9.02, 0.25, 0.88, 0.62, "identity\n$\\hat{y} = y$", "#2d3748", "#2d3748", fs=7.1)
arrow(6.65, 1.75, 6.95, 1.75, FUS)                                # f_energy -> concat
arrow(8.55, 5.55, 7.62, 2.15, PHY, ls="--", rad=-0.30, lw=1.2)    # theta_Tx -> concat
arrow(8.30, 1.75, 8.50, 1.75, FUS)                                # concat -> classifier
arrow(9.20, 1.35, 9.20, 0.87, FUS)                                 # classifier -> output
label(7.60, 1.00, "constructive invariance", color=FUS, fs=6.3)

# ===== Legend =====
legend_elems = [
    Line2D([0],[0], color=SIG, lw=2.4, label="signal / complex path"),
    Line2D([0],[0], color=PHY, lw=2.4, label="physical-impairment ($\\theta_{Tx}$) path"),
    Line2D([0],[0], color=INV, lw=2.4, label="invariance objective"),
    Line2D([0],[0], color=FUS, lw=2.4, label="fusion + classifier"),
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=6.8,
          framealpha=0.92, edgecolor="#e2e8f0", ncol=2, borderpad=0.5)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.06)
plt.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.06, dpi=200)
print("wrote", OUT_PDF, "and", OUT_PNG)
