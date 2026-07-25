"""Generate _static/beam-convention.svg (run from anywhere) — the three-panel beam
convention figure (top view / vertical plane / beam chart).

Design constraints learned from the retired conventions.pdf: every
panel must carry its own absolute anchor (no 3D perspective, no
observer-dependent words); the two tangent directions keep one color
each across panels (Okabe-Ito colorblind-safe pair).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

C_EL = "#0072B2"   # blue: e_el (increasing elevation)
C_AZ = "#D55E00"   # vermillion: e_az (increasing azimuth)
GREY = "#666666"
LGREY = "#bbbbbb"

A_DEG, E_DEG = 50.0, 35.0
A, E = np.deg2rad(A_DEG), np.deg2rad(E_DEG)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

# ------------------------------------------------------------------ #
# Panel 1: view from above (looking down): N up, E right             #
# ------------------------------------------------------------------ #
ax = axes[0]
tt = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(tt), np.sin(tt), color="black", lw=1.2)
for ang, lab in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
    r = np.deg2rad(ang)
    x, y = np.sin(r), np.cos(r)  # map view: azimuth ang at (sinA, cosA)
    ax.plot([0.97 * x, 1.03 * x], [0.97 * y, 1.03 * y], color="black", lw=1.2)
    ax.text(1.14 * x, 1.14 * y, lab, ha="center", va="center", fontsize=13,
            fontweight="bold")
ax.plot(0, 0, marker="+", color="black", ms=10, mew=1.5)
ax.text(0.03, -0.13, "antenna site\n(zenith: toward\nthe viewer)", fontsize=8.5,
        color=GREY, ha="left", va="top")

# pointing's horizontal direction at azimuth A
px, py = np.sin(A), np.cos(A)
ax.plot([0, px], [0, py], color="black", lw=1.6, ls="--")
ax.annotate("", xy=(px, py), xytext=(0.88 * px, 0.88 * py),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
ax.text(1.22 * px, 1.22 * py, "azimuth $A$\ndirection", ha="center",
        va="center", fontsize=10)

# azimuth arc from N to A (clockwise on this map view)
arc_t = np.linspace(0, A, 60)
ax.plot(0.45 * np.sin(arc_t), 0.45 * np.cos(arc_t), color=GREY, lw=1.2)
ax.annotate("", xy=(0.45 * np.sin(A), 0.45 * np.cos(A)),
            xytext=(0.45 * np.sin(A - 0.12), 0.45 * np.cos(A - 0.12)),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
ax.text(0.30 * np.sin(A / 2), 0.30 * np.cos(A / 2) + 0.05, "$A$",
        fontsize=12, color=GREY)

# e_az: tangent, direction of increasing azimuth
tx, ty = np.cos(A), -np.sin(A)
ax.annotate("", xy=(px + 0.34 * tx, py + 0.34 * ty), xytext=(px, py),
            arrowprops=dict(arrowstyle="-|>", color=C_AZ, lw=2.4))
ax.text(px + 0.44 * tx + 0.05, py + 0.44 * ty - 0.06,
        r"$\hat e_{\mathrm{az}}$", fontsize=14, color=C_AZ)

ax.set_title("View from above (looking down)\n"
             "azimuth: N = 0° $\\to$ E = 90°", fontsize=10.5)
ax.set_xlim(-1.55, 1.75)
ax.set_ylim(-1.5, 1.62)

# ------------------------------------------------------------------ #
# Panel 2: the vertical plane through zenith and the pointing        #
# ------------------------------------------------------------------ #
ax = axes[1]
tt = np.linspace(0, np.pi / 2, 100)
ax.plot(np.cos(tt), np.sin(tt), color="black", lw=1.2)
ax.plot([-0.18, 1.12], [0, 0], color="black", lw=1.2)          # horizon
ax.plot([0, 0], [0, 1.06], color=LGREY, lw=1.0, ls=":")        # vertical
ax.text(0, 1.13, "zenith", ha="center", fontsize=11, fontweight="bold")
ax.text(1.30, -0.06, "horizon point\nat azimuth $A$", fontsize=9.5,
        va="top", ha="center")

bx, by = np.cos(E), np.sin(E)
ax.plot([0, bx], [0, by], color="black", lw=1.8)
ax.annotate("", xy=(bx, by), xytext=(0.9 * bx, 0.9 * by),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8))
ax.text(0.58 * bx + 0.02, 0.58 * by - 0.09, r"$\hat b$ (boresight)",
        fontsize=11)

arc_t = np.linspace(0, E, 50)
ax.plot(0.38 * np.cos(arc_t), 0.38 * np.sin(arc_t), color=GREY, lw=1.2)
ax.text(0.46 * np.cos(E / 2), 0.46 * np.sin(E / 2) - 0.03, "$e$",
        fontsize=12, color=GREY)

# e_el: tangent at b, toward the zenith side
ex, ey = -np.sin(E), np.cos(E)
ax.annotate("", xy=(bx + 0.34 * ex, by + 0.34 * ey), xytext=(bx, by),
            arrowprops=dict(arrowstyle="-|>", color=C_EL, lw=2.4))
ax.text(bx + 0.40 * ex - 0.03, by + 0.44 * ey + 0.02,
        r"$\hat e_{\mathrm{el}} = \partial\hat b/\partial e$",
        fontsize=12.5, color=C_EL, ha="center")

ax.text(0.28, -0.44,
        r"$\hat e_{\mathrm{az}}$ is horizontal, $\perp$ to this plane;"
        "\nits sense is fixed in the left panel",
        fontsize=9, color=C_AZ, ha="center")
ax.set_title("The vertical plane containing\nthe zenith and the pointing",
             fontsize=10.5)
ax.set_xlim(-0.45, 1.62)
ax.set_ylim(-0.70, 1.35)

# ------------------------------------------------------------------ #
# Panel 3: the beam map (chart around the boresight)                 #
# ------------------------------------------------------------------ #
ax = axes[2]
for r in (0.35, 0.7, 1.05):
    ax.plot(r * np.cos(tt2 := np.linspace(0, 2 * np.pi, 200)),
            r * np.sin(tt2), color=LGREY, lw=0.9, ls="--")
ax.text(0.39, -0.30, r"$\theta$", fontsize=11, color=GREY)

# example asymmetric beam: major axis along phi = 0
ax.add_patch(Ellipse((0, 0), 0.42, 0.9, facecolor="#e8f1f8",
                     edgecolor=C_EL, lw=1.0, alpha=0.9, zorder=0))

ax.annotate("", xy=(0, 1.28), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=C_EL, lw=2.4))
ax.text(0, 1.36, r"$\varphi = 0 \;\to\; \hat e_{\mathrm{el}}$"
        "\n(increasing elevation)", ha="center", fontsize=11, color=C_EL)
ax.annotate("", xy=(1.28, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=C_AZ, lw=2.4))
ax.text(1.34, 0.02, r"$\varphi = 90°$" "\n" r"$\to\; \hat e_{\mathrm{az}}$"
        "\n(increasing\nazimuth)", ha="left", va="center", fontsize=11,
        color=C_AZ)
ax.plot([0, 0], [0, -1.18], color=GREY, lw=1.0, ls=":")
ax.text(0, -1.30, r"$\varphi = 180°$", ha="center", fontsize=9.5, color=GREY)
ax.plot([0, -1.18], [0, 0], color=GREY, lw=1.0, ls=":")
ax.text(-1.24, 0, r"$\varphi = 270°$", ha="right", va="center",
        fontsize=9.5, color=GREY)

# selfrot sense
arc_t = np.linspace(np.pi / 2 - 0.12, 0.30, 40)
ax.plot(0.55 * np.cos(arc_t), 0.55 * np.sin(arc_t), color="black", lw=1.3)
ax.annotate("", xy=(0.55 * np.cos(0.28), 0.55 * np.sin(0.28)),
            xytext=(0.55 * np.cos(0.42), 0.55 * np.sin(0.42)),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))
ax.text(0.62 * np.cos(0.75), 0.62 * np.sin(0.75) + 0.05, r"$+\psi$",
        fontsize=12)

ax.plot(0, 0, marker=".", color="black", ms=6)
ax.text(-0.14, 0.14, "boresight " r"($\theta = 0$)", fontsize=8.5,
        ha="right", va="bottom")
ax.set_title("The beam map, drawn in the\n"
             r"$(\hat e_{\mathrm{el}},\, \hat e_{\mathrm{az}})$ tangent frame",
             fontsize=10.5)
ax.text(0, -1.62,
        "parked ($A=0°$, $e=90°$):  $\\varphi = 0, 90°, 180°, 270°$"
        "  $\\to$  S, E, N, W points",
        ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec=LGREY))
ax.set_xlim(-1.85, 2.15)
ax.set_ylim(-1.85, 1.72)

fig.tight_layout(w_pad=2.0)
from pathlib import Path

out = Path(__file__).resolve().parent / "_static" / "beam-convention.svg"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
