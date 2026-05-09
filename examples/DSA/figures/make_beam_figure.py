"""Proposal-quality figure of the DSA zenith beam.

Two panels of equal visual size:
  (a) Log-scale 2D beam (gnomonic projection around the peak), zoomed to
      a few FWHM so the main lobe and first sidelobes are clear.
  (b) Radial profile (azimuthal mean) with FWHM marker.

The beam is peak-normalised (max = 1) so the colour scale reads off as
dB below peak: log10(1) = 0 means "peak", log10(1e-3) = -3 means
"30 dB below peak".
"""

from __future__ import annotations

import os

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
BEAM_FITS = os.path.join(HERE, "..", "beam_map_zenith.fits")

# ---------- Publication styling (match make_figures.py) ----------
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def _gnom_projection(
    beam_map: np.ndarray, nside: int, extent_deg: float, n_pix: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project beam around its peak onto an (n_pix x n_pix) gnomonic grid of
    angular half-width ``extent_deg / 2`` on each axis.

    Robust against peaks near the north pole: builds the tangent-plane
    direction vectors in a "peak = +z" frame, rotates them to world
    coordinates, then uses ``hp.vec2pix`` (no sin(theta) singularity).
    """
    peak_pix = int(np.argmax(beam_map))
    vec_peak = np.asarray(hp.pix2vec(nside, peak_pix))  # unit vector to peak

    # Tangent-plane grid (radians) in the "peak = +z" frame
    half_rad = np.radians(extent_deg / 2.0)
    g = np.linspace(-half_rad, half_rad, n_pix)
    X, Y = np.meshgrid(g, g)  # (x-east, y-north) tangent-plane coordinates
    Z = np.sqrt(np.maximum(1.0 - X * X - Y * Y, 0.0))
    local = np.stack([X, Y, Z], axis=-1)  # shape (n_pix, n_pix, 3)

    # Rotation that maps +z -> vec_peak.  Use Rodrigues' formula.
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, vec_peak)
    s = np.linalg.norm(axis)
    c = float(np.dot(z, vec_peak))
    if s < 1e-12:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        k = axis / s
        K = np.array([
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ])
        R = np.eye(3) + s * K + (1.0 - c) * (K @ K)

    world = local @ R.T  # same shape as local, rotated into world frame
    pix = hp.vec2pix(nside, world[..., 0], world[..., 1], world[..., 2])
    return beam_map[pix], np.degrees(g)


def _radial_profile(
    beam_map: np.ndarray, nside: int, max_deg: float, n_bin: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Radial profile of a peak-centred beam.

    Returns ``(angle_deg, median, p16, p84)`` — the median is robust to
    outliers and the (p16, p84) band gives a sense of the beam's azimuthal
    asymmetry without the bin-to-bin spikes that an azimuthal-max trace
    introduces.
    """
    peak_pix = int(np.argmax(beam_map))
    peak_vec = hp.pix2vec(nside, peak_pix)

    pix_vec = np.asarray(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    cos_ang = np.clip(np.einsum("i,ij->j", peak_vec, pix_vec), -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(cos_ang))

    edges = np.linspace(0.0, max_deg, n_bin + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    median = np.full(n_bin, np.nan)
    p16 = np.full(n_bin, np.nan)
    p84 = np.full(n_bin, np.nan)
    for i in range(n_bin):
        mask = (ang_deg >= edges[i]) & (ang_deg < edges[i + 1])
        if mask.sum() >= 3:  # need a few pixels for a meaningful percentile
            vals = beam_map[mask]
            median[i] = np.median(vals)
            p16[i], p84[i] = np.percentile(vals, [16, 84])
    return centres, median, p16, p84


# ---------- Load and peak-normalise ----------
beam = hp.read_map(BEAM_FITS)
nside = hp.get_nside(beam)
beam = beam / beam.max()  # peak = 1 for dB-like reading of log colour scale

# Estimate FWHM from a finely-binned median profile (robust)
centres_init, median_init, _, _ = _radial_profile(
    beam, nside, max_deg=20.0, n_bin=200
)
half_idx = np.where(median_init < 0.5)[0]
fwhm_half_rad = centres_init[half_idx[0]] if len(half_idx) else np.nan
fwhm_deg = 2.0 * fwhm_half_rad

# Beam solid angle (in deg^2) at peak-normalised scale
solid_rad = beam.sum() * hp.nside2pixarea(nside)
solid_deg2 = solid_rad * (180.0 / np.pi) ** 2

# ---------- Compute panels ----------
EXTENT_DEG = 6.0 * fwhm_deg  # show ±3×FWHM around the peak
N_PIX = 601
img, grid = _gnom_projection(beam, nside, EXTENT_DEG, N_PIX)

# Coarser radial bins (~0.3°) so each bin averages many HEALPix pixels.
# At nside=128 the pixel size is ~0.46°; bin width >= 0.3° smooths the
# profile while still resolving the FWHM and the first sidelobe ring.
n_radial_bin = max(20, int(np.ceil(3.0 * fwhm_deg / 0.3)))
centres, profile_med, profile_p16, profile_p84 = _radial_profile(
    beam, nside, max_deg=3.0 * fwhm_deg, n_bin=n_radial_bin
)

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.3))

# --- Panel (a): log-scale 2D beam ---
im = ax1.imshow(
    np.log10(np.maximum(img, 1e-6)),
    extent=[grid[0], grid[-1], grid[0], grid[-1]],
    origin="lower",
    cmap="viridis",
    vmin=-4,
    vmax=0,
)
ax1.set_xlabel(r"$\Delta\phi\,\sin\theta$  [deg]")
ax1.set_ylabel(r"$\Delta\theta$  [deg]")
ax1.set_title(r"Zenith beam  ($\log_{10}$, peak-normalised)")
ax1.set_aspect("equal")

# FWHM ring overlay
theta_ring = np.linspace(0, 2 * np.pi, 360)
ax1.plot(
    0.5 * fwhm_deg * np.cos(theta_ring), 0.5 * fwhm_deg * np.sin(theta_ring),
    color="white", lw=1.0, ls="--", alpha=0.7,
)
ax1.plot(0, 0, "+", color="white", ms=8, mew=1.2)

cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
cbar.set_label(r"$\log_{10}(B / B_{\rm peak})$")

# --- Panel (b): radial profile (robust median + 16-84% asymmetry band) ---
# Mask any all-NaN bins (innermost rings with <3 pixels) for clean plotting
ok = np.isfinite(profile_med)
ax2.fill_between(
    centres[ok], profile_p16[ok], profile_p84[ok],
    color="#1f77b4", alpha=0.20,
    label="16–84% azimuthal spread",
)
ax2.plot(
    centres[ok], profile_med[ok], color="#1f77b4", lw=1.8,
    label="azimuthal median",
)
ax2.axvline(0.5 * fwhm_deg, color="0.4", ls="--", lw=0.8,
            label=rf"FWHM/2 $= {0.5 * fwhm_deg:.2f}^\circ$")
ax2.axhline(0.5, color="0.7", ls=":", lw=0.8)

ax2.set_yscale("log")
ax2.set_xlim(0, 3.0 * fwhm_deg)
ax2.set_ylim(1e-5, 1.5)
ax2.set_xlabel("Angular distance from peak [deg]")
ax2.set_ylabel(r"$B / B_{\rm peak}$")
ax2.set_title("Radial profile")
ax2.grid(True, which="both", alpha=0.25, lw=0.5)
ax2.legend(loc="upper right", framealpha=0.9)
ax2.set_box_aspect(1.0)  # same square footprint as the image panel

fig.suptitle(
    rf"DSA zenith beam  •  HEALPix nside = {nside},  "
    rf"FWHM $= {fwhm_deg:.2f}^\circ$,  "
    rf"$\Omega_{{\rm beam}} = {solid_deg2:.1f}$ deg$^2$",
    y=1.03,
    fontsize=12,
)

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(HERE, "dsa_beam.pdf"))
plt.savefig(os.path.join(HERE, "dsa_beam.png"), dpi=300)
plt.close()
print(f"Saved dsa_beam.{{pdf,png}}  (FWHM={fwhm_deg:.2f}°, Ω={solid_deg2:.1f} deg²)")
