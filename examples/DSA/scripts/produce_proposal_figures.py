"""Two proposal-quality figures focused on MeerKLASS vs stop-and-stare.

1. azimuth_vs_stare_crosscorr.{png,pdf}
   Two scenarios overlaid on the same axes:
     - MeerKLASS baseline (single elevation, n_repeats=13)
     - Stop-and-stare (9-pointing hex grid)
   Top: Cℓ^{rec × truth} for each scenario, with GDSM truth as
        a wide grey band.
   Bottom: correlation coefficient rℓ = Cℓ^cross/√(Cℓ^rec Cℓ^truth).
   Each scenario uses its OWN sensitivity mask.

2. azimuth_true_rec_residual.{png,pdf}
   The three panels for MeerKLASS baseline ns=64 HP: GDSM truth,
   recovered map, residual = recovered − truth. (Essentially a
   re-export of the existing compare_focus_meerklass_baseline__
   autonoise_ns64_hp.png with tighter labels/title for the proposal.)
"""

from __future__ import annotations

import os
import sys

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DSA_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, DSA_DIR)

from compare_maps import (  # noqa: E402
    DEFAULT_WHITE_VAR, load_ops, load_tods, run_mapmaking,
)
from compare_power_spectra import (  # noqa: E402
    bin_cls, embed_patch, masked_cls, masked_cross_cls,
    DSA_BEAM_FWHM_DEG, FREQ_MHZ,
)
from dsa_vis import plot_patch  # noqa: E402
from limTOD import GDSM_sky_model  # noqa: E402

PRIOR_KWARGS = dict(
    no_prior=False,
    prior_mean_mode="smoothed",
    prior_sigma_factor=1.0,
    auto_noise=True,
)

# Two scenarios we want to overlay
CASES = [
    # (kind, tod-suffix, label, colour)
    ("meerklass",       "_baseline", "Azimuth-scan",   "tab:blue"),
    ("steer_and_stare", "_baseline", "Stop-and-stare", "tab:green"),
]


# ---------------------------------------------------------------------------
# Figure 1 — cross-correlation comparison
# ---------------------------------------------------------------------------
def figure_crosscorr(nside: int, hp_cutoff: float = 3e-2) -> None:
    lmax = 3 * nside - 1
    sky_truth_full = GDSM_sky_model(freq=FREQ_MHZ, nside=nside)
    ell = np.arange(lmax + 1)
    n_bins = max(4, min(12, int(np.log10(lmax + 1) * 6)))
    bin_edges = np.unique(
        np.logspace(np.log10(2), np.log10(lmax + 1), n_bins).astype(int))
    bin_edges = bin_edges[bin_edges <= lmax + 1]

    results = []
    for kind, suffix, label, colour in CASES:
        TOD_group = load_tods(kind, suffix=suffix)
        mm = load_ops(kind, nside, suffix=suffix)
        sky_est, _, rms = run_mapmaking(
            mm, TOD_group, use_hp_filter=True, hp_cutoff=hp_cutoff,
            white_var=DEFAULT_WHITE_VAR, **PRIOR_KWARGS,
        )
        mask = np.zeros(hp.nside2npix(nside), dtype=np.float64)
        mask[mm.pixel_indices] = 1.0
        sky_est_full = embed_patch(sky_est, mm.pixel_indices, nside)
        cl_truth = masked_cls(sky_truth_full, mask, lmax)
        cl_rec = masked_cls(sky_est_full, mask, lmax)
        cl_cross = masked_cross_cls(sky_est_full, sky_truth_full, mask, lmax)
        lc, cl_truth_b = bin_cls(ell, cl_truth, bin_edges)
        _, cl_rec_b = bin_cls(ell, cl_rec, bin_edges)
        _, cl_cross_b = bin_cls(ell, cl_cross, bin_edges)
        denom = np.sqrt(np.abs(cl_rec_b) * np.abs(cl_truth_b))
        r_ell = cl_cross_b / np.maximum(denom, 1e-30)
        results.append({
            "label": label, "colour": colour,
            "lc": lc, "cl_truth": cl_truth_b, "cl_cross": cl_cross_b,
            "r_ell": r_ell, "rms": float(rms),
        })
        print(f"[prop] ns={nside} {label}: RMS={rms:.3f} K, "
              f"f_sky={mask.mean():.4f}")

    ell_beam = 180.0 / DSA_BEAM_FWHM_DEG
    fig, (ax_cl, ax_r) = plt.subplots(
        2, 1, figsize=(7.0, 6.6), constrained_layout=True, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    # Top — each scenario gets its OWN truth band (different mask/f_sky
    # → different Cℓ^truth). Truth as a thick semi-transparent line in
    # the scenario's colour; cross-spectrum as a sharp solid line with
    # markers in the same colour.
    for r in results:
        ax_cl.loglog(r["lc"], r["cl_truth"], color=r["colour"],
                     lw=8, alpha=0.22, solid_capstyle="round", zorder=1,
                     label=fr"$C_\ell^{{\rm truth}}$ — {r['label']}")
        ax_cl.loglog(r["lc"], np.abs(r["cl_cross"]), color=r["colour"],
                     marker="o", ms=6, mec="white", mew=0.8, lw=2,
                     label=fr"$|C_\ell^{{\rm rec\times truth}}|$ — {r['label']}",
                     zorder=3)
    if ell_beam < lmax + 1:
        ax_cl.axvline(ell_beam, color="0.35", ls="--", lw=1.2,
                      label=r"$\ell_{\rm beam}\approx %.0f$" % ell_beam)
    ax_cl.set_ylabel(r"$C_\ell$  [K$^2$]")
    ax_cl.legend(loc="lower left", frameon=True, framealpha=0.9,
                 fancybox=False, edgecolor="0.7", fontsize=10)

    # Bottom — correlation coefficient
    for r in results:
        ax_r.semilogx(r["lc"], r["r_ell"], color=r["colour"],
                      marker="o", ms=6, mec="white", mew=0.8, lw=2,
                      label=r["label"], zorder=3)
    ax_r.axhline(1.0, color="k", ls="-", lw=1.0, alpha=0.6)
    ax_r.axhline(0.0, color="k", ls=":", lw=0.8, alpha=0.4)
    if ell_beam < lmax + 1:
        ax_r.axvline(ell_beam, color="0.35", ls="--", lw=1.2)

    # Auto-zoom to the data range so small differences are visible.
    r_min = float(np.nanmin([np.min(r["r_ell"]) for r in results]))
    if r_min > 0.4:
        ax_r.set_ylim(max(-0.05, r_min - 0.1), 1.05)
    else:
        ax_r.set_ylim(-0.15, 1.1)
    ax_r.set_xlim(results[0]["lc"][0] * 0.9, results[0]["lc"][-1] * 1.1)
    ax_r.set_xlabel(r"multipole $\ell$")
    ax_r.set_ylabel(r"$r_\ell$")
    ax_r.legend(loc="lower left", frameon=True, framealpha=0.9,
                fancybox=False, edgecolor="0.7", fontsize=10)

    suffix = "" if nside == 64 else f"_ns{nside}"
    out_base = os.path.join(DSA_DIR, "figures",
                            f"azimuth_vs_stare_crosscorr{suffix}")
    fig.savefig(out_base + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[prop] wrote {out_base}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 2 — MeerKLASS baseline: truth / recovered / residual
# ---------------------------------------------------------------------------
def figure_meerklass_maps(nside: int, hp_cutoff: float = 3e-2) -> None:
    TOD_group = load_tods("meerklass", suffix="_baseline")
    mm = load_ops("meerklass", nside, suffix="_baseline")
    sky_est, sky_truth, rms = run_mapmaking(
        mm, TOD_group, use_hp_filter=True, hp_cutoff=hp_cutoff,
        white_var=DEFAULT_WHITE_VAR, **PRIOR_KWARGS,
    )
    residual = sky_est - sky_truth
    lo, hi = np.percentile(sky_truth, [2, 98])
    bmag = max(float(np.percentile(np.abs(residual), 98)), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    plot_patch(
        map_vec=sky_truth, nside=nside, pixel_indices=mm.pixel_indices,
        ax=axes[0], title="",
        unit="K", cmap="inferno", vmin=lo, vmax=hi,
    )
    plot_patch(
        map_vec=sky_est, nside=nside, pixel_indices=mm.pixel_indices,
        ax=axes[1], title="",
        unit="K", cmap="inferno", vmin=lo, vmax=hi,
    )
    plot_patch(
        map_vec=residual, nside=nside, pixel_indices=mm.pixel_indices,
        ax=axes[2], title="",
        unit="K", cmap="RdBu_r", vmin=-bmag, vmax=+bmag,
    )
    print(f"[prop] ns={nside} azimuth-scan recovered RMS = {rms*1e3:.1f} mK")
    suffix = "" if nside == 64 else f"_ns{nside}"
    out_base = os.path.join(DSA_DIR, "figures",
                            f"azimuth_true_rec_residual{suffix}")
    fig.savefig(out_base + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[prop] wrote {out_base}.{{png,pdf}}  RMS={rms*1e3:.1f} mK")


def main() -> None:
    import argparse

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=64,
                        help="HEALPix nside for the recovered maps and "
                             "cross-correlation (default: 64). The map-maker "
                             "operator must already be cached at this nside.")
    parser.add_argument("--hp-cutoff", type=float, default=3e-2,
                        help="High-pass filter cutoff in Hz (default: 0.03). "
                             "Lower values let more large-scale signal through "
                             "at the cost of more 1/f leakage.")
    args = parser.parse_args()
    figure_crosscorr(args.nside, hp_cutoff=args.hp_cutoff)
    figure_meerklass_maps(args.nside, hp_cutoff=args.hp_cutoff)


if __name__ == "__main__":
    main()
