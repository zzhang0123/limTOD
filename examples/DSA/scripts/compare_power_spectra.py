"""Angular power spectra for the three DSA ns=64 HP-filter recoveries.

For each of {baseline meerklass, stop-and-stare, cascade meerklass},
runs the same map-making as produce_focus_maps.py (strong prior +
auto-noise), embeds the recovered patch into a full HEALPix map,
and computes pseudo-Cℓ on the *common* observed patch (intersection
of all three sensitivity masks, leading-order mask-corrected by
dividing by f_sky).

Three quantities per scenario:
  Cℓ^rec   : recovered angular power
  T_ℓ      : transfer function  = Cℓ^rec / Cℓ^truth
  Cℓ^bias  : residual power     = Cℓ[rec − truth]

Output: figures/power_spectra_comparison.{png,pdf,npz}
"""

from __future__ import annotations

import os
import sys
from functools import reduce

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
from limTOD import GDSM_sky_model  # noqa: E402

DSA_BEAM_FWHM_DEG = 4.5
NSIDE = 64
LMAX = 3 * NSIDE - 1  # 191
FREQ_MHZ = 1000.0

CASES = [
    # (kind, tod-suffix, label, colour)
    ("meerklass",       "_baseline", "MeerKLASS baseline (1 el)",  "tab:blue"),
    ("steer_and_stare", "_baseline", "Stop-and-stare",             "tab:green"),
    ("meerklass",       "_cascade",  "MeerKLASS cascade (5 el)",   "tab:orange"),
]

PRIOR_KWARGS = dict(
    no_prior=False,
    prior_mean_mode="smoothed",
    prior_sigma_factor=1.0,
    auto_noise=True,
)


def embed_patch(values: np.ndarray, pixel_indices: np.ndarray, nside: int,
                fill: float = 0.0) -> np.ndarray:
    npix = hp.nside2npix(nside)
    full = np.full(npix, fill, dtype=np.float64)
    full[pixel_indices] = values
    return full


def masked_cls(full_map: np.ndarray, mask: np.ndarray, lmax: int) -> np.ndarray:
    """Pseudo-Cℓ on a masked full-sky map, divided by f_sky.

    Good enough for between-scenario ratios when the mask is identical.
    """
    m = mask.astype(np.float64)
    fsky = m.mean()
    if fsky <= 0.0:
        raise ValueError("Mask is empty.")
    masked = (full_map - (full_map * m).sum() / m.sum() * m) * m
    return hp.anafast(masked, lmax=lmax) / fsky


def bin_cls(ell: np.ndarray, cl: np.ndarray,
            edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin Cℓ in arbitrary ℓ-bins (returns bin centres + mean power)."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    binned = np.empty(len(edges) - 1, dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (ell >= lo) & (ell < hi)
        binned[i] = cl[m].mean() if m.any() else np.nan
    return centres, binned


def _run_and_compute(use_hp_filter: bool, sky_truth_full: np.ndarray):
    """Map-make all 3 scenarios and return results + common mask."""
    results = []
    for kind, suffix, label, colour in CASES:
        TOD_group = load_tods(kind, suffix=suffix)
        mm = load_ops(kind, NSIDE, suffix=suffix)
        sky_est, sky_truth_patch, rms = run_mapmaking(
            mm, TOD_group, use_hp_filter=use_hp_filter,
            white_var=DEFAULT_WHITE_VAR, **PRIOR_KWARGS,
        )
        results.append({
            "kind": kind, "suffix": suffix, "label": label, "colour": colour,
            "sky_est": sky_est, "sky_truth_patch": sky_truth_patch,
            "pixel_indices": np.asarray(mm.pixel_indices),
            "rms_K": float(rms),
        })
        tag = "HP" if use_hp_filter else "noHP"
        print(f"[pk] ({tag}) {label}: {len(sky_est)} pixels, RMS = {rms:.3f} K")
    return results


def main() -> None:
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

    sky_truth_full = GDSM_sky_model(freq=FREQ_MHZ, nside=NSIDE)

    for use_hp_filter in (True, False):
        hp_tag = "hp" if use_hp_filter else "noHP"
        title_suffix = "HP filter on" if use_hp_filter else "no HP filter"
        print(f"\n=== Running with {title_suffix} ===")
        results = _run_and_compute(use_hp_filter, sky_truth_full)
        _plot_and_dump(results, sky_truth_full, hp_tag, title_suffix)


def _plot_and_dump(results, sky_truth_full, hp_tag, title_suffix):

    # --- 2. Common mask = intersection of all three sensitivity patches ---

    # (continues from old main with `results` already computed)
    common_pix = reduce(np.intersect1d, [r["pixel_indices"] for r in results])
    mask = np.zeros(hp.nside2npix(NSIDE), dtype=np.float64)
    mask[common_pix] = 1.0
    f_sky = mask.mean()
    print(f"[pk] common mask: {len(common_pix)} pixels (f_sky = {f_sky:.4f})")

    # --- 3. Cℓ for truth (restricted to common patch) ---
    ell = np.arange(LMAX + 1)
    cl_truth = masked_cls(sky_truth_full, mask, LMAX)

    # --- 4. Cℓ for each scenario ---
    bin_edges = np.unique(np.logspace(np.log10(2), np.log10(LMAX + 1), 12).astype(int))
    bin_edges = bin_edges[bin_edges <= LMAX + 1]
    l_centre_truth, cl_truth_b = bin_cls(ell, cl_truth, bin_edges)

    for r in results:
        sky_est_full = embed_patch(r["sky_est"], r["pixel_indices"], NSIDE)
        cl_rec = masked_cls(sky_est_full, mask, LMAX)
        cl_bias = masked_cls(sky_est_full - sky_truth_full, mask, LMAX)
        _, cl_rec_b = bin_cls(ell, cl_rec, bin_edges)
        _, cl_bias_b = bin_cls(ell, cl_bias, bin_edges)
        r["cl_rec"] = cl_rec
        r["cl_rec_binned"] = cl_rec_b
        r["cl_bias_binned"] = cl_bias_b
        r["transfer_binned"] = cl_rec_b / cl_truth_b

    # --- 5. Plot: 2-panel (Cℓ + transfer function), proposal-quality ---
    ell_beam = 180.0 / DSA_BEAM_FWHM_DEG
    fig, (ax_cl, ax_T) = plt.subplots(
        2, 1, figsize=(8.0, 7.5), constrained_layout=True, sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    subbeam_kw = dict(color="lightgrey", alpha=0.35, zorder=0)
    ax_cl.axvspan(ell_beam, LMAX + 1, **subbeam_kw)
    ax_T.axvspan(ell_beam, LMAX + 1, **subbeam_kw)

    # Top: Cℓ^truth plotted as a wide, semi-transparent "band" so
    # overlapping recovery lines remain visible through it. zorder=1
    # keeps it below the sharp recovery lines.
    ax_cl.loglog(l_centre_truth, cl_truth_b, color="black", lw=8,
                 alpha=0.22, label="GDSM truth", solid_capstyle="round",
                 zorder=1)
    for r in results:
        ax_cl.loglog(l_centre_truth, r["cl_rec_binned"],
                     color=r["colour"], label=r["label"],
                     marker="o", ms=6, mec="white", mew=0.8)
    ax_cl.axvline(ell_beam, color="0.35", ls="--", lw=1.2,
                  label=r"$\ell_{\rm beam}\approx %.0f$" % ell_beam)
    ax_cl.set_ylabel(r"$C_\ell$  [K$^2$]")
    ax_cl.set_title(
        f"Angular power spectra on the common observed patch  ({title_suffix})",
        pad=10,
    )
    ax_cl.legend(loc="lower left", frameon=True, framealpha=0.9,
                 fancybox=False, edgecolor="0.7")
    ax_cl.text(0.98, 0.96, "sub-beam →",
               transform=ax_cl.transAxes, ha="right", va="top",
               color="0.35", fontsize=11, style="italic")

    # Bottom: Transfer function
    for r in results:
        ax_T.semilogx(l_centre_truth, r["transfer_binned"],
                      color=r["colour"], label=r["label"],
                      marker="o", ms=6, mec="white", mew=0.8)
    ax_T.axhline(1.0, color="k", ls="-", lw=1.0, alpha=0.6)
    ax_T.axvline(ell_beam, color="0.35", ls="--", lw=1.2)
    ax_T.set_ylabel(r"transfer  $C_\ell^{\rm rec}\,/\,C_\ell^{\rm truth}$")
    ax_T.set_xlabel(r"multipole $\ell$")
    ax_T.set_ylim(0.0, 2.2)
    ax_T.set_xlim(l_centre_truth[0] * 0.9, l_centre_truth[-1] * 1.1)
    ax_T.legend(loc="lower left", frameon=True, framealpha=0.9,
                fancybox=False, edgecolor="0.7")

    out_base = os.path.join(DSA_DIR, "figures", f"power_spectra_comparison_{hp_tag}")
    fig.savefig(out_base + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)

    # --- 6. Dump binned arrays for reproducibility / further analysis ---
    np.savez(
        out_base + ".npz",
        l_centre=l_centre_truth,
        cl_truth=cl_truth_b,
        **{f"cl_rec_{r['kind']}{r['suffix']}": r["cl_rec_binned"] for r in results},
        **{f"transfer_{r['kind']}{r['suffix']}": r["transfer_binned"] for r in results},
        **{f"cl_bias_{r['kind']}{r['suffix']}": r["cl_bias_binned"] for r in results},
        ell_beam=ell_beam, f_sky=f_sky,
    )
    print(f"[pk] wrote {out_base}.{{png,pdf,npz}}")

    # --- 7. Stdout summary ---
    print("\n=== Transfer function by ℓ bin ===")
    header = f"{'ℓ bin':>10} {'C_l^truth':>12}"
    for r in results:
        header += f" {r['label'][:16]:>18}"
    print(header)
    for i, lc in enumerate(l_centre_truth):
        row = f"{lc:10.1f} {cl_truth_b[i]:12.3e}"
        for r in results:
            row += f" {r['transfer_binned'][i]:18.3f}"
        print(row)


if __name__ == "__main__":
    main()
