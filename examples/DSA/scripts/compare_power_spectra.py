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
    """Per-scenario panels: each uses its OWN sensitivity mask for both
    truth and recovery, so each survey is compared against the piece of
    sky it actually observes (not the 3-way intersection)."""

    ell = np.arange(LMAX + 1)
    bin_edges = np.unique(np.logspace(np.log10(2), np.log10(LMAX + 1), 12).astype(int))
    bin_edges = bin_edges[bin_edges <= LMAX + 1]

    # Compute each scenario's own-mask Cℓ
    for r in results:
        mask = np.zeros(hp.nside2npix(NSIDE), dtype=np.float64)
        mask[r["pixel_indices"]] = 1.0
        f_sky = mask.mean()
        sky_est_full = embed_patch(r["sky_est"], r["pixel_indices"], NSIDE)
        cl_truth = masked_cls(sky_truth_full, mask, LMAX)
        cl_rec = masked_cls(sky_est_full, mask, LMAX)
        l_centre, cl_truth_b = bin_cls(ell, cl_truth, bin_edges)
        _, cl_rec_b = bin_cls(ell, cl_rec, bin_edges)
        r["l_centre"] = l_centre
        r["cl_truth_binned"] = cl_truth_b
        r["cl_rec_binned"] = cl_rec_b
        r["transfer_binned"] = cl_rec_b / cl_truth_b
        r["f_sky"] = f_sky
        print(f"[pk] ({hp_tag}) {r['label']}: f_sky={f_sky:.4f}, "
              f"n_pix={len(r['pixel_indices'])}")

    # --- Plot: 2 rows × 3 columns. Row 1 = Cℓ, Row 2 = transfer ---
    ell_beam = 180.0 / DSA_BEAM_FWHM_DEG
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.0), constrained_layout=True, sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    subbeam_kw = dict(color="lightgrey", alpha=0.35, zorder=0)

    # Consistent y-limits for Cℓ and transfer across the three columns
    cl_all = np.concatenate(
        [r["cl_truth_binned"] for r in results] +
        [r["cl_rec_binned"] for r in results]
    )
    cl_pos = cl_all[cl_all > 0]
    cl_ymin = 10 ** np.floor(np.log10(cl_pos.min()))
    cl_ymax = 10 ** np.ceil(np.log10(cl_pos.max()))

    for col, r in enumerate(results):
        ax_cl = axes[0, col]
        ax_T = axes[1, col]
        ax_cl.axvspan(ell_beam, LMAX + 1, **subbeam_kw)
        ax_T.axvspan(ell_beam, LMAX + 1, **subbeam_kw)

        # Cℓ panel — wide transparent truth + sharp recovery
        ax_cl.loglog(r["l_centre"], r["cl_truth_binned"], color="black",
                     lw=8, alpha=0.22, label="GDSM truth",
                     solid_capstyle="round", zorder=1)
        ax_cl.loglog(r["l_centre"], r["cl_rec_binned"], color=r["colour"],
                     label="recovered", marker="o", ms=6,
                     mec="white", mew=0.8)
        ax_cl.axvline(ell_beam, color="0.35", ls="--", lw=1.2)
        ax_cl.set_title(r["label"], pad=8)
        ax_cl.set_ylim(cl_ymin, cl_ymax)
        if col == 0:
            ax_cl.set_ylabel(r"$C_\ell$  [K$^2$]")
        ax_cl.legend(loc="lower left", frameon=True, framealpha=0.9,
                     fancybox=False, edgecolor="0.7", fontsize=10)

        # Transfer panel
        ax_T.semilogx(r["l_centre"], r["transfer_binned"],
                      color=r["colour"], marker="o", ms=6,
                      mec="white", mew=0.8, label=r"$T_\ell$")
        ax_T.axhline(1.0, color="k", ls="-", lw=1.0, alpha=0.6)
        ax_T.axvline(ell_beam, color="0.35", ls="--", lw=1.2,
                     label=r"$\ell_{\rm beam}\approx %.0f$" % ell_beam)
        ax_T.set_ylim(0.0, 2.2)
        ax_T.set_xlim(r["l_centre"][0] * 0.9, r["l_centre"][-1] * 1.1)
        ax_T.set_xlabel(r"multipole $\ell$")
        if col == 0:
            ax_T.set_ylabel(
                r"transfer  $C_\ell^{\rm rec}\,/\,C_\ell^{\rm truth}$")
        ax_T.legend(loc="lower left", frameon=True, framealpha=0.9,
                    fancybox=False, edgecolor="0.7", fontsize=10)

    fig.suptitle(
        f"Angular power spectra: each survey on its own observed patch  "
        f"({title_suffix})",
        fontsize=13,
    )

    out_base = os.path.join(DSA_DIR, "figures",
                            f"power_spectra_comparison_{hp_tag}")
    fig.savefig(out_base + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Dump binned arrays for reproducibility ---
    np.savez(
        out_base + ".npz",
        ell_beam=ell_beam,
        **{f"l_centre_{r['kind']}{r['suffix']}": r["l_centre"] for r in results},
        **{f"cl_truth_{r['kind']}{r['suffix']}": r["cl_truth_binned"] for r in results},
        **{f"cl_rec_{r['kind']}{r['suffix']}": r["cl_rec_binned"] for r in results},
        **{f"transfer_{r['kind']}{r['suffix']}": r["transfer_binned"] for r in results},
        **{f"f_sky_{r['kind']}{r['suffix']}": r["f_sky"] for r in results},
    )
    print(f"[pk] wrote {out_base}.{{png,pdf,npz}}")

    # Stdout summary per scenario
    for r in results:
        print(f"\n=== {r['label']} ({hp_tag}, own mask) ===")
        print(f"{'ℓ bin':>10} {'C_l^truth':>12} {'C_l^rec':>12} {'T_l':>8}")
        for i, lc in enumerate(r["l_centre"]):
            print(f"{lc:10.1f} {r['cl_truth_binned'][i]:12.3e} "
                  f"{r['cl_rec_binned'][i]:12.3e} "
                  f"{r['transfer_binned'][i]:8.3f}")


if __name__ == "__main__":
    main()
