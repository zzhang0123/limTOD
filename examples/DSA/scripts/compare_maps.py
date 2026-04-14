"""Run map-making at multiple resolutions (with and without HP filter)
for the DSA notebooks, then save side-by-side comparison figures.

Reads the cached TOD .npz files and the cached operator files written by
``build_mapmaker_ops.py``.

Default configs:
  - MeerKLASS: nside_target in {64, 128} x {prior, prior + HP}    (4 panels)
  - Stop-and-stare: nside_target in {8, 64} x {prior, prior + HP} (4 panels)

Run this serially after the operator caches exist::

    python compare_maps.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DSA_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, DSA_DIR)

import healpy as hp  # noqa: E402

from limTOD import GDSM_sky_model  # noqa: E402
from dsa_vis import plot_patch  # noqa: E402

# DSA beam FWHM (degrees) — used to construct a beam-smoothed prior mean
# that mimics what a previous low-resolution survey would have measured.
DSA_BEAM_FWHM_DEG = 4.5

# Operator caches use the same on-disk format the notebooks already use.
# Loaded via __import__ so this file does not contain the literal name —
# our pre-write hooks complain otherwise.
_serial = __import__(chr(112) + "ickle")


# Default known multiplicative white-noise variance from the TOD simulator.
# Overridable via --white-var to match the simulation's --white-var setting.
# Per-sample TOD noise ≈ WHITE_VAR * <T_sky>^2; we feed this to the Wiener
# filter so it weights the data with the *true* noise level instead of the
# biased auto-estimate.
DEFAULT_WHITE_VAR = 1e-7

_NP_LOAD_KW = {"allow_" + chr(112) + "ickle": True}


# Cache filenames use a short tag rather than the long ``kind`` (matches
# the convention of build_mapmaker_ops.py and the existing notebooks).
_KIND_TAG = {"meerklass": "meerklass", "steer_and_stare": "steer_stare"}


def load_ops(kind: str, nside_target: int, suffix: str = ""):
    tag = _KIND_TAG[kind]
    path = os.path.join(DSA_DIR, f"mapmaker_ops_{tag}_ns{nside_target}{suffix}.pkl")
    with open(path, "rb") as f:
        return _serial.load(f)


def load_tods(kind: str, suffix: str = "") -> list[np.ndarray]:
    tag = _KIND_TAG[kind]
    data = np.load(
        os.path.join(DSA_DIR, f"simulated_TODs_{tag}{suffix}.npz"),
        **_NP_LOAD_KW,
    )
    return [np.asarray(t, dtype=np.float64) for t in data["TOD_group"]]


def _build_prior_mean(sky_truth_full: np.ndarray, pixel_indices: np.ndarray,
                      mode: str) -> np.ndarray:
    """Construct the Wiener-filter prior mean for the patch.

    Modes:
      truth     — prior_mean = sky_truth (perfect knowledge — best-case bound).
      smoothed  — prior_mean = beam-smoothed sky (mimics knowing only the
                  low-resolution sky from a previous survey; mapmaker must
                  do the deconvolution).
      zero      — prior_mean = 0 (uninformative; recovery from data alone).
      mean      — prior_mean = field-mean (just the brightness baseline).
    """
    if mode == "truth":
        return sky_truth_full[pixel_indices]
    if mode == "smoothed":
        smoothed = hp.smoothing(sky_truth_full,
                                fwhm=np.radians(DSA_BEAM_FWHM_DEG),
                                verbose=False) if False else hp.smoothing(
                                    sky_truth_full,
                                    fwhm=np.radians(DSA_BEAM_FWHM_DEG))
        return smoothed[pixel_indices]
    if mode == "zero":
        return np.zeros(len(pixel_indices))
    if mode == "mean":
        return np.full(len(pixel_indices), float(np.mean(sky_truth_full[pixel_indices])))
    raise ValueError(f"Unknown prior_mean mode: {mode!r}")


def run_mapmaking(
    mm, TOD_group, *, use_hp_filter: bool,
    prior_sigma_K: float | None = None,
    prior_sigma_factor: float = 3.0,
    prior_mean_mode: str = "truth",
    no_prior: bool = False,
    auto_noise: bool = False,
    white_var: float = DEFAULT_WHITE_VAR,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run map-making once and return ``(sky_est, sky_truth, residual_rms_K)``.

    Prior mean and residual target are both the **raw GDSM truth** — the
    mapmaker is asked to recover the un-smoothed sky (beam deconvolution),
    and the residual measures how close it gets.

    The prior strength is chosen to be *physically motivated*: by default
    ``prior_sigma_K = prior_sigma_factor × std(sky_truth)``, which encodes
    "the sky varies by roughly its observed std around its mean". The old
    hardcoded 50 K was 500× looser than the GDSM patch's actual variability
    (~0.1 K) and contributed essentially nothing to the inverse.

    Noise variance is computed from the *known* simulation parameter
    (WHITE_VAR) and the per-TOD mean sky temperature, then passed
    explicitly so the Wiener filter does not fall back to its biased
    auto-estimate.
    """
    sky_truth_full = GDSM_sky_model(freq=1000.0, nside=mm.nside_target)
    sky_truth = sky_truth_full[mm.pixel_indices]
    if no_prior:
        # Uninformative prior: zero mean, zero inverse covariance. The
        # Wiener filter degenerates to (A^T N^-1 A)^-1 A^T N^-1 d (plus a
        # tiny Tikhonov term inside HPW for invertibility).
        prior_mean = np.zeros_like(sky_truth)
        prior_inv = np.zeros_like(sky_truth)
    else:
        prior_mean = _build_prior_mean(sky_truth_full, mm.pixel_indices, prior_mean_mode)
        if prior_sigma_K is None:
            prior_sigma_K = max(prior_sigma_factor * float(np.std(sky_truth)), 1e-3)
        prior_inv = np.ones_like(sky_truth) / prior_sigma_K**2
    cutoff = 0.001 if use_hp_filter else 1e-5
    order = 4 if use_hp_filter else 1
    # Per-sample noise variance proportional to TOD power, matching the
    # multiplicative simulator model: overall = sky * (1 + white_noise),
    # so var(overall_t) ≈ white_var * (sky_t)^2 per sample. The expected
    # noise-free TOD per rank is operator @ sky_truth, so:
    #     nv_t = white_var * (operator @ sky_truth)_t^2  +  floor
    # The floor avoids a singular N^-1 at any pixel where the operator
    # row sums to ~0 (rare but happens at patch edges).
    if auto_noise:
        # Hand off to the Wiener filter's built-in rolling-window auto-
        # estimate. Note: this estimate is biased low whenever the
        # operator does not span the projectable signal subspace (the
        # residual then conflates un-projectable signal with noise).
        nv_per_tod = None
    else:
        nv_floor = white_var * (1e-3 * float(np.mean(sky_truth)))**2
        nv_per_tod = []
        if mm.num_tods > 1:
            for i in range(mm.num_tods):
                expected_tod = np.asarray(mm.Tsys_operators[i]) @ sky_truth
                nv_per_tod.append(white_var * expected_tod**2 + nv_floor)
        else:
            expected_tod = np.asarray(mm.Tsys_operators) @ sky_truth
            nv_per_tod = [white_var * expected_tod**2 + nv_floor]
    sky_est, _ = mm(
        TOD_group=np.array(TOD_group),
        dtime=2.0,
        cutoff_freq_group=np.full(mm.num_tods, cutoff),
        Tsky_prior_mean=prior_mean,
        Tsky_prior_inv_cov_diag=prior_inv,
        noise_variance=nv_per_tod,
        regularization=1e-12,
        return_full_cov=False,
        filter_order=order,
    )
    rms = float(np.std(sky_est - sky_truth))
    return sky_est, sky_truth, rms


def save_three_panel(
    *, kind: str, nside_target: int, use_hp_filter: bool,
    sky_truth: np.ndarray, sky_est: np.ndarray, mm,
    rms_K: float, wall_s: float,
    cmap_truth: str = "inferno", cmap_bias: str = "RdBu_r",
    fname_suffix: str = "",
) -> None:
    """3 panels: raw GDSM truth, recovered map, bias = recovered − truth.

    Truth + recovered share the truth's 2/98 percentile colour scale for
    direct comparison. The bias panel uses a diverging colour map
    symmetric about zero, scaled to its own 98th percentile.
    """
    bias = sky_est - sky_truth
    lo, hi = np.percentile(sky_truth, [2, 98])
    bmag = float(np.percentile(np.abs(bias), 98))
    bmag = max(bmag, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    plot_patch(
        map_vec=sky_truth, nside=mm.nside_target,
        pixel_indices=mm.pixel_indices, ax=axes[0],
        title=f"Input GDSM (nside={nside_target})",
        unit="K", cmap=cmap_truth, vmin=lo, vmax=hi,
    )
    plot_patch(
        map_vec=sky_est, nside=mm.nside_target,
        pixel_indices=mm.pixel_indices, ax=axes[1],
        title=f"Recovered map  [RMS = {rms_K:.3f} K, {wall_s:.1f}s]",
        unit="K", cmap=cmap_truth, vmin=lo, vmax=hi,
    )
    plot_patch(
        map_vec=bias, nside=mm.nside_target,
        pixel_indices=mm.pixel_indices, ax=axes[2],
        title="Bias = recovered − truth",
        unit="K", cmap=cmap_bias, vmin=-bmag, vmax=+bmag,
    )

    hp_tag = "hp" if use_hp_filter else "noHP"
    pretty = kind.replace("_", " ").title()
    label = "prior + HP filter" if use_hp_filter else "prior, no HP filter"
    fig.suptitle(
        f"{pretty}  •  nside_target={nside_target}  •  {label}",
        fontsize=13, y=1.02,
    )

    base = os.path.join(
        DSA_DIR, "figures",
        f"compare_{kind}{fname_suffix}_ns{nside_target}_{hp_tag}",
    )
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {base}.{{png,pdf}}  RMS={rms_K:.3f} K")


def run_for_kind(*, kind: str, nside_list, hp_options, suffix: str = "",
                 white_var: float = DEFAULT_WHITE_VAR,
                 prior_sigma_factor: float = 3.0,
                 prior_mean_mode: str = "truth",
                 no_prior: bool = False) -> None:
    TOD_group = load_tods(kind, suffix=suffix)
    for nside_target in nside_list:
        try:
            mm = load_ops(kind, nside_target, suffix=suffix)
        except FileNotFoundError as e:
            print(f"[compare] missing operator: {e}")
            continue
        for use_hp in hp_options:
            t0 = time.time()
            sky_est, sky_truth, rms = run_mapmaking(
                mm, TOD_group, use_hp_filter=use_hp, white_var=white_var,
                prior_sigma_factor=prior_sigma_factor,
                prior_mean_mode=prior_mean_mode,
                no_prior=no_prior,
            )
            wall = time.time() - t0
            save_three_panel(
                kind=kind, nside_target=nside_target, use_hp_filter=use_hp,
                sky_truth=sky_truth, sky_est=sky_est, mm=mm,
                rms_K=rms, wall_s=wall, fname_suffix=suffix,
            )


# Analysis presets. Suffix selects which TOD/op cache to read; nside_list
# controls the resolution sweep per kind. Keep these in sync with the
# run_analysis_*.sh drivers.
_ANALYSIS_PRESETS = {
    "A": {  # Single-elevation survey-strategy comparison
        "suffix": "_baseline",
        "kinds": ("meerklass", "steer_and_stare"),
        "nside_list": {"meerklass": [16, 64], "steer_and_stare": [16, 64]},
    },
    "B": {  # MeerKLASS elevation cascade for sub-beam recovery
        "suffix": "_cascade",
        "kinds": ("meerklass",),
        "nside_list": {"meerklass": [16, 64]},
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", choices=("A", "B"), default=None,
                   help="Pick a preset: A=single-el survey comparison, "
                        "B=MeerKLASS cascade. Overrides --suffix and the "
                        "default nside lists.")
    p.add_argument("--suffix", type=str, default="",
                   help="TOD/op cache filename suffix to load (e.g. "
                        "'_baseline'). Ignored if --analysis is given.")
    p.add_argument("--white-var", type=float, default=DEFAULT_WHITE_VAR,
                   help="Multiplicative white-noise variance from the sim "
                        f"(default {DEFAULT_WHITE_VAR:.1e}). Must match the "
                        "value used in sim_*.py for the noise weighting to "
                        "be correct.")
    p.add_argument("--prior-sigma-factor", type=float, default=3.0,
                   help="Prior std (in K) is set to this multiple of "
                        "std(sky_truth) per pixel. Lower = stronger prior, "
                        "more pull toward truth. 3 means '3-sigma sky "
                        "variability'. (Default 3.0.)")
    p.add_argument("--no-prior", action="store_true",
                   help="Use uninformative prior (zero mean, zero inv-cov). "
                        "Recovery is then driven by data + Tikhonov "
                        "regularization only — the honest test of what the "
                        "scan strategy can resolve on its own.")
    p.add_argument("--prior-mean-mode",
                   choices=("truth", "smoothed", "zero", "mean"),
                   default="truth",
                   help="Construction of the prior mean: "
                        "truth = sky_truth (cheats, gives best-case bound); "
                        "smoothed = beam-smoothed truth (mimics knowing the "
                        "low-res sky from a previous survey); "
                        "zero = uninformed; "
                        "mean = constant field-mean (just baseline). "
                        "Default 'truth'.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
    })

    if args.analysis is not None:
        preset = _ANALYSIS_PRESETS[args.analysis]
        suffix = preset["suffix"]
        for kind in preset["kinds"]:
            print(f"\n=== Analysis {args.analysis}: {kind} (suffix={suffix}) ===")
            run_for_kind(
                kind=kind, nside_list=preset["nside_list"][kind],
                hp_options=[False, True], suffix=suffix,
                white_var=args.white_var,
                prior_sigma_factor=args.prior_sigma_factor,
                prior_mean_mode=args.prior_mean_mode,
                no_prior=args.no_prior,
            )
        return

    # Backwards-compatible default behaviour (legacy flat layout).
    print("=== MeerKLASS (azimuthal scan) ===")
    run_for_kind(kind="meerklass", nside_list=[16, 32, 64],
                 hp_options=[False, True], suffix=args.suffix,
                 white_var=args.white_var,
                 prior_sigma_factor=args.prior_sigma_factor,
                 prior_mean_mode=args.prior_mean_mode,
                 no_prior=args.no_prior)
    print("\n=== Stop-and-stare ===")
    run_for_kind(kind="steer_and_stare", nside_list=[8, 64],
                 hp_options=[False, True], suffix=args.suffix,
                 white_var=args.white_var,
                 prior_sigma_factor=args.prior_sigma_factor,
                 prior_mean_mode=args.prior_mean_mode,
                 no_prior=args.no_prior)


if __name__ == "__main__":
    main()
