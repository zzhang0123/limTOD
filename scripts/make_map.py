"""Generic HPW map-making driver for limTOD.

Reads a TOD cache produced by ``scripts/simulate_tod.py`` (or any
equivalent cache with the same schema) and reconstructs a patch of sky
with the high-pass + Wiener filter solver ``HPW_mapmaking``.

Output
------
- ``<out>.npz``: recovered sky, input GDSM truth restricted to the
  sensitivity mask, pixel indices, residual RMS, run config.
- ``<out>.png``: side-by-side truth/recovered/residual quick-look plot.

Usage::

    python scripts/make_map.py \\
        --tod /tmp/tod.npz  --out /tmp/map --nside 64 --hp-cutoff 0.03
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import healpy as hp  # noqa: E402

from limTOD import (  # noqa: E402
    GDSM_sky_model,
    HPW_mapmaking,
    example_symm_beam_map,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HPW map-making driver for limTOD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tod", type=str, required=True,
                   help="Input TOD cache (npz from simulate_tod.py).")
    p.add_argument("--out", type=str, required=True,
                   help="Output path base (writes <out>.npz and <out>.png).")
    p.add_argument("--nside", type=int, default=64,
                   help="HEALPix nside for the recovered map.")
    p.add_argument("--hp-cutoff", type=float, default=3e-2,
                   help="High-pass filter cutoff in Hz.")
    p.add_argument("--filter-order", type=int, default=4,
                   help="Butterworth HP filter order.")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Pixel selection threshold (beam-peak fraction).")
    p.add_argument("--beam-nside-map", type=int, default=64,
                   help="Beam HEALPix nside used inside the mapmaker.")
    p.add_argument("--prior-sigma-factor", type=float, default=1.0,
                   help="Prior sigma in units of std(sky_truth). <=0 "
                        "disables the prior.")
    p.add_argument("--auto-noise", action="store_true",
                   help="Let the Wiener filter auto-estimate per-sample "
                        "noise (default: explicit white_var*(op@sky)²).")
    return p.parse_args(argv)


def _load_cache(path: str) -> dict:
    """Load a simulate_tod.py npz cache into plain float64 arrays."""
    data = np.load(path)

    def per_tod(key: str) -> list[np.ndarray]:
        arr = np.asarray(data[key], dtype=np.float64)
        if arr.ndim == 1:  # back-compat with single-TOD dumps
            arr = arr[None, :]
        return [arr[i] for i in range(arr.shape[0])]

    freq_list = np.asarray(data["freq_list"], dtype=np.float64).tolist()

    meta: dict[str, float | int | str] = {}
    for key in data.files:
        if not key.startswith("meta_"):
            continue
        val = data[key]
        if val.dtype.kind in "US":
            meta[key[5:]] = str(val)
        elif val.dtype.kind in "if":
            meta[key[5:]] = val.item()

    return dict(
        TOD_group=per_tod("TOD_group"),
        LST_deg_list_group=per_tod("LST_deg_list_group"),
        azimuth_deg_list_group=per_tod("azimuth_deg_list_group"),
        elevation_deg_list_group=per_tod("elevation_deg_list_group"),
        freq_list=freq_list,
        meta=meta,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cache = _load_cache(args.tod)
    freq_mhz = float(cache["freq_list"][0])

    meta = cache["meta"]
    fwhm_deg = float(meta.get("beam_fwhm", 1.1))
    lat_deg = float(meta.get("lat", -30.7130))
    dt_seconds = float(meta.get("dt", 2.0))
    white_var = float(meta.get("white_var", 1e-7))

    # Same beam the simulation used — sum-normalised Gaussian.
    beam_map = example_symm_beam_map(
        freq=freq_mhz, nside=args.beam_nside_map, FWHM=fwhm_deg,
    )

    print(f"[map] building mapmaker operator (nside={args.nside}, "
          f"beam_nside={args.beam_nside_map}, threshold={args.threshold})")
    t0 = time.perf_counter()
    mm = HPW_mapmaking(
        beam_map=beam_map,
        LST_deg_list_group=cache["LST_deg_list_group"],
        lat_deg=lat_deg,
        azimuth_deg_list_group=cache["azimuth_deg_list_group"],
        elevation_deg_list_group=cache["elevation_deg_list_group"],
        threshold=args.threshold,
        nside_target=args.nside,
        beam_truncate_frac_thres=args.threshold,
    )
    t_build = time.perf_counter() - t0
    pixel_indices = mm.pixel_indices
    print(f"[map] operator built in {t_build:.1f} s  "
          f"({len(pixel_indices)} pixels)")

    # Prior: beam-smoothed truth as the prior mean; prior_sigma data-driven.
    sky_truth_full = GDSM_sky_model(freq=freq_mhz, nside=args.nside)
    sky_truth = sky_truth_full[pixel_indices]
    if args.prior_sigma_factor > 0:
        prior_mean_full = hp.smoothing(
            sky_truth_full, fwhm=np.radians(fwhm_deg))
        prior_mean = prior_mean_full[pixel_indices]
        prior_sigma = max(
            args.prior_sigma_factor * float(np.std(sky_truth)), 1e-3)
        prior_inv = np.ones_like(sky_truth) / prior_sigma**2
    else:
        prior_mean = np.zeros_like(sky_truth)
        prior_inv = np.zeros_like(sky_truth)

    # Explicit proportional noise variance matches the simulator's
    # multiplicative white-noise model, avoiding the Wiener filter's
    # biased auto-estimate.
    if args.auto_noise:
        noise_variance = None
    else:
        nv_floor = white_var * (1e-3 * float(np.mean(sky_truth)))**2
        noise_variance = []
        if mm.num_tods > 1:
            for op_i in mm.Tsys_operators:
                expected = np.asarray(op_i) @ sky_truth
                noise_variance.append(white_var * expected**2 + nv_floor)
        else:
            expected = np.asarray(mm.Tsys_operators) @ sky_truth
            noise_variance = [white_var * expected**2 + nv_floor]

    cutoff_group = np.full(mm.num_tods, args.hp_cutoff, dtype=np.float64)

    print(f"[map] solving (HP cutoff = {args.hp_cutoff} Hz, "
          f"order={args.filter_order})")
    t0 = time.perf_counter()
    sky_est, sky_unc = mm(
        TOD_group=cache["TOD_group"],
        dtime=dt_seconds,
        cutoff_freq_group=cutoff_group,
        Tsky_prior_mean=prior_mean,
        Tsky_prior_inv_cov_diag=prior_inv,
        noise_variance=noise_variance,
        regularization=1e-12,
        filter_order=args.filter_order,
    )
    t_solve = time.perf_counter() - t0
    rms_K = float(np.std(sky_est - sky_truth))
    print(f"[map] solved in {t_solve:.1f} s  |  residual RMS = "
          f"{rms_K*1e3:.1f} mK")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    npz_path = args.out + ".npz"
    np.savez(
        npz_path,
        sky_est=sky_est,
        sky_truth=sky_truth,
        sky_unc=sky_unc,
        pixel_indices=pixel_indices,
        residual_rms_K=rms_K,
        nside=args.nside,
        freq_mhz=freq_mhz,
        hp_cutoff=args.hp_cutoff,
        threshold=args.threshold,
    )
    print(f"[map] wrote {npz_path}")

    # Quick-look plot.
    try:
        _plot_three_panel(
            sky_truth=sky_truth, sky_est=sky_est,
            pixel_indices=pixel_indices, nside=args.nside,
            freq_mhz=freq_mhz, rms_K=rms_K,
            png_path=args.out + ".png",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[map] plot skipped: {exc}")


def _plot_three_panel(
    *,
    sky_truth: np.ndarray,
    sky_est: np.ndarray,
    pixel_indices: np.ndarray,
    nside: int,
    freq_mhz: float,
    rms_K: float,
    png_path: str,
) -> None:
    """Simple truth / recovered / residual figure via healpy cartview."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _embed(vec: np.ndarray) -> np.ndarray:
        m = np.full(hp.nside2npix(nside), hp.UNSEEN)
        m[pixel_indices] = vec
        return m

    bias = sky_est - sky_truth
    lo, hi = np.percentile(sky_truth, [2, 98])
    bmag = max(float(np.percentile(np.abs(bias), 98)), 1e-6)

    fig = plt.figure(figsize=(15, 4.4))
    hp.cartview(_embed(sky_truth), fig=fig, sub=(1, 3, 1),
                title=f"Truth ({freq_mhz:.0f} MHz)",
                unit="K", min=lo, max=hi, cmap="inferno")
    hp.cartview(_embed(sky_est), fig=fig, sub=(1, 3, 2),
                title=f"Recovered (RMS={rms_K*1e3:.1f} mK)",
                unit="K", min=lo, max=hi, cmap="inferno")
    hp.cartview(_embed(bias), fig=fig, sub=(1, 3, 3),
                title="Residual = recovered - truth",
                unit="K", min=-bmag, max=bmag, cmap="RdBu_r")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[map] wrote {png_path}")


if __name__ == "__main__":
    main()
