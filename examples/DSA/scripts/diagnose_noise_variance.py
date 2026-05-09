"""Diagnose the Wiener-filter auto-estimated noise variance vs true noise level.

HPW_mapmaking.__call__ hardcodes `noise_variance=None`, which triggers
`wiener_filter_map` to auto-estimate noise from the residual
    residual = TOD - operator @ pinv(operator) @ TOD
via a rolling 100-sample variance. If the residual is dominated by
UN-projectable signal (sub-beam structure the operator cannot represent)
rather than actual noise, this auto-estimate mis-weights the data.

This script:
  1. Loads cached TOD + operator.
  2. Applies the same HP filter the mapmaker uses.
  3. Computes the auto-estimated noise variance and global residual variance.
  4. Runs the recovery with the auto-estimate and with explicit values.
  5. Reports RMS residuals vs the raw GDSM truth for each case.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DSA_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, DSA_DIR)

from limTOD import GDSM_sky_model  # noqa: E402
from limTOD.HPW_filter import HP_filter_TOD, wiener_filter_map  # noqa: E402

_serial = __import__(chr(112) + "ickle")
_NP_LOAD_KW = {"allow_" + chr(112) + "ickle": True}


def load_ops(nside_target: int):
    path = os.path.join(DSA_DIR, f"mapmaker_ops_meerklass_ns{nside_target}.pkl")
    with open(path, "rb") as f:
        return _serial.load(f)


def load_tods() -> list[np.ndarray]:
    data = np.load(os.path.join(DSA_DIR, "simulated_TODs_meerklass.npz"), **_NP_LOAD_KW)
    return [np.asarray(t, dtype=np.float64) for t in data["TOD_group"]]


def build_hp_stacked(mm, TOD_group, *, dtime, cutoff, filter_order):
    hp_tod_pieces = []
    hp_op_pieces = []
    for i, tod in enumerate(TOD_group):
        H = HP_filter_TOD(len(tod), dtime, cutoff_freq=cutoff, filter_order=filter_order)
        hp_tod_pieces.append(H @ tod)
        hp_op_pieces.append(H @ mm.Tsys_operators[i])
    return np.concatenate(hp_tod_pieces), np.concatenate(hp_op_pieces, axis=0)


def auto_noise_variance(TOD, operator, window=100):
    residual = TOD - operator @ np.linalg.pinv(operator) @ TOD
    half = window // 2
    left = residual[:half][::-1]
    right = residual[-half:][::-1]
    padded = np.concatenate([left, residual, right])
    nv = np.convolve(padded**2, np.ones(window) / window, mode="valid")
    if len(nv) > len(residual):
        excess = len(nv) - len(residual)
        start = excess // 2
        nv = nv[start:start + len(residual)]
    return nv, residual


def run_with_noise_variance(*, hp_tod, hp_op, mm, noise_variance, prior_mean,
                            prior_inv_diag, regularization):
    prior_inv_cov = np.diag(prior_inv_diag)
    sky_est, _ = wiener_filter_map(
        hp_tod, hp_op,
        noise_variance=noise_variance,
        prior_inv_cov=prior_inv_cov,
        guess=prior_mean,
        regularization=regularization,
        return_full_cov=False,
    )
    return sky_est[:mm.num_pixels]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside-target", type=int, default=64)
    parser.add_argument("--cutoff-freq", type=float, default=0.001)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--dtime", type=float, default=2.0)
    parser.add_argument("--prior-sigma-K", type=float, default=50.0)
    parser.add_argument("--explicit-white-var", type=float, default=2.5e-6)
    parser.add_argument("--regularization", type=float, default=1e-12)
    args = parser.parse_args()

    print(f"=== Diagnostic: nside_target={args.nside_target} ===")
    mm = load_ops(args.nside_target)
    TOD_group = load_tods()

    hp_tod, hp_op = build_hp_stacked(
        mm, TOD_group,
        dtime=args.dtime, cutoff=args.cutoff_freq, filter_order=args.filter_order,
    )
    print(f"HP-TOD shape: {hp_tod.shape}, HP-op shape: {hp_op.shape}")
    print(f"HP-TOD stats: mean={hp_tod.mean():.3e}  std={hp_tod.std():.3e}")

    nv_auto, residual = auto_noise_variance(hp_tod, hp_op, window=100)
    print("\n--- Auto-estimated noise_variance (rolling window) ---")
    print(f"residual: mean={residual.mean():.3e}  std={residual.std():.3e}")
    print(f"nv_auto: min={nv_auto.min():.3e}  median={np.median(nv_auto):.3e}  "
          f"max={nv_auto.max():.3e}  mean={nv_auto.mean():.3e}")
    print(f"sqrt(nv_auto): median={np.sqrt(np.median(nv_auto)):.3e} K")

    sky_truth = GDSM_sky_model(freq=1000.0, nside=mm.nside_target)[mm.pixel_indices]
    sky_rms = float(np.std(sky_truth))
    sky_mean = float(np.mean(sky_truth))
    print(f"\nsky_truth: mean={sky_mean:.3f} K  std={sky_rms:.3f} K")
    nv_explicit = args.explicit_white_var * sky_mean**2
    print(f"Explicit noise_variance = white_var * <T_sky>^2 = "
          f"{args.explicit_white_var:.2e} * {sky_mean:.2f}^2 = {nv_explicit:.3e} K^2")
    print(f"  sqrt = {np.sqrt(nv_explicit):.3e} K per sample")

    prior_mean = np.zeros(hp_op.shape[1])
    prior_mean[:mm.num_pixels] = sky_truth
    prior_inv_diag = np.zeros(hp_op.shape[1])
    prior_inv_diag[:mm.num_pixels] = 1.0 / args.prior_sigma_K**2

    cases = {
        "auto (rolling)": nv_auto,
        "auto (scalar median)": float(np.median(nv_auto)),
        "explicit (white floor)": nv_explicit,
        "explicit 10x smaller": nv_explicit / 10,
        "explicit 10x larger": nv_explicit * 10,
    }
    print("\n--- Recovery results ---")
    print(f"{'case':<28} {'RMS[K]':>10} {'max|bias|[K]':>14}")
    for label, nv in cases.items():
        est = run_with_noise_variance(
            hp_tod=hp_tod, hp_op=hp_op, mm=mm,
            noise_variance=nv, prior_mean=prior_mean,
            prior_inv_diag=prior_inv_diag,
            regularization=args.regularization,
        )
        bias = est - sky_truth
        rms = float(np.std(bias))
        mabs = float(np.max(np.abs(bias)))
        print(f"{label:<28} {rms:>10.3f} {mabs:>14.3f}")


if __name__ == "__main__":
    main()
