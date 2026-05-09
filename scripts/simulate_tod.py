"""Generic TOD simulation driver for limTOD.

Runs a single back-and-forth azimuth scan at constant elevation and
saves the resulting TOD (plus pointing metadata) to an ``.npz`` cache
that ``scripts/make_map.py`` can consume.

Key point vs. the older ``examples/DSA/scripts/sim_*.py`` drivers:
this script does **not** implement its own MPI harness. Since limTOD
1.2 the parallelism lives inside ``TODSim.generate_TOD`` and kicks
in automatically under ``mpirun``. Serial and MPI invocations produce
bit-identical output given the same RNG seed.

Usage (serial)::

    python scripts/simulate_tod.py --out /tmp/tod.npz

Usage (4-way MPI)::

    env OMP_NUM_THREADS=1 mpirun -n 4 python scripts/simulate_tod.py --out /tmp/tod.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Sequence

import numpy as np

# Make the repo importable when the script is launched directly.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from limTOD import (  # noqa: E402
    GDSM_sky_model,
    TODSim,
    example_scan,
    example_symm_beam_map,
    mpiutil,
)


def _log(msg: str) -> None:
    """Print only on rank 0 to keep MPI logs readable."""
    if mpiutil.rank0:
        print(msg, flush=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generic limTOD TOD simulation driver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=str, required=True,
                   help="Output .npz cache path (rank 0 writes).")
    p.add_argument("--freq-mhz", type=float, default=1000.0,
                   help="Observing frequency in MHz (single-frequency sim).")
    p.add_argument("--az-start", type=float, default=-60.0,
                   help="Azimuth sweep start [deg].")
    p.add_argument("--az-end", type=float, default=-42.0,
                   help="Azimuth sweep end [deg].")
    p.add_argument("--el", type=float, default=45.0,
                   help="Constant elevation [deg].")
    p.add_argument("--dt", type=float, default=2.0,
                   help="Sample interval in seconds.")
    p.add_argument("--n-repeats", type=int, default=13,
                   help="Number of back-and-forth sweeps.")
    p.add_argument("--start-utc", type=str,
                   default="2024-04-15 05:00:00",
                   help="UTC start time.")
    p.add_argument("--lat", type=float, default=-30.7130,
                   help="Antenna latitude [deg] (default: MeerKAT).")
    p.add_argument("--lon", type=float, default=21.4430,
                   help="Antenna longitude [deg] (default: MeerKAT).")
    p.add_argument("--height", type=float, default=1054.0,
                   help="Antenna height [m] (default: MeerKAT).")
    p.add_argument("--beam-fwhm", type=float, default=1.1,
                   help="Gaussian beam FWHM [deg] (uses example_symm_beam_map).")
    p.add_argument("--sky-nside", type=int, default=64,
                   help="HEALPix nside for the sky map.")
    p.add_argument("--beam-nside", type=int, default=64,
                   help="HEALPix nside for the beam map.")
    p.add_argument("--truncate-frac", type=float, default=1e-3,
                   help="Beam truncation fraction-of-peak.")
    p.add_argument("--white-var", type=float, default=1e-7,
                   help="Multiplicative white-noise variance (fractional).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (applied on rank 0 before noise draws).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    # Bind the beam FWHM into the callable that TODSim expects.
    fwhm_deg = args.beam_fwhm

    def beam_func(*, freq: float, nside: int) -> np.ndarray:
        return example_symm_beam_map(freq=freq, nside=nside, FWHM=fwhm_deg)

    tod_sim = TODSim(
        ant_latitude_deg=args.lat,
        ant_longitude_deg=args.lon,
        ant_height_m=args.height,
        beam_func=beam_func,
        sky_func=GDSM_sky_model,
        beam_nside=args.beam_nside,
        sky_nside=args.sky_nside,
    )

    time_list, az_list = example_scan(
        az_s=args.az_start, az_e=args.az_end,
        dt=args.dt, n_repeats=args.n_repeats,
    )
    _log(f"[sim] {mpiutil.size} MPI rank(s)  ntime={len(time_list)}  "
         f"freq={args.freq_mhz:.1f} MHz  el={args.el:.1f}°")

    # Library handles the time-axis partition + noise broadcast internally
    # (since limTOD 1.2) — no manual harness needed.
    np.random.seed(args.seed)  # only rank 0 consumes this stream
    t0 = time.perf_counter()
    overall, sky, gain_noise, lst_deg = tod_sim.generate_TOD(
        freq_list=[args.freq_mhz],
        time_list=time_list,
        azimuth_deg_list=az_list,
        elevation_deg=args.el,
        start_time_utc=args.start_utc,
        white_noise_var=args.white_var,
        return_LSTs=True,
        truncate_frac_thres=args.truncate_frac,
    )
    wall = time.perf_counter() - t0

    # Only rank 0 writes. Every rank already holds the full result after
    # the library's internal allgather, so a broadcast would be redundant.
    if mpiutil.rank0:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Store TODs as a dense (ntods, ntime) array plus individual
        # scalar metadata fields so the cache can be loaded with plain
        # ``np.load`` (no object-array deserialisation needed).
        np.savez(
            args.out,
            TOD_group=np.asarray(overall[0], dtype=np.float64)[None, :],
            sky_TOD_group=np.asarray(sky[0], dtype=np.float64)[None, :],
            gain_noise_group=np.asarray(gain_noise[0],
                                        dtype=np.float64)[None, :],
            LST_deg_list_group=np.asarray(lst_deg,
                                          dtype=np.float64)[None, :],
            azimuth_deg_list_group=np.asarray(az_list,
                                              dtype=np.float64)[None, :],
            elevation_deg_list_group=(
                args.el * np.ones((1, len(time_list)), dtype=np.float64)
            ),
            freq_list=np.asarray([args.freq_mhz], dtype=np.float64),
            meta_az_start=np.float64(args.az_start),
            meta_az_end=np.float64(args.az_end),
            meta_el=np.float64(args.el),
            meta_dt=np.float64(args.dt),
            meta_n_repeats=np.int64(args.n_repeats),
            meta_start_utc=np.asarray(args.start_utc),
            meta_lat=np.float64(args.lat),
            meta_lon=np.float64(args.lon),
            meta_height=np.float64(args.height),
            meta_beam_fwhm=np.float64(args.beam_fwhm),
            meta_sky_nside=np.int64(args.sky_nside),
            meta_beam_nside=np.int64(args.beam_nside),
            meta_truncate_frac=np.float64(args.truncate_frac),
            meta_white_var=np.float64(args.white_var),
            meta_seed=np.int64(args.seed),
        )
        print(f"[sim] wrote {args.out}  ({len(time_list)} samples, "
              f"wall={wall:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
