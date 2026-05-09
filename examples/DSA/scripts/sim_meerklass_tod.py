"""MPI-parallel TOD simulation for the DSA azimuth-scan (MeerKLASS-type) mode.

Generates the *setting* and *rising* azimuth-scan TODs at nside=128 with
beam truncation 1e-3, then writes them to ``../simulated_TODs_meerklass.npz``
in the same format the matching notebook expects (so the notebook simply
loads the cache and skips its own slow simulation).

Run with one of::

    OMP_NUM_THREADS=2 python sim_meerklass_tod.py            # serial smoke test
    OMP_NUM_THREADS=1 mpirun -n 8 python sim_meerklass_tod.py  # parallel

Speedup design:
  - The dominant cost is `hp.rotate_alm` + `hp.alm2map` per time sample.
  - Each scan's time-sample axis is partitioned contiguously across MPI
    ranks; each rank calls ``TODSim.generate_TOD`` on its slice.
  - 1/f gain noise is generated *globally* on rank 0 first (so its
    inter-sample correlation structure is preserved), then broadcast
    and sliced per rank — the actual rotate-and-convolve work is done
    in parallel.
  - The per-rank TOD slices are gathered onto rank 0 and saved.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import healpy as hp
import numpy as np

# Make the repo importable when the script is launched directly.
HERE = os.path.dirname(os.path.abspath(__file__))
DSA_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from limTOD import example_scan, GDSM_sky_model, mpiutil  # noqa: E402
from limTOD.flicker_model import sim_noise  # noqa: E402
from limTOD.simulator import generate_TOD_sky, generate_LSTs_deg  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — matches dsa_meerklass_scan.ipynb (cells c9d0e1f2 / d3e4f5a6)
# ---------------------------------------------------------------------------
DSA_LAT = 39.553969
DSA_LON = -114.423973
DSA_HGT = 1746.51

EL_SCAN = 55.0  # degrees
FREQ_MHZ = 1000.0  # single frequency
SKY_NSIDE = 64
BEAM_NSIDE = 64
TRUNC_FRAC_THRES = 1e-3
DT = 2.0  # seconds per sample
N_REPEATS = 13

# 1/f gain-noise parameters (limTOD defaults from generate_TOD)
GAIN_F0 = 1.335e-5
GAIN_FC = 1.099e-3
GAIN_ALPHA = 2
WHITE_NOISE_VAR = 2.5e-6  # fractional

SCANS = (
    {"name": "setting", "az_s": -60.3, "az_e": -42.3,
     "start": "2024-04-15 08:25:05"},
    {"name": "rising",  "az_s":  42.3, "az_e":  60.3,
     "start": "2024-04-15 02:03:50"},
)

OUT_PATH = os.path.join(DSA_DIR, "simulated_TODs_meerklass.npz")
BEAM_FITS = os.path.join(DSA_DIR, "beam_map_zenith.fits")


# ---------------------------------------------------------------------------
# Beam wrapper — sum-normalised, cached per nside (matches notebook)
# ---------------------------------------------------------------------------
_beam_cache: dict[int, np.ndarray] = {}


def dsa_beam_func(*, freq, nside):
    if nside not in _beam_cache:
        beam = hp.read_map(BEAM_FITS)
        if hp.get_nside(beam) != nside:
            beam = hp.ud_grade(beam, nside)
        _beam_cache[nside] = beam
    out = _beam_cache[nside].copy()
    return out / out.sum()


def load_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Load the (sum-normalised) beam and the GDSM sky once per rank."""
    beam = dsa_beam_func(freq=FREQ_MHZ, nside=BEAM_NSIDE)
    sky = GDSM_sky_model(freq=FREQ_MHZ, nside=SKY_NSIDE)
    return beam, sky


# ---------------------------------------------------------------------------
# MPI gather helper for contiguous-chunk per-rank arrays
# ---------------------------------------------------------------------------
def gather_contiguous(arr_mine: np.ndarray, n_total: int) -> np.ndarray | None:
    """Gather contiguous per-rank slices of a 1-D float array onto rank 0.

    Returns the assembled array on rank 0, ``None`` on other ranks.
    Uses ``comm.gather`` with default Python serialisation — fine for our
    O(few-thousand)-sized arrays. For larger workloads, ``Gatherv`` would
    be more efficient.
    """
    arr_mine = np.ascontiguousarray(arr_mine, dtype=np.float64)
    if mpiutil.size == 1:
        assert len(arr_mine) == n_total
        return arr_mine
    pieces = mpiutil.world.gather(arr_mine, root=0)
    if mpiutil.rank0:
        out = np.concatenate(pieces, axis=0)
        assert out.shape == (n_total,), (
            f"Gathered array has shape {out.shape}, expected ({n_total},). "
            "Did all ranks produce contiguous, in-order slices?"
        )
        return out
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--elevations", nargs="+", type=float, default=[EL_SCAN],
        metavar="DEG",
        help="Elevation(s) in degrees. Pass multiple values to run an "
             "overlap-in-Dec cascade, e.g. --elevations 53 54 55 56 57 "
             "(default: single elevation = 55°).",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=N_REPEATS,
        help="Number of back-and-forth example_scan repeats per (el, scan). "
             "Lower this when using multiple elevations so total integration "
             "time stays reasonable (e.g. --n-repeats 3 for a 5-el cascade).",
    )
    parser.add_argument(
        "--truncate-frac", type=float, default=TRUNC_FRAC_THRES,
        help="Beam-truncation fractional threshold (default 1e-3). Pixels "
             "below this fraction of the rotated-beam peak are zeroed.",
    )
    parser.add_argument(
        "--no-noise", action="store_true",
        help="Disable both 1/f gain noise and white noise. Output TOD = "
             "pure beam-convolved sky.",
    )
    parser.add_argument(
        "--gain-f0", type=float, default=GAIN_F0,
        help="1/f gain-noise amplitude at the knee frequency. Default "
             f"{GAIN_F0:.3e} (limTOD reference). Lower values tune down the "
             "1/f drift; the per-sample variance scales linearly with f0.",
    )
    parser.add_argument(
        "--white-var", type=float, default=WHITE_NOISE_VAR,
        help=f"White-noise variance (fractional). Default {WHITE_NOISE_VAR:.1e}. "
             "Set to 0 to isolate the 1/f contribution; lower values to study "
             "the noise-floor scaling.",
    )
    parser.add_argument(
        "--out-suffix", type=str, default="",
        help="Suffix inserted before .npz in the output filename, e.g. "
             "'_baseline' writes simulated_TODs_meerklass_baseline.npz. "
             "Lets multiple analyses coexist on disk without overwriting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start_total = time.time()
    if mpiutil.rank0:
        print(
            f"[sim] {mpiutil.size} MPI ranks, "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}\n"
            f"[sim] elevations={args.elevations}  n_repeats={args.n_repeats}  "
            f"trunc={args.truncate_frac:.0e}  noise={'OFF' if args.no_noise else 'on'}"
            f"  f0={args.gain_f0:.3e}  white_var={args.white_var:.3e}",
            flush=True,
        )

    beam, sky = load_inputs()

    TOD_group: list[np.ndarray] = []
    LST_group: list[np.ndarray] = []
    az_group: list[np.ndarray] = []
    el_group: list[np.ndarray] = []

    # Iterate over the (el, scan) cascade. Each pair gets its own RNG seed
    # so adjacent cascaded scans carry independent 1/f + white-noise
    # realisations (otherwise all 10 TODs would be pointwise correlated).
    for el_idx, el_deg in enumerate(args.elevations):
        for scan_idx, scan in enumerate(SCANS):
            rng_seed = 42 + 100 * el_idx + scan_idx

            time_full, az_full = example_scan(
                az_s=scan["az_s"], az_e=scan["az_e"],
                dt=DT, n_repeats=args.n_repeats,
            )
            n_samp = len(time_full)

            # FULL LST array (cheap, deterministic) on every rank.
            lst_full = generate_LSTs_deg(
                DSA_LAT, DSA_LON, DSA_HGT, time_full, scan["start"],
            )

            # --- Partition contiguous chunks of TIME indices across ranks ---
            idx_mine = np.asarray(
                mpiutil.partition_list_mpi(list(range(n_samp)), method="con"),
                dtype=int,
            )

            # --- FULL coherent 1/f gain noise + white noise on rank 0,
            #     broadcast to all ranks, then sliced by idx_mine. ---
            if args.no_noise:
                gain_noise_mine = np.zeros(len(idx_mine), dtype=np.float64)
                white_noise_mine = np.zeros(len(idx_mine), dtype=np.float64)
            else:
                gain_noise_full = None
                white_noise_full = None
                if mpiutil.rank0:
                    # Note: sim_noise's `white_n_variance` arg adds white noise
                    # to the SAME 1/f process internally — set to 0 here so we
                    # only get the 1/f component, then add white separately so
                    # users can independently tune both via CLI.
                    gain_noise_full = sim_noise(
                        f0=args.gain_f0, fc=GAIN_FC, alpha=GAIN_ALPHA,
                        time_list=time_full, n_samples=1,
                        white_n_variance=0.0,
                    )[0]
                    rng = np.random.default_rng(rng_seed)
                    white_noise_full = rng.normal(
                        0.0, np.sqrt(args.white_var), size=n_samp,
                    ) if args.white_var > 0 else np.zeros(n_samp)
                gain_noise_full = mpiutil.world.bcast(gain_noise_full, root=0)
                white_noise_full = mpiutil.world.bcast(white_noise_full, root=0)
                gain_noise_mine = gain_noise_full[idx_mine]
                white_noise_mine = white_noise_full[idx_mine]

            # --- Each rank's sky_TOD via the LOW-LEVEL generate_TOD_sky ---
            t_rank = time.time()
            sky_tod_mine = generate_TOD_sky(
                beam, sky,
                lst_full[idx_mine], DSA_LAT,
                az_full[idx_mine],
                np.full(len(idx_mine), el_deg, dtype=np.float64),
                np.zeros(len(idx_mine), dtype=np.float64),
                normalize_beam=False,
                truncate_frac_thres=args.truncate_frac,
            )

            # Noise-dressing: overall = (1 + gain_noise) * sky * (1 + white_noise)
            overall_tod_mine = (
                (1.0 + gain_noise_mine) * sky_tod_mine * (1.0 + white_noise_mine)
            )

            if mpiutil.rank0:
                print(
                    f"[sim] el={el_deg:.1f}° {scan['name']}: rank-0 slice "
                    f"{len(idx_mine)}/{n_samp} samples in "
                    f"{time.time() - t_rank:.1f} s",
                    flush=True,
                )

            # --- Gather per-rank slices to rank 0 ---
            tod_full = gather_contiguous(overall_tod_mine, n_samp)
            if mpiutil.rank0:
                TOD_group.append(tod_full)
                LST_group.append(lst_full)
                az_group.append(np.asarray(az_full, dtype=np.float64))
                el_group.append(np.full(n_samp, el_deg, dtype=np.float64))

            mpiutil.barrier()

    # --- Rank 0 saves the .npz in the format the notebook expects ---
    if mpiutil.rank0:
        out_path = OUT_PATH
        if args.out_suffix:
            base, ext = os.path.splitext(OUT_PATH)
            out_path = base + args.out_suffix + ext
        np.savez(
            out_path,
            TOD_group=np.array(TOD_group, dtype=object),
            LST_deg_list_group=np.array(LST_group, dtype=object),
            azimuth_deg_list_group=np.array(az_group, dtype=object),
            elevation_deg_list_group=np.array(el_group, dtype=object),
            freq_list=np.array([FREQ_MHZ]),
        )
        n_tods = len(TOD_group)
        total_samples = int(sum(len(t) for t in TOD_group))
        print(
            f"[sim] wrote {out_path}\n"
            f"      {n_tods} TODs, {total_samples} samples total, "
            f"wall = {time.time() - t_start_total:.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
