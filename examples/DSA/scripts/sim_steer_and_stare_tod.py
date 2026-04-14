"""MPI-parallel TOD simulation for the DSA stop-and-stare strategy.

Generates the 9 stop-and-stare pointings (3x3 hex grid centred on
(RA=180°, Dec=52°)) at nside=128 / beam-truncate 1e-3, with the same
1/f gain noise model as the matching notebook. Writes
``../simulated_TODs_steer_stare.npz`` in the format the notebook's
TOD-load cell expects.

Run with one of::

    OMP_NUM_THREADS=2 python sim_steer_and_stare_tod.py             # serial
    OMP_NUM_THREADS=1 mpirun -n 9 python sim_steer_and_stare_tod.py   # parallel

Parallelism design: distinct pointings are physically independent (each has
its own start time and 21-min duration; the 1/f-noise correlation length
~1000 s is shorter than the 21 min between adjacent pointings). So we
partition over **pointings** rather than within-pointing time samples,
which sidesteps the cross-rank noise-correlation gather we needed for the
MeerKLASS scan. Each rank handles its assigned pointings serially.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import healpy as hp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DSA_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from astropy.coordinates import SkyCoord, AltAz, EarthLocation  # noqa: E402
from astropy.time import Time, TimeDelta  # noqa: E402
import astropy.units as u  # noqa: E402

from limTOD import GDSM_sky_model, mpiutil  # noqa: E402
from limTOD.flicker_model import sim_noise  # noqa: E402
from limTOD.simulator import generate_TOD_sky, generate_LSTs_deg  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — matches dsa_steer_and_stare.ipynb
# ---------------------------------------------------------------------------
DSA_LAT = 39.553969
DSA_LON = -114.423973
DSA_HGT = 1746.51

FREQ_MHZ = 1000.0
SKY_NSIDE = 64
BEAM_NSIDE = 64
TRUNC_FRAC_THRES = 1e-3
DT = 2.0
T_PER_POINTING = 21 * 60  # seconds
BASE_START_UTC = "2024-04-15 04:00:00"

# Hex pointing grid (3x3) centred on (RA=180, Dec=52)
CENTER_RA, CENTER_DEC = 180.0, 52.0
D_DEC = 3.5
D_RA = D_DEC / np.cos(np.radians(CENTER_DEC))

# 1/f gain-noise + white-noise parameters
GAIN_F0 = 1.335e-5
GAIN_FC = 1.099e-3
GAIN_ALPHA = 2
WHITE_NOISE_VAR = 2.5e-6

OUT_PATH = os.path.join(DSA_DIR, "simulated_TODs_steer_stare.npz")
BEAM_FITS = os.path.join(DSA_DIR, "beam_map_zenith.fits")


# ---------------------------------------------------------------------------
# Beam wrapper — sum-normalised
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


def make_pointings() -> list[tuple[float, float]]:
    """3x3 hex grid (RA, Dec)."""
    ps: list[tuple[float, float]] = []
    for i_row, dec_off in enumerate([-D_DEC, 0.0, D_DEC]):
        ra_shift = D_RA / 2 if i_row % 2 == 1 else 0.0
        for ra_off in [-D_RA, 0.0, D_RA]:
            ps.append((CENTER_RA + ra_off + ra_shift, CENTER_DEC + dec_off))
    return ps


def radec_to_azel(ra_deg, dec_deg, t_list_sec, start_utc, location):
    start = Time(start_utc)
    times = start + TimeDelta(t_list_sec, format="sec")
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz = target.transform_to(AltAz(obstime=times, location=location))
    return altaz.az.deg, altaz.alt.deg


def simulate_one_pointing(
    ra: float, dec: float, idx: int, beam, sky, location, *,
    gain_f0: float, white_var: float, no_noise: bool,
):
    """Simulate the TOD for a single stop-and-stare pointing.

    Returns (overall_tod, lst_arr, az_arr, el_arr) — all length T_PER_POINTING/DT.
    Noise is seeded deterministically by the pointing index so the result
    is reproducible across rank counts.
    """
    start_i = (Time(BASE_START_UTC) + TimeDelta(idx * T_PER_POINTING, format="sec")).iso
    t_list = np.arange(0.0, T_PER_POINTING, DT)
    az, el = radec_to_azel(ra, dec, t_list, start_i, location)
    lst = generate_LSTs_deg(DSA_LAT, DSA_LON, DSA_HGT, t_list, start_i)

    if no_noise:
        gain_noise = np.zeros_like(t_list)
        white = np.zeros_like(t_list)
    else:
        # Coherent 1/f noise per pointing (independent across pointings —
        # adjacent pointings are 21 min apart, longer than the 1/f
        # correlation length 1/fc). Set internal white_n_variance=0 here so
        # we control the white component separately.
        gain_noise = sim_noise(
            f0=gain_f0, fc=GAIN_FC, alpha=GAIN_ALPHA,
            time_list=t_list, n_samples=1, white_n_variance=0.0,
        )[0]
        rng = np.random.default_rng(seed=42 + idx)
        white = rng.normal(0.0, np.sqrt(white_var), size=len(t_list)) \
            if white_var > 0 else np.zeros_like(t_list)

    sky_tod = generate_TOD_sky(
        beam, sky, lst, DSA_LAT,
        az.astype(np.float64), el.astype(np.float64),
        np.zeros_like(t_list, dtype=np.float64),
        normalize_beam=False,
        truncate_frac_thres=TRUNC_FRAC_THRES,
    )
    overall = (1.0 + gain_noise) * sky_tod * (1.0 + white)
    return (
        np.asarray(overall, dtype=np.float64),
        np.asarray(lst, dtype=np.float64),
        np.asarray(az, dtype=np.float64),
        np.asarray(el, dtype=np.float64),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain-f0", type=float, default=GAIN_F0,
                        help=f"1/f gain-noise amplitude (default {GAIN_F0:.3e}).")
    parser.add_argument("--white-var", type=float, default=WHITE_NOISE_VAR,
                        help=f"White-noise variance (fractional, default {WHITE_NOISE_VAR:.1e}).")
    parser.add_argument("--no-noise", action="store_true",
                        help="Disable both 1/f and white noise (output = pure beam-convolved sky).")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix inserted before .npz in the output filename, "
                             "e.g. '_baseline'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.time()
    if mpiutil.rank0:
        print(
            f"[sim-ss] {mpiutil.size} MPI ranks, "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}\n"
            f"[sim-ss] noise={'OFF' if args.no_noise else 'on'}  "
            f"f0={args.gain_f0:.3e}  white_var={args.white_var:.3e}",
            flush=True,
        )

    location = EarthLocation(
        lat=DSA_LAT * u.deg, lon=DSA_LON * u.deg, height=DSA_HGT * u.m,
    )
    beam = dsa_beam_func(freq=FREQ_MHZ, nside=BEAM_NSIDE)
    sky = GDSM_sky_model(freq=FREQ_MHZ, nside=SKY_NSIDE)
    pointings = make_pointings()

    # --- Distribute pointings across ranks ---
    indices = list(range(len(pointings)))
    my_indices = mpiutil.partition_list_mpi(indices, method="con")

    # Each rank simulates its assigned pointings
    my_results: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for idx in my_indices:
        ra, dec = pointings[idx]
        t0 = time.time()
        tod, lst, az, el = simulate_one_pointing(
            ra, dec, idx, beam, sky, location,
            gain_f0=args.gain_f0, white_var=args.white_var, no_noise=args.no_noise,
        )
        if mpiutil.rank0:
            print(
                f"[sim-ss] PC{idx}: rank-{mpiutil.rank} (RA={ra:.1f}, Dec={dec:.1f}) "
                f"in {time.time() - t0:.1f} s",
                flush=True,
            )
        my_results.append((idx, tod, lst, az, el))

    # --- Gather pointings on rank 0 ---
    all_results = mpiutil.world.gather(my_results, root=0)
    if mpiutil.rank0:
        flat: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for chunk in all_results:
            flat.extend(chunk)
        flat.sort(key=lambda x: x[0])
        TOD_group = [r[1] for r in flat]
        LST_group = [r[2] for r in flat]
        az_group = [r[3] for r in flat]
        el_group = [r[4] for r in flat]

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
        print(
            f"[sim-ss] wrote {out_path}\n"
            f"          total wall time: {time.time() - t_total:.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
