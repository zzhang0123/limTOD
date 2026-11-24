"""Time a synthetic patch-beam raster, including beam, sky and noise generation.

From the checkout, run PYTHONPATH=. python scripts/benchmark_patchbeam.py --repeats 5.
Use the same native thread settings and Python environment across revisions.
"""

import argparse
from pathlib import Path
from time import perf_counter

import healpy as hp
import numpy as np
from astropy.utils import iers

from limTOD import example_scan
from limTOD.patchbeam import PatchBeamTODSim, synthetic_gaussian_beam


def run_scan():
    """Return one complete seeded scan and the time spent in generate_TOD."""
    beam_freqs = np.arange(650.0, 810.0, 10.0)
    beam = synthetic_gaussian_beam(
        freq_MHz=beam_freqs,
        margin_deg=np.linspace(-6.0, 6.0, 501),
        fwhm_deg=1.8 * 750.0 / beam_freqs,
        polarizations=("HH",),
    )
    sky_nside = 128
    x, y, z = hp.pix2vec(sky_nside, np.arange(hp.nside2npix(sky_nside)))
    sky_750 = 40.0 + 7.0 * x - 3.0 * y + 5.0 * x * z

    def sky_func(*, freq, nside):
        return sky_750 * (freq / 750.0) ** -2.6

    simulator = PatchBeamTODSim(beam=beam, sky_func=sky_func, sky_nside=sky_nside)
    seconds, azimuth = example_scan(n_repeats=1)
    np.random.seed(2026)
    start = perf_counter()
    result = simulator.generate_TOD(
        freq_list=beam_freqs[::2],
        time_list=seconds,
        azimuth_deg_list=azimuth,
        elevation_deg=41.5,
        return_LSTs=True,
    )
    return result, perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="Timed scans after one warm-up.")
    parser.add_argument("--save-output", type=Path, help="Save the last scan as a NumPy NPZ.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    total_times = []
    scan_times = []
    with iers.conf.set_temp("auto_download", False):
        run_scan()
        for _ in range(args.repeats):
            start = perf_counter()
            result, scan_time = run_scan()
            total_times.append(perf_counter() - start)
            scan_times.append(scan_time)

    overall, sky, gain, lst = result
    print(f"{sky.shape[0]} channels, {sky.shape[1]} samples, {args.repeats} timed runs")
    for label, times in (("Setup + generate_TOD", total_times), ("generate_TOD", scan_times)):
        print(f"{label}: median {np.median(times):.4f} s (range {min(times):.4f}–{max(times):.4f} s)")
    if args.save_output is not None:
        np.savez(args.save_output, overall=overall, sky=sky, gain=gain, lst=lst)


if __name__ == "__main__":
    main()
