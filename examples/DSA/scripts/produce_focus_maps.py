"""Produce the focused ns=64 HP-filter map figures used in README.

Three configurations, all with the same prior treatment ("autonoise"):
beam-smoothed truth as prior mean, prior_sigma = 1·std(sky_truth), and
per-segment noise variance estimated automatically by the Wiener filter
from the rolling residual.

| Config                                       | output                                              |
|---|---|
| MeerKLASS baseline (single el, n_repeats=13) | compare_focus_meerklass_baseline__autonoise_ns64_hp |
| Stop-and-stare (9 pointings)                 | compare_focus_steer_and_stare_baseline__autonoise_  |
| MeerKLASS cascade (5 el × n_repeats=3)       | compare_focus_meerklass_cascade__autonoise_ns64_hp  |

Output: figures/compare_focus_<kind><suffix>__autonoise_ns64_hp.{png,pdf}
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib as mpl
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DSA_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, DSA_DIR)

from compare_maps import (  # noqa: E402
    DEFAULT_WHITE_VAR, load_ops, load_tods, run_mapmaking, save_three_panel,
)


CASES = (
    ("meerklass",       "_baseline", "MeerKLASS baseline (single el)"),
    ("steer_and_stare", "_baseline", "Stop-and-stare"),
    ("meerklass",       "_cascade",  "MeerKLASS cascade (5 el)"),
)

PRIOR_MODES = (
    # Only the auto-noise variant is kept; previous prior comparisons
    # have been removed so the focused panels match the headline
    # result in the README.
    ("autonoise",   {"no_prior": False,
                     "prior_mean_mode": "smoothed",
                     "prior_sigma_factor": 1.0,
                     "auto_noise": True}),
)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
    })
    rms_table: list[tuple[str, str, str, float]] = []

    for kind, suffix, label in CASES:
        TOD_group = load_tods(kind, suffix=suffix)
        mm = load_ops(kind, 64, suffix=suffix)
        for prior_label, kwargs in PRIOR_MODES:
            t0 = time.time()
            sky_est, sky_truth, rms = run_mapmaking(
                mm, TOD_group, use_hp_filter=True,
                white_var=DEFAULT_WHITE_VAR, **kwargs,
            )
            wall = time.time() - t0
            # Distinct filename: focus_<kind><suffix>_ns64_hp_<prior>
            fname_suffix = f"{suffix}__{prior_label}"  # double underscore
            save_three_panel(
                kind=f"focus_{kind}", nside_target=64, use_hp_filter=True,
                sky_truth=sky_truth, sky_est=sky_est, mm=mm,
                rms_K=rms, wall_s=wall, fname_suffix=fname_suffix,
            )
            rms_table.append((label, prior_label, suffix, rms))

    # Plain-text summary on stdout
    print("\n=== Focused-maps RMS summary (ns=64, HP filter on) ===")
    print(f"{'config':<32} {'prior':<10} {'RMS [K]':>10}")
    for label, prior_label, suffix, rms in rms_table:
        print(f"{label:<32} {prior_label:<10} {rms:>10.3f}")


if __name__ == "__main__":
    main()
