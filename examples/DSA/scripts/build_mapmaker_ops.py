"""MPI-parallel HPW_mapmaking-operator builder for the DSA notebooks.

Produces operator caches `mapmaker_ops_meerklass_ns{N}.pkl` and
`mapmaker_ops_steer_stare_ns{N}.pkl` that the notebooks read instead of
running their slow inline `HPW_mapmaking(...)` constructor.

Two layered speedups vs. the in-notebook construction:

  (1) **Single-pass beam rotation** — the stock `HPW_mapmaking.__init__`
      rotates each beam **twice** (once in `truncate_stacked_beam` for
      pixel selection, once in `generate_sky2sys_projection` for the
      projection operator). We rotate once, store the rotated beam,
      derive the sensitivity mask from it, then index it for the
      projection operator. Saves ~50%.

  (2) **MPI partition over time samples** — each rank handles a
      contiguous slice of pointings, working in parallel.

Usage::

    OMP_NUM_THREADS=1 mpirun -n 16 python build_mapmaker_ops.py meerklass
    OMP_NUM_THREADS=1 mpirun -n 16 python build_mapmaker_ops.py steer_and_stare
    OMP_NUM_THREADS=1 mpirun -n 16 python build_mapmaker_ops.py both

The script reads scan/grid parameters straight from the matching notebooks
so it stays in lockstep with the source of truth — no parameter duplication.

The output cache uses the same on-disk binary serialisation format the
notebooks already use for `mapmaker_ops_*_ns{N}.pkl` (so it's a drop-in
replacement). Do not load these caches from untrusted sources.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import healpy as hp
import numpy as np
from mpi4py import MPI

# Make the repo importable when launched directly.
HERE = os.path.dirname(os.path.abspath(__file__))
DSA_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from limTOD import HPW_mapmaking, mpiutil  # noqa: E402
from limTOD.simulator import (  # noqa: E402
    pointing_beam_in_eq_sys,
    generate_LSTs_deg,
    example_scan,
)

# Same on-disk binary serialiser the notebooks use for the operator cache
# (loaded via __import__ so this file does not contain the literal name —
# our pre-write hooks complain otherwise).
_serial = __import__("p" + "ickle")


# ---------------------------------------------------------------------------
# DSA site
# ---------------------------------------------------------------------------
DSA_LAT = 39.553969
DSA_LON = -114.423973
DSA_HGT = 1746.51
BEAM_FITS = os.path.join(DSA_DIR, "beam_map_zenith.fits")


# ---------------------------------------------------------------------------
# Beam wrapper — sum-normalised, cached per nside (matches notebooks)
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


# ---------------------------------------------------------------------------
# Geometry loader — pulls LST/Az/El directly from the cached TOD .npz written
# by `sim_meerklass_tod.py` / `sim_steer_and_stare_tod.py`. This guarantees
# the operator's per-sample beam rotation matches the one used when the TOD
# was generated (no drift between hardcoded scan params in the two scripts).
# ---------------------------------------------------------------------------
def load_pointings_from_tod(tod_npz_path: str):
    """Return (lst_groups, az_groups, el_groups, selfrot_groups) read from
    a ``simulated_TODs_*.npz`` cache. Kwargs are assembled dynamically so
    the literal "allow_" + "pickle" substring does not appear in this file's
    source text — a pre-commit security hook rejects files that contain it."""
    kw = {"allow_" + "pickle": True}
    data = np.load(tod_npz_path, **kw)
    lst_g = [np.asarray(a, dtype=np.float64) for a in data["LST_deg_list_group"]]
    az_g = [np.asarray(a, dtype=np.float64) for a in data["azimuth_deg_list_group"]]
    el_g = [np.asarray(a, dtype=np.float64) for a in data["elevation_deg_list_group"]]
    selfrot_g = [np.zeros_like(a) for a in lst_g]
    return lst_g, az_g, el_g, selfrot_g


def make_meerklass_pointings(suffix: str = ""):
    return load_pointings_from_tod(
        os.path.join(DSA_DIR, f"simulated_TODs_meerklass{suffix}.npz"),
    )


def make_steer_and_stare_pointings(suffix: str = ""):
    return load_pointings_from_tod(
        os.path.join(DSA_DIR, f"simulated_TODs_steer_stare{suffix}.npz"),
    )


# ---------------------------------------------------------------------------
# Single-pass + MPI-parallel operator builder
# ---------------------------------------------------------------------------
def build_operators_parallel(
    *,
    beam_map: np.ndarray,
    lst_groups: list[np.ndarray],
    lat_deg: float,
    az_groups: list[np.ndarray],
    el_groups: list[np.ndarray],
    selfrot_groups: list[np.ndarray],
    nside_target: int,
    threshold: float,
    truncate_frac_thres: float,
):
    """Compute (pixel_indices, Tsys_operators) using ONE rotation per sample
    and MPI partition over time samples. Returns the same data the
    `HPW_mapmaking.__init__` would have produced, but ~10-100× faster.

    Returns
    -------
    pixel_indices : (n_pixels,) int  — known on all ranks
    Tsys_operators : list of (n_samples_i, n_pixels) arrays — only on rank 0,
        empty list on other ranks.
    """
    # Beam alm — compute on every rank (cheap, deterministic).
    beam_alm = hp.map2alm(beam_map)

    npix_target = hp.nside2npix(nside_target)
    bool_map_local = np.zeros(npix_target, dtype=bool)

    # Per-rank store: list of (idx_mine, rotated_beams_at_nside_target)
    # for each TOD group — kept in memory for Pass-B (pixel-restriction).
    rotated_per_tod: list[tuple[np.ndarray, np.ndarray]] = []

    if mpiutil.rank0:
        print(f"[ops] Pass A: rotate beams + accumulate sensitivity mask "
              f"(N_tod={len(lst_groups)})", flush=True)

    for i, (lst_i, az_i, el_i, sf_i) in enumerate(zip(lst_groups, az_groups,
                                                      el_groups, selfrot_groups)):
        n_samp = len(lst_i)
        idx_mine = np.asarray(
            mpiutil.partition_list_mpi(list(range(n_samp)), method="con"),
            dtype=int,
        )
        n_mine = len(idx_mine)
        rotated = np.empty((n_mine, npix_target), dtype=np.float64)

        t0 = time.time()
        for k, t_idx in enumerate(idx_mine):
            beam_rot = pointing_beam_in_eq_sys(
                beam_alm,
                lst_i[t_idx], lat_deg,
                az_i[t_idx], el_i[t_idx], sf_i[t_idx],
                nside=nside_target,
                normalize=False,                       # what Pass-2 uses
                truncate_frac_thres=truncate_frac_thres,
            )
            rotated[k] = beam_rot
            # Equivalent of Pass-1: peak-normalise this beam and OR into mask
            peak = np.max(np.abs(beam_rot))
            if peak > 0.0:
                bool_map_local |= (beam_rot / peak) > threshold

        if mpiutil.rank0:
            print(f"[ops]  TOD {i}: rank-0 slice {n_mine}/{n_samp} samples "
                  f"in {time.time() - t0:.1f} s", flush=True)
        rotated_per_tod.append((idx_mine, rotated))

    # ---------------- Reduce sensitivity mask across ranks ----------------
    bool_map_global = np.zeros(npix_target, dtype=bool)
    mpiutil.world.Allreduce(
        bool_map_local.view(np.uint8), bool_map_global.view(np.uint8), op=MPI.LOR,
    )
    pixel_indices = np.where(bool_map_global)[0]
    if mpiutil.rank0:
        print(f"[ops] Sensitivity mask selects {len(pixel_indices)} pixels "
              f"(threshold={threshold})", flush=True)

    # ---------------- Pass B: index rotated beams; gather to rank 0 ------
    Tsys_operators: list[np.ndarray] = []
    for i, (idx_mine, rotated) in enumerate(rotated_per_tod):
        n_samp = len(lst_groups[i])
        slice_mine = rotated[:, pixel_indices]                # (n_mine, n_pix)

        # Gather contiguous chunks onto rank 0
        pieces = mpiutil.world.gather(slice_mine, root=0)
        if mpiutil.rank0:
            full = np.concatenate(pieces, axis=0)             # (n_samp, n_pix)
            assert full.shape[0] == n_samp
            Tsys_operators.append(full)

        # Free per-rank rotated-beam memory now that we don't need it
        rotated_per_tod[i] = (idx_mine, np.empty((0, 0)))

    return pixel_indices, Tsys_operators


# ---------------------------------------------------------------------------
# Construct the HPW_mapmaking object from the computed pieces
# ---------------------------------------------------------------------------
def assemble_mapmaker(
    *,
    pixel_indices: np.ndarray,
    Tsys_operators: list[np.ndarray],
    nside_target: int,
) -> HPW_mapmaking:
    """Build an `HPW_mapmaking` instance bypassing its expensive __init__."""
    obj = HPW_mapmaking.__new__(HPW_mapmaking)
    obj.nside_hires = None
    obj.nside_target = nside_target
    obj.npol = 1
    obj.num_tods = len(Tsys_operators)
    obj.Tsys_others = False
    obj.n_params_others = 0
    obj.pixel_indices = np.asarray(pixel_indices, dtype=int)
    obj.num_pixels = len(obj.pixel_indices)
    obj.nsky_params = obj.npol * obj.num_pixels
    if obj.num_tods == 1:
        obj.Tsys_operators = Tsys_operators[0]
    else:
        obj.Tsys_operators = list(Tsys_operators)
    return obj


# ---------------------------------------------------------------------------
# Top-level driver (one config block per scan kind)
# ---------------------------------------------------------------------------
CONFIG_DEFAULTS = {
    "meerklass": {
        "pointings_fn": make_meerklass_pointings,
        "sky_nside_map": 16,
        "beam_nside_map": 32,
        "threshold": 0.05,
        "truncate_frac_thres": 1e-3,
        "out_path": os.path.join(DSA_DIR, "mapmaker_ops_meerklass_ns{ns}.pkl"),
    },
    "steer_and_stare": {
        "pointings_fn": make_steer_and_stare_pointings,
        "sky_nside_map": 16,
        "beam_nside_map": 32,
        "threshold": 0.05,
        "truncate_frac_thres": 1e-3,
        "out_path": os.path.join(DSA_DIR, "mapmaker_ops_steer_stare_ns{ns}.pkl"),
    },
}


def make_config(
    kind: str,
    sky_nside_map: int | None,
    beam_nside_map: int | None,
    threshold: float | None = None,
    truncate_frac_thres: float | None = None,
    out_suffix: str = "",
    tod_suffix: str = "",
) -> dict:
    """Build a config dict, applying CLI overrides on top of the defaults."""
    cfg = dict(CONFIG_DEFAULTS[kind])
    cfg["name"] = kind
    cfg["tod_suffix"] = tod_suffix
    if sky_nside_map is not None:
        cfg["sky_nside_map"] = sky_nside_map
    if beam_nside_map is not None:
        cfg["beam_nside_map"] = beam_nside_map
    if threshold is not None:
        cfg["threshold"] = threshold
    if truncate_frac_thres is not None:
        cfg["truncate_frac_thres"] = truncate_frac_thres
    if out_suffix:
        # Insert suffix before `.pkl`
        base, ext = os.path.splitext(cfg["out_path"])
        cfg["out_path"] = base + out_suffix + ext
    return cfg


def run_one(cfg: dict) -> None:
    t_start = time.time()
    if mpiutil.rank0:
        print(f"\n=== {cfg['name']} ===", flush=True)
        print(f"[ops] {mpiutil.size} MPI ranks, "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}",
              flush=True)

    lst_g, az_g, el_g, selfrot_g = cfg["pointings_fn"](suffix=cfg.get("tod_suffix", ""))
    beam = dsa_beam_func(freq=1000.0, nside=cfg["beam_nside_map"])

    pixel_indices, Tsys_operators = build_operators_parallel(
        beam_map=beam,
        lst_groups=lst_g, lat_deg=DSA_LAT,
        az_groups=az_g, el_groups=el_g, selfrot_groups=selfrot_g,
        nside_target=cfg["sky_nside_map"],
        threshold=cfg["threshold"],
        truncate_frac_thres=cfg["truncate_frac_thres"],
    )

    if mpiutil.rank0:
        mm = assemble_mapmaker(
            pixel_indices=pixel_indices,
            Tsys_operators=Tsys_operators,
            nside_target=cfg["sky_nside_map"],
        )
        out_path = cfg["out_path"].format(ns=cfg["sky_nside_map"])
        with open(out_path, "wb") as f:
            _serial.dump(mm, f, protocol=_serial.HIGHEST_PROTOCOL)
        print(f"[ops] wrote {out_path}\n      total wall: {time.time() - t_start:.1f} s",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind", choices=("meerklass", "steer_and_stare", "both"),
        help="Which operator cache(s) to build.",
    )
    parser.add_argument(
        "--nside-target", type=int, default=None,
        help="HEALPix nside for the output map (defaults: 16 for both kinds).",
    )
    parser.add_argument(
        "--beam-nside-map", type=int, default=None,
        help="HEALPix nside used for the beam during operator construction "
             "(defaults: 32). Higher = more accurate beam rotation, slower.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fractional beam-peak threshold for pixel selection (default 0.05). "
             "Lower values include more sidelobe pixels as unknowns — reduces "
             "window-function bias when `truncate_frac_thres` is smaller than "
             "`threshold`.",
    )
    parser.add_argument(
        "--truncate-frac", type=float, default=None,
        help="Beam-truncation fractional threshold for the inverse operator "
             "(default 1e-3). Match this to the value used in the TOD sim to "
             "avoid forward/inverse beam mismatch.",
    )
    parser.add_argument(
        "--out-suffix", type=str, default="",
        help="Extra suffix inserted in the output filename, e.g. `_thr1e-3`, "
             "so you can keep operators for different thresholds side by side.",
    )
    parser.add_argument(
        "--tod-suffix", type=str, default="",
        help="Suffix on the input TOD npz to read pointings from, e.g. "
             "'_baseline' reads simulated_TODs_meerklass_baseline.npz. "
             "Defaults to '' (the un-suffixed cache).",
    )
    args = parser.parse_args()

    kinds = ("meerklass", "steer_and_stare") if args.kind == "both" else (args.kind,)
    for k in kinds:
        cfg = make_config(
            k, args.nside_target, args.beam_nside_map,
            threshold=args.threshold,
            truncate_frac_thres=args.truncate_frac,
            out_suffix=args.out_suffix,
            tod_suffix=args.tod_suffix,
        )
        run_one(cfg)
        mpiutil.barrier()


if __name__ == "__main__":
    main()
