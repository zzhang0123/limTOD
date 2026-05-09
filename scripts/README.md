# limTOD reference scripts

Two standalone reference drivers that exercise the full limTOD pipeline
end-to-end. They demonstrate the intended usage pattern for the
library and serve as starting points for project-specific scripts.

| Script | What it does |
|---|---|
| [`simulate_tod.py`](./simulate_tod.py) | Generate a TOD cache (sky + 1/f gain noise + multiplicative white noise) for a single back-and-forth azimuth scan at constant elevation. Writes an `.npz` the map-making script can consume. |
| [`make_map.py`](./make_map.py) | Reconstruct a sky patch from that cache with the HPW (high-pass + Wiener filter) solver. Writes an `.npz` + quick-look PNG. |

Both scripts are deliberately thin wrappers around library calls — the
heavy lifting (beam rotation, convolution, 1/f noise, Wiener filter) is
inside `limTOD.TODSim.generate_TOD` and `limTOD.HPW_mapmaking`.

For the project-specific DSA pipeline (full MPI harness, cached
operators, custom beam FITS, multi-elevation cascade, etc.) see
[`examples/DSA/scripts/`](../examples/DSA/scripts/).

## Prerequisites

- A conda environment with `limTOD` installed (editable install is
  fine); the repo uses the `TOD` env in the author's setup.
- `mpi4py` is a runtime dependency of limTOD; you only need `mpirun`
  if you want to run in parallel.

## Quick start (serial)

```bash
# 1. Simulate
python scripts/simulate_tod.py \
    --out /tmp/tod.npz \
    --el 45 --n-repeats 13 --beam-fwhm 1.1

# 2. Map-make
python scripts/make_map.py \
    --tod /tmp/tod.npz \
    --out /tmp/map \
    --nside 64 --hp-cutoff 0.03
```

The map-making stage prints the residual RMS and writes
`/tmp/map.npz` + `/tmp/map.png`.

## Parallel simulation (MPI)

Since limTOD 1.2 the simulator is MPI-aware by default — contiguous
time-axis partition + `allgather`. To scale out you just launch with
`mpirun`; no code changes required:

```bash
env OMP_NUM_THREADS=1 mpirun -n 4 \
    python scripts/simulate_tod.py --out /tmp/tod.npz --n-repeats 13
```

Setting `OMP_NUM_THREADS=1` prevents healpy's internal OpenMP from
oversubscribing the cores you already pinned to MPI ranks. Serial and
MPI runs produce bit-identical output given the same `--seed`.

The map-making stage is intrinsically serial and not worth MPI-ifying
at the nside=64 scale these scripts target; run it without `mpirun`.

## Tunable parameters

`simulate_tod.py`:

| Flag | Default | Effect |
|---|---|---|
| `--freq-mhz` | 1000 | Observing frequency. |
| `--az-start` / `--az-end` | -60, -42 | Azimuth sweep range (MeerKLASS-style). |
| `--el` | 45 | Constant elevation. |
| `--n-repeats` | 13 | Number of back-and-forth sweeps (~95 min at dt=2 s). |
| `--dt` | 2.0 | Sample interval in seconds. |
| `--beam-fwhm` | 1.1 | Gaussian beam FWHM (uses `example_symm_beam_map`). |
| `--sky-nside` / `--beam-nside` | 64 / 64 | HEALPix resolutions. |
| `--truncate-frac` | 1e-3 | Beam-edge truncation (0.1% of peak). |
| `--white-var` | 1e-7 | Multiplicative white-noise variance. |
| `--seed` | 0 | Rank-0 RNG seed (deterministic under MPI). |

`make_map.py`:

| Flag | Default | Effect |
|---|---|---|
| `--nside` | 64 | Output map resolution. |
| `--hp-cutoff` | 0.03 | HP filter cutoff in Hz. Lower = more 1/f leakage, less signal loss. |
| `--filter-order` | 4 | Butterworth order. |
| `--threshold` | 0.05 | Pixel selection threshold (5% of beam peak). |
| `--prior-sigma-factor` | 1.0 | Prior strength in σ(sky). Set to 0 to disable. |
| `--auto-noise` | off | Let the Wiener filter auto-estimate noise variance. |

## Output schema

`simulate_tod.py` writes one `.npz` file containing:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `TOD_group` | `(1, ntime)` float64 | Overall TOD: sky × (1 + gain) × (1 + white). |
| `sky_TOD_group` | `(1, ntime)` float64 | Noise-free sky TOD (for diagnostics). |
| `gain_noise_group` | `(1, ntime)` float64 | 1/f gain-noise sequence. |
| `LST_deg_list_group` | `(1, ntime)` float64 | Local sidereal time per sample. |
| `azimuth_deg_list_group` | `(1, ntime)` float64 | Azimuth pointing. |
| `elevation_deg_list_group` | `(1, ntime)` float64 | Elevation pointing. |
| `freq_list` | `(1,)` float64 | Frequency in MHz. |
| `meta_*` | scalar | All CLI args preserved as float/int/str scalars. |

All entries are plain numeric arrays — the cache loads with the default
`np.load(...)` without any object-array deserialisation.

`make_map.py` writes `<out>.npz` with the recovered map and a side-by-side
truth/recovered/residual PNG at `<out>.png`.
