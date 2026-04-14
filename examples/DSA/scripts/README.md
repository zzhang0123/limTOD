# DSA simulation + map-making scripts

## Organised analyses

Two end-to-end driver scripts run the analyses described in
[../README.md](../README.md):

- [run_analysis_A.sh](run_analysis_A.sh) — Single-elevation survey
  comparison: meerklass vs steer-and-stare at fixed integration time.
- [run_analysis_B.sh](run_analysis_B.sh) — MeerKLASS elevation cascade
  for sub-beam recovery (5 elevations × 3 repeats × 2 scans = 10 TODs).

Both default to `WHITE_VAR=1e-7`, `GAIN_F0=1.335e-7` (low-noise regime,
see `../README.md` for rationale). Suffix-based filenames
(`*_baseline.npz` / `*_cascade.npz` / `*_baseline.pkl` / `*_cascade.pkl`)
let the two analyses coexist on disk and be re-run independently.

Override defaults via env vars: `WHITE_VAR=2.5e-6 NRANKS=8 ./run_analysis_A.sh`.

## CLI flags (added with the analysis presets)

| Script | New flag | Purpose |
|---|---|---|
| `sim_meerklass_tod.py` | `--out-suffix STR` | Write to `simulated_TODs_meerklass{suffix}.npz`. |
| `sim_steer_and_stare_tod.py` | `--gain-f0 FLOAT` `--white-var FLOAT` `--no-noise` `--out-suffix STR` | Full noise-knob parity with `sim_meerklass_tod.py`. Previously hardcoded. |
| `build_mapmaker_ops.py` | `--tod-suffix STR` | Read pointings from `simulated_TODs_*{suffix}.npz`. Pairs with `--out-suffix`. |
| `compare_maps.py` | `--analysis {A,B}` `--suffix STR` `--white-var FLOAT` | Pick preset analysis, choose cache suffix, pass *known* noise variance to the Wiener filter. |
| `compare_maps.py` (default) | unchanged | Backwards-compatible: no flags → reads un-suffixed cache, runs full nside sweep. |

The `--white-var` flag is required to be consistent with the value used
in `sim_*.py`; otherwise the Wiener-filter noise weighting will be wrong
and recovery quality will degrade.



Standalone, MPI-parallel scripts for the DSA survey-strategy notebooks.
The notebooks ([dsa_meerklass_scan.ipynb](../dsa_meerklass_scan.ipynb), etc.)
call the same code in-line, but in-notebook execution is bottlenecked on a
single Python process (~30-40 min at nside=128 for the TOD step alone, and
~60-90 min for the map-making operator construction). These scripts split
the work across MPI ranks and write `.npz` / cache files the notebooks
then load instantly.

Two scripts are provided:

| Script | Builds | Notebook cache it replaces |
|---|---|---|
| [sim_meerklass_tod.py](sim_meerklass_tod.py) | TODs for the azimuth-scan strategy | `../simulated_TODs_meerklass.npz` |
| [build_mapmaker_ops.py](build_mapmaker_ops.py) | HPW_mapmaking operator (pixel selection + projection) | `../mapmaker_ops_*_ns{N}.pkl` |

Together they reduce the notebook end-to-end time from ~hours to **~1 minute**.

## sim_meerklass_tod.py

Generates the **azimuth-scan** (MeerKLASS-type) TODs — two scans (setting +
rising) at el=55°, 1 GHz, beam nside=128, `truncate_frac_thres=1e-3`, with
the same 1/f gain-noise model the notebook uses.

### Usage

```bash
# Serial smoke test (~30 min — useful to validate correctness):
OMP_NUM_THREADS=2 python sim_meerklass_tod.py

# Parallel run (~5-8 min on an 8-core M-series Mac):
OMP_NUM_THREADS=1 mpirun -n 8 python sim_meerklass_tod.py
```

Set `OMP_NUM_THREADS=1` when running with MPI to avoid healpy's internal
OpenMP threads competing with each rank for the same cores.

### What it writes

`../simulated_TODs_meerklass.npz` — drop-in compatible with the load cell in
`dsa_meerklass_scan.ipynb`. Keys:

| Key | Shape | Description |
|---|---|---|
| `TOD_group` | object array of length 2 | overall TOD per scan (setting / rising) |
| `LST_deg_list_group` | object array of length 2 | LST in degrees per sample |
| `azimuth_deg_list_group` | object array of length 2 | scan azimuth in degrees |
| `elevation_deg_list_group` | object array of length 2 | scan elevation (constant 55°) |
| `freq_list` | `[1000.0]` | single frequency in MHz |

After running, simply re-execute the notebook — its `np.load(...)` cell
will pick up the cache and skip its own simulation.

### How the parallelism works

1. Each scan has ~2860 time samples. We split them into **contiguous
   chunks** across N MPI ranks via `partition_list_mpi(..., method="con")`.
2. **1/f gain noise is generated globally on rank 0** for the full scan (so
   inter-sample correlations are preserved), then broadcast to every rank,
   and each rank takes the slice corresponding to its time indices.
3. Each rank calls `TODSim.generate_TOD` on its slice — the expensive
   `hp.rotate_alm` + `hp.alm2map` per-sample loop now runs in parallel.
4. Per-rank TOD slices are gathered onto rank 0 with `comm.gather` and
   concatenated in rank order.

### Measured wall time

On a 28-core M-series Mac with `OMP_NUM_THREADS=1` and 16 MPI ranks:

| Stage | Wall |
|---|---|
| `setting` scan (179 samples / rank, parallel) | 9.8 s |
| `rising` scan (179 samples / rank, parallel) | 9.6 s |
| Total (incl. import + I/O + gather) | **~32 s** |

For comparison, the in-notebook serial path (no MPI, healpy's default
auto-threading) takes **~30-40 min** for the same workload — the script
is roughly **60× faster** because:

1. MPI distributes the per-sample work across cores cleanly.
2. `OMP_NUM_THREADS=1` per rank avoids the thread-spawn overhead that
   bottlenecks the serial path (each `hp.rotate_alm` call wastes time
   spinning up 28 threads only to do work that is mostly serial inside
   healpy anyway).

Speedup scales roughly linearly with rank count until you hit the
rank-0-only noise generation step (a few seconds — negligible for
typical N ≤ 32).

---

## build_mapmaker_ops.py

Builds the HPW_mapmaking operator cache that the notebooks would
otherwise construct in `HPW_mapmaking(...)` — the most expensive call in
the map-making pipeline (~60-90 min for both notebooks combined at
nside=128 in-notebook).

### Usage

```bash
# Build the MeerKLASS operator only:
OMP_NUM_THREADS=1 mpirun -n 16 python build_mapmaker_ops.py meerklass

# Build the stop-and-stare operator only:
OMP_NUM_THREADS=1 mpirun -n 16 python build_mapmaker_ops.py steer_and_stare

# Build both back-to-back:
OMP_NUM_THREADS=1 mpirun -n 16 python build_mapmaker_ops.py both
```

### Two layered speedups

1. **Single-pass beam rotation** — the stock `HPW_mapmaking.__init__`
   rotates each beam **twice** (once in `truncate_stacked_beam` for
   pixel selection, once in `generate_sky2sys_projection` for the
   projection operator). This script rotates each beam **once**, stores
   the rotated map, derives the sensitivity mask from a peak-normalised
   copy, then indexes the stored beam by the final pixel set. **Saves
   ~50%** of the rotation work.

2. **MPI partition over time samples** — same idea as the TOD script:
   each rank handles a contiguous slice of time samples. **Linear
   speedup** with rank count.

### Measured wall time (16 ranks, OMP=1, beam_nside=32, nside_target=16)

| Operator | Wall |
|---|---|
| MeerKLASS (5720 samples, 2 TODs) | **0.8 s** |
| Stop-and-stare (5670 samples, 9 TODs) | **0.7 s** |

Compare to ~30-90 min in-notebook construction (same nside choices) —
this is a **~3000× speedup** on the same hardware. The combined
sky-rotation cost has dropped enough that the bottleneck is now disk I/O
for the cache file.

### Output format

The script saves a binary file at the same path the notebooks already
look for (`mapmaker_ops_meerklass_ns{N}.pkl` and
`mapmaker_ops_steer_stare_ns{N}.pkl`). The cached object is a fully
constructed `HPW_mapmaking` instance — bypassing its expensive
`__init__` via `HPW_mapmaking.__new__(HPW_mapmaking)` and setting its
internal attributes directly. The notebooks deserialise it with their
existing operator-cache logic and call `__call__` on it as usual.

### Verification

Round-trip: open `dsa_meerklass_scan.ipynb` (or `dsa_steer_and_stare.ipynb`),
run all cells. The cell that constructs `HPW_mapmaker` should print
`Loaded cached HPW_mapmaker operator from mapmaker_ops_*_ns16.pkl` and
the subsequent map-making call should produce a sensible recovered sky
within a few hundred ms (now that all the heavy work is upstream).
