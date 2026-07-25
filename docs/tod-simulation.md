# TOD simulation with `TODSim`

`TODSim` simulates time-ordered data for a single-dish telescope with a
single-pixel beam. To model a multi-dish or multi-beam instrument, run one
instance per dish/beam. Each call returns the sky temperature TOD, the gain
noise TOD, and the overall TOD combining all components.

## Quick start

```python
import numpy as np
from limTOD import TODSim, example_scan

simulator = TODSim(
    ant_latitude_deg=-30.7130,   # MeerKAT latitude
    ant_longitude_deg=21.4430,   # MeerKAT longitude
    ant_height_m=1054,           # MeerKAT altitude [m]
    beam_nside=256,              # HEALPix resolution for the beam
    sky_nside=256,               # HEALPix resolution for the sky
)

time_list, azimuth_list = example_scan()   # simple raster scan

tod_array, sky_tod, gain_noise = simulator.generate_TOD(
    freq_list=[950, 1000, 1050],           # MHz
    time_list=time_list,
    azimuth_deg_list=azimuth_list,
    elevation_deg=41.5,
)
print(tod_array.shape)                     # (3, n_time)
```

> The default `sky_func` is `GDSM_sky_model`, which needs the `[gdsm]`
> extra (`pip install "limTOD[gdsm]"`). Pass your own `sky_func` to run
> without it.

Worked notebooks:
[TODsim_examples.ipynb](https://github.com/zzhang0123/limTOD/blob/main/examples/TODsim_examples.ipynb) (simulation),
[mm_example.ipynb](https://github.com/zzhang0123/limTOD/blob/main/examples/mm_example.ipynb) (map-making workflow).

## Beam and sky functions

`beam_func` and `sky_func` are callables taking **keyword-only** arguments,
two of which must be `freq` and `nside`, returning a HEALPix map:

- 1D array of length `npix` — unpolarized (Stokes I);
- 2D array `(3, npix)` — polarized (I, Q, U);
- 2D array `(4, npix)` — full Stokes (I, Q, U, V).

**Beam orientation.** The beam map must follow the
[beam coordinate convention](theory.md#beam-coordinate-convention):
boresight at the map's north pole (θ = 0); the φ = 0 meridian is
carried to the direction of **increasing elevation** (ê_el ≡ ∂b̂/∂e)
and φ = 90° to the direction of **increasing azimuth** (ê_az) at the
pointing, with positive `selfrot_deg` rotating the pattern from ê_el
toward ê_az. Symmetric beams are unaffected; for asymmetric or
polarized beams this is load-bearing.

Built-ins: `example_beam_map` (elliptical Gaussian),
`example_symm_beam_map` (symmetric Gaussian), `GDSM_sky_model`
(Global Sky Model, `[gdsm]` extra), and
`generate_gaussian_field` (correlated Gaussian sky realizations from a
frequency–frequency angular power spectrum).

## Input parameters

### Telescope configuration (`TODSim.__init__`)

| Parameter | Type | Meaning |
|---|---|---|
| `ant_latitude_deg` | float | Site latitude [deg] (N positive) |
| `ant_longitude_deg` | float | Site longitude [deg] (E positive) |
| `ant_height_m` | float | Site altitude [m] |
| `beam_func` | callable | Beam map factory (see above) |
| `sky_func` | callable | Sky map factory (see above) |
| `beam_nside` | int | Beam map resolution — large enough to resolve beam features |
| `sky_nside` | int | Sky map resolution — sets how the sky is parametrized |

### Observation parameters (`generate_TOD` / `simulate_sky_TOD`)

| Parameter | Type | Meaning |
|---|---|---|
| `freq_list` | array-like | Frequencies [MHz] |
| `time_list` | array-like | Time offsets [s] from `start_time_utc` |
| `azimuth_deg_list` | array-like | Per-sample azimuth [deg], east of north |
| `elevation_deg` | float or array-like | Elevation [deg]; a scalar applies to all samples |
| `selfrot_deg_list` | array-like, optional | Antenna self-rotation [deg]; default zero |
| `start_time_utc` | str | UTC start, e.g. `"2019-04-23 20:41:56.397"` |
| `horizontal_mask` | array, optional | Binary HEALPix mask in horizontal coordinates (1 = keep) |
| `nside_hires` | int, optional | Upgrade the beam to this nside first (narrow beams) |
| `normalize_beam` | bool | Divide each pointed beam by its pixel sum (default False) |
| `truncate_frac_thres` | float | Zero pixels below this fraction of the beam peak (default 1e-10) |

### Noise and calibration (`generate_TOD` only)

| Parameter | Type | Meaning |
|---|---|---|
| `Tsys_others_TOD` | array `(nfreq, ntime)`, optional | Additional system-temperature components |
| `background_gain_TOD` | array `(nfreq, ntime)`, optional | Background gain (default unity) |
| `gain_noise_TOD` | array `(nfreq, ntime)`, optional | Precomputed gain noise (overrides generation) |
| `gain_noise_params` | list | `[f0, fc, alpha]` for 1/f noise generation; default `[1.335e-5, 1.099e-3, 2]` |
| `white_noise_var` | float, optional | White-noise variance (default `2.5e-6`, dimensionless fractional) |
| `return_LSTs` | bool | Also return the LST array (default False) |

## Outputs

| Output | Shape | Meaning |
|---|---|---|
| `overall_TOD` | `(nfreq, ntime)` | All components combined (model below) |
| `sky_TOD` | `(nfreq, ntime)` | Beam-weighted sky signal only |
| `gain_noise_TOD` | `(nfreq, ntime)` | The 1/f gain-noise realization |
| `LST_deg_list` | `(ntime,)` | Only when `return_LSTs=True` |

The combination implemented in `generate_TOD` is

```
overall_TOD = G_bg · (1 + G_noise) · (sky_TOD + Tsys_others) · (1 + η)
```

with `G_bg` the background gain, `G_noise` the 1/f gain noise, and `η`
white noise. See [theory.md](theory.md) for how `sky_TOD` itself is
computed (beam rotation in harmonic space + beam-weighted sum).

## MPI parallelism

The time axis is partitioned across ranks; every rank returns the full
`(nfreq, ntime)` array:

```bash
pip install "limTOD[mpi]"
mpirun -n 4 python your_simulation.py
```

Without mpi4py installed, the same code runs serially (`rank=0, size=1`).
Launching with `mpirun -n N` *without* mpi4py raises a `RuntimeError`
instead of silently running N duplicate serial copies
(escape hatch: `LIMTOD_FORCE_SERIAL=1`). Per-frequency work also benefits
from numpy/scipy/healpy internal threading — 2–4 cores per rank is a good
default.

## Troubleshooting

- **Installation errors** — use a fresh virtual environment; the base
  install is wheel-only (no compiler needed).
- **`ImportError: ... limTOD[gdsm]`** — the default sky model needs the
  `[gdsm]` extra, or pass your own `sky_func`.
- **`RuntimeError: An MPI launcher is detected ...`** — install the
  `[mpi]` extra for parallel runs.
- **Memory errors** — reduce `nside` or process fewer frequencies at once.
- **Slow narrow beams** — set `nside_hires` for the beam instead of
  raising `sky_nside`.
- **Time errors** — `start_time_utc` must be an ISO-format UTC string.
