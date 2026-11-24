# Installation

```bash
pip install limTOD
```

The base install is deliberately lightweight (wheel-only: numpy, healpy,
astropy, scipy, tqdm, mpmath — no compiler needed) and supports
Python ≥ 3.8. Heavier dependencies are opt-in extras:

| Extra | Installs | When you need it |
|-------|----------|------------------|
| `[mpi]` | mpi4py | MPI-parallel simulation (`mpirun -n N ...`). Without it limTOD runs serially; launching under `mpirun` *without* mpi4py fails loudly instead of silently duplicating work. |
| `[gdsm]` | pygdsm | The `GDSM_sky_model` sky function (Global Sky Model). Everything else works without it. |
| `[jax]` | jax, s2fft | The `limtod_jax` package (Python ≥ 3.11). |
| `[uvbeam]` | pyuvdata | `limTOD.uvbeam`: use pyuvdata `UVBeam` objects as beams (Python ≥ 3.11). |
| `[parallel]` | joblib | Parallel sample loop in `limTOD.patchbeam` (`n_jobs != 1`). |
| `[full]` | the MPI, GDSM, JAX, UVBeam and parallel extras | Simulation with all optional backends. |
| `[mmode-bayes]` | bayesmith | Bayesian m-mode inference (Python ≥ 3.11). Install it together with `[full]` when you want both. |

```bash
pip install "limTOD[full]"
```

## From source

The published wheel supports Python 3.8 and later. Building from source
requires Python 3.9 or later for setuptools. The following development
setup includes the JAX and UVBeam extras and requires Python 3.11 or later.

```bash
git clone https://github.com/zzhang0123/limTOD.git
cd limTOD
pip install -e ".[dev,full]"
pytest
```

Running the suite in an environment *without* the extras exercises the
serial/fallback code paths — both matrices are supported.
