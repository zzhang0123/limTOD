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
| `[full]` | all of the above | The complete setup. |

```bash
pip install "limTOD[full]"
```

## From source

Clone the [repository](https://github.com/zzhang0123/limTOD) and install
in editable mode with the development extras:

```bash
git clone https://github.com/zzhang0123/limTOD.git
cd limTOD
pip install -e ".[dev,full]"
pytest          # ~310 tests; optional-dependency paths skip cleanly
```

Running the suite in an environment *without* the extras exercises the
serial/fallback code paths — both matrices are supported.
