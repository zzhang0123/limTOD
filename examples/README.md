# Examples

## Self-contained (run after `pip install "limTOD[gdsm]"`)

| Notebook / directory | Contents |
|---|---|
| [TODsim_examples.ipynb](TODsim_examples.ipynb) | TOD simulation walkthroughs with `TODSim` |
| [mm_example.ipynb](mm_example.ipynb) | Full high-pass + Wiener map-making workflow (`HPW_mapmaking`) |
| [DSA/](DSA) | DSA-2000-style single-dish survey study: scan strategies, TOD simulation, map-making comparisons. The large operator/TOD caches are **not** in the repo — regenerate them with the scripts in `DSA/scripts/` (see `DSA/README.md`); the beam-map FITS inputs are included. |

## Require external data NOT included in this repository

These notebooks reference beam files from other projects (absolute paths on
the author's machine) and will raise `FileNotFoundError` as-is; adapt the
paths or substitute `limTOD.example_beam_map` / `example_symm_beam_map`:

| Notebook | Missing input |
|---|---|
| [demonstration.ipynb](demonstration.ipynb) | `REACH_Efield.txt` / `REACH_beam_func.pkl` from the TIBEC project |
| [RotatingBeam/baseSim.ipynb](RotatingBeam/baseSim.ipynb), [RotatingBeam/baseSim2.ipynb](RotatingBeam/baseSim2.ipynb) | `HornWet70.0.fits` beam model (MERS project) |
