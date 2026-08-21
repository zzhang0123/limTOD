# limTOD

**Time-Ordered Data simulator for single-dish (autocorrelation) radio
intensity mapping — with a differentiable pure-JAX port.**

📖 **Documentation: <https://limtod.readthedocs.io>**

> ### 🧭 Beam convention in one line
>
> **Multiplying a beam map by a sky map with no rotation — `sum(beam * sky)` —
> is the pointing `lat_deg=90, el_deg=90, az_deg=0` (North Pole, zenith,
> azimuth 0), at `lst_deg=0`, `selfrot=0`.**
>
> Azimuth **0**, not 180: at `el=90` the boresight is the zenith whatever the
> azimuth, so azimuth only *rolls* the beam about the boresight and the
> identity is the zero roll. This is also the quickest way to *verify* the
> convention — see
> [the one-test box in Theory & conventions](https://limtod.readthedocs.io/en/latest/theory.html#beam-coordinate-convention).

limTOD simulates the time-ordered data (TOD) of a single-dish telescope
scanning a HEALPix sky with an arbitrary (asymmetric) beam: the beam is
rotated to each pointing in spherical-harmonic space and dotted with the
sky, then combined with 1/f gain noise and white noise. The package also
ships:

- **Map-making** — `HPW_mapmaking` (high-pass + Wiener) and
  `GLS_mapmaking` (full 1/f + white noise covariance, ported from
  hydra-tod's iterative GLS);
- **`limTOD.patchbeam`** — a MeerKLASS-optimal sky-TOD path that keeps
  narrow, finely-gridded beams on their native (l, m) grid and integrates
  disc-restricted sky patches (no harmonic rotation);
- **`limTOD.uvbeam`** — adapters for
  [pyuvdata `UVBeam`](https://pyuvdata.readthedocs.io/en/latest/uvbeam.html)
  objects, feeding measured/simulated beams into either path;
- **`limTOD.cstbeam`** — the same for CST Studio far-field exports, the
  format a simulated horn usually arrives in (no extra needed);
- **`limTOD.tris`** — an offline bridge to the public TRIS archive: strict
  readers, a beam built from the archive's own E/H cuts, the verified
  zenith/roll geometry, and operator + noise + prior objects for
  prior-regularized map-making of the dec +42° drift ring;
- **`limtod_jax`** — a pure-JAX, jit/vmap/grad-safe port of the sky→TOD
  chain, verified against the numpy implementation to ~1e-12 in float64.
  It powers the differentiable pipeline of
  [rheplicant](https://github.com/RHINO-Experiment/rheplicant).

Latest changes:
[CHANGELOG](https://github.com/zzhang0123/limTOD/blob/main/CHANGELOG.md).

## Installation

```bash
pip install limTOD
```

The base install is deliberately lightweight (wheel-only: numpy, healpy,
astropy, scipy, tqdm, mpmath — no compiler needed). Heavier dependencies
are opt-in extras:

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

From source: clone the
[repository](https://github.com/zzhang0123/limTOD) and
`pip install -e ".[dev,full]"` (runs the test suite over both the
MPI-present and serial-fallback paths).

## Quick start

Simulate multi-frequency TOD for a MeerKAT-like scan
(sky model here needs `[gdsm]`; pass your own `sky_func` to go without):

```python
from limTOD import TODSim, example_scan

simulator = TODSim(
    ant_latitude_deg=-30.7130, ant_longitude_deg=21.4430, ant_height_m=1054,
    beam_nside=256, sky_nside=256,
)
time_list, azimuth_list = example_scan()
tod, sky_tod, gain_noise = simulator.generate_TOD(
    freq_list=[950, 1000, 1050],          # MHz
    time_list=time_list,
    azimuth_deg_list=azimuth_list,
    elevation_deg=41.5,
)                                          # each (n_freq, n_time)
```

The same sky→TOD chain, differentiable in JAX (`[jax]` extra):

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import limtod_jax as ltj

sky_alm = ltj.map2alm_quad(sky_map, nside=nside, lmax=lmax)
psi, theta, phi = ltj.zyz_of_pointing(lst_deg, lat_deg, az_deg, el_deg, 0.0)
tod = ltj.generate_tod_sky(
    beam_alm, sky_alm, jnp.stack([psi, theta, phi], axis=-1), lmax=lmax,
)
grad = jax.grad(lambda b: ltj.generate_tod_sky(
    b, sky_alm, jnp.stack([psi, theta, phi], axis=-1), lmax=lmax).sum().real
)(beam_alm)                                # d(TOD sum)/d(beam alms)
```

## Documentation

Full documentation: **<https://limtod.readthedocs.io>**

| Page | Contents |
|------|----------|
| [TOD simulation](https://limtod.readthedocs.io/en/latest/tod-simulation.html) | `TODSim` guide: inputs, outputs, noise model, MPI, troubleshooting |
| [Map-making](https://limtod.readthedocs.io/en/latest/mapmaking.html) | `HPW_mapmaking` (high-pass + Wiener) and `GLS_mapmaking` (full 1/f covariance) |
| [Patch-beam path](https://limtod.readthedocs.io/en/latest/patchbeam.html) | `limTOD.patchbeam`: disc-restricted (l, m) beam interpolation |
| [UVBeam support](https://limtod.readthedocs.io/en/latest/uvbeam.html) | pyuvdata beams as `beam_func` or patch beams |
| [CST beams](https://limtod.readthedocs.io/en/latest/cstbeam.html) | CST Studio far-field exports as `beam_func` or HEALPix maps |
| [TRIS support](https://limtod.readthedocs.io/en/latest/tris.html) | Public TRIS archive conventions, offline readers, and safe reduced inference |
| [Theory & conventions](https://limtod.readthedocs.io/en/latest/theory.html) | Signal model, coordinate chain, Euler-angle conventions |
| [API reference](https://limtod.readthedocs.io/en/latest/api/index.html) | Generated from docstrings |
| [limtod_jax](https://limtod.readthedocs.io/en/latest/limtod-jax.html) | The JAX port: usage, exactness contract, precision requirements |

Worked notebooks:
[TOD simulation](https://github.com/zzhang0123/limTOD/blob/main/examples/TODsim_examples.ipynb)
and
[map-making](https://github.com/zzhang0123/limTOD/blob/main/examples/mm_example.ipynb).
Coordinate and beam-orientation conventions:
[Theory & conventions](https://limtod.readthedocs.io/en/latest/theory.html).

## Citation

If you use limTOD in your research, please cite:

```bibtex
@ARTICLE{2026RASTI...5ag024Z,
       author = {{Zhang}, Zheng and {Bull}, Philip and {Santos}, Mario G. and {Nasirudin}, Ainulnabilah},
        title = "{Joint Bayesian calibration and map-making for intensity mapping experiments}",
      journal = {RAS Techniques and Instruments},
     keywords = {Data Methods, methods: data analysis, techniques: spectroscopic, radio lines: general, Instrumentation and Methods for Astrophysics},
         year = 2026,
        month = jan,
       volume = {5},
          eid = {rzag024},
        pages = {rzag024},
          doi = {10.1093/rasti/rzag024},
archivePrefix = {arXiv},
       eprint = {2509.10992},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026RASTI...5ag024Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## License and authorship

MIT License — see
[LICENSE](https://github.com/zzhang0123/limTOD/blob/main/LICENSE).

limTOD is developed and maintained by Zheng Zhang (University of
Manchester), with help and advice from members of the MeerKLASS
and RHINO collaborations — including Phil Bull, Piyanat Kittiwisit,
Geoff Murphy, and Mario Santos.
