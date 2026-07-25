# limTOD

**Time-Ordered Data simulator for single-dish (autocorrelation) radio
intensity mapping — with map-making and a differentiable pure-JAX port.**

limTOD simulates the time-ordered data (TOD) of a single-dish telescope
scanning a HEALPix sky with an arbitrary (asymmetric) beam: the beam is
rotated to each pointing in spherical-harmonic space and dotted with the
sky, then combined with 1/f gain noise and white noise. Around that core
the package ships:

- **Map-making** — the high-pass + Wiener `HPW_mapmaking` and the
  GLS `GLS_mapmaking` (full 1/f + white noise covariance, ported from
  [hydra-tod](https://github.com/hydra-cosmology/hydra-tod));
- **`limTOD.patchbeam`** — a disc-restricted sky-TOD path that keeps
  narrow, finely-gridded beams on their native $(l, m)$ grid
  (MeerKLASS-optimal, no harmonic rotation);
- **`limTOD.uvbeam`** — adapters for
  [pyuvdata `UVBeam`](https://pyuvdata.readthedocs.io/en/latest/uvbeam.html)
  objects, feeding measured/simulated beams into either path;
- **`limtod_jax`** — a pure-JAX, jit/vmap/grad-safe port of the sky→TOD
  chain, verified against the numpy implementation to ~1e-12 in float64.
  It powers the differentiable pipeline of
  [replicant-telescope](https://github.com/zzhang0123/replicant-telescope).

```{toctree}
:caption: Getting started
:maxdepth: 1

installation
quickstart
```

```{toctree}
:caption: User guide
:maxdepth: 1

tod-simulation
mapmaking
patchbeam
uvbeam
limtod-jax
```

```{toctree}
:caption: Reference
:maxdepth: 1

theory
api/index
changelog
```

## Citation

If you use limTOD in your research, please cite
[Zhang et al. (2026), RASTI, rzag024](https://doi.org/10.1093/rasti/rzag024):

```bibtex
@ARTICLE{2026RASTI...5ag024Z,
       author = {{Zhang}, Zheng and {Bull}, Philip and {Santos}, Mario G. and {Nasirudin}, Ainulnabilah},
        title = "{Joint Bayesian calibration and map-making for intensity mapping experiments}",
      journal = {RAS Techniques and Instruments},
         year = 2026,
        month = jan,
       volume = {5},
          eid = {rzag024},
        pages = {rzag024},
          doi = {10.1093/rasti/rzag024},
archivePrefix = {arXiv},
       eprint = {2509.10992},
 primaryClass = {astro-ph.IM},
}
```

## License and authorship

MIT License — see
[LICENSE](https://github.com/zzhang0123/limTOD/blob/main/LICENSE).

limTOD is developed and maintained by Zheng Zhang (University of
Manchester), with help and advice from members of the MeerKLASS and
RHINO collaborations — including Phil Bull, Piyanat Kittiwisit, Geoff
Murphy, and Mario Santos.
