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
- **`limTOD.cstbeam`** — the same for CST Studio far-field exports, the
  format a simulated horn usually arrives in;
- **`limtod_jax`** — a pure-JAX, jit/vmap/grad-safe port of the sky→TOD
  chain, verified against the numpy implementation to ~1e-12 in float64,
  written to be the forward model of a differentiable inference pipeline
  (see [Downstream](#downstream)).

:::{admonition} 🧭 Beam convention, in one line
:class: important

Multiplying a beam map by a sky map with
no rotation — `sum(beam * sky)` — is the pointing `lat_deg=90, el_deg=90,
az_deg=0` (North Pole, zenith, **azimuth 0**, not 180), at `lst_deg=0`,
`selfrot=0`. It is also the fastest way to *verify* the convention:
[the one-test box in Theory & conventions](theory.md#beam-coordinate-convention).
:::

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
cstbeam
limtod-jax
driftscan
tris
```

```{toctree}
:caption: Reference
:maxdepth: 1

theory
api/index
changelog
```

## Downstream

The dependency runs one way. limTOD needs numpy, healpy, astropy, scipy, tqdm
and mpmath, and nothing that consumes it; the pages here name a downstream
pipeline only where something is deliberately *not* limTOD's job.

That pipeline is
[rheplicant](https://github.com/RHINO-Experiment/rheplicant) — the RHINO
experiment's digital twin, formerly e-RHINO and package `replicant` — which
requires limTOD and wraps the projectors here as signal-path operators
(`GeneralPointingProjector`, `DriftScanProjector`, `MatrixProjector`), with
`SkySpaceFilter` supplying the matrix-free CG map-making that
[Map-making](mapmaking.md#jax-alternative) and
[limtod_jax](limtod-jax.md) point at.

The split of subject matter is deliberate, and it runs both ways: how a beam
weights the sky, where the horizon falls in it and what share of it survives
are limTOD's subject; placing the result on a signal path is not.

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
