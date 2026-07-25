# limTOD.patchbeam — patch-beam TOD path (MeerKLASS-optimal)

`limTOD.patchbeam` computes the sky TOD **without ever rotating the beam in
harmonic space**: the beam stays on its native direction-cosine grid
`(l, m)` and, for every pointing, only the HEALPix sky pixels inside a
small disc around the pointing are touched. This is the right tool when
the beam is *narrow* and *finely gridded* — the MeerKLASS holographic
primary beam (±6°, tens of GB on disk) being the motivating case — where
the classic HEALPix spherical-harmonic path would need very high nside
to preserve precision.

Release 1.4.0 briefly shipped this subpackage as `limTOD.simeer`
(`SimeerTODSim`); those import paths still work as a deprecated alias
and will be removed in 2.0.

Merged from the standalone [Simeer](https://github.com/zzhang0123/Simeer)
package; this subpackage is the maintained home going forward. Classic
limTOD usage is unaffected.

## Algorithm (per pointing)

1. Rotate the pointing to equatorial coordinates and query a HEALPix
   disc of sky pixels around it (radius ~8° for a ±6° beam).
2. Rotate those pixels back to the horizontal frame at this LST and
   project onto the pointing's tangent plane (SIN projection). The
   direction-cosine axes follow the
   [beam coordinate convention](theory.md#beam-coordinate-convention):
   `l` along ê_az (**increasing azimuth**) and `m` along ê_el
   (**increasing elevation**) — e.g. for a pointing at azimuth 0°,
   `l` is toward East; at azimuth 180°, toward West. The beam cube's
   `(l, m)` axes are assumed to follow the same convention.
3. Bilinearly interpolate the beam power cube at those `(l, m)` — the
   weights are computed once per pointing and applied vectorially across
   the whole frequency axis.
4. Multiply by the sky, sum over the disc, normalize by the beam solid
   angle `Ω_b(ν)`.

## Quick start

```python
import numpy as np
from limTOD.patchbeam import MeerKLASSBeam, PatchBeamTODSim

beam = MeerKLASSBeam("meerklass_beam.npz", antenna="array_average")

sim = PatchBeamTODSim(
    beam=beam,
    sky_func=my_sky_func,          # limTOD convention: f(freq=..., nside=...)
    sky_nside=256,
    disc_radius_deg=8.0,
    polarization="HH",
)
tod, sky_tod, gain_noise = sim.generate_TOD(     # inherited from TODSim:
    freq_list=beam.freq_MHz[:16],                # same noise model, same API
    time_list=t_list,
    azimuth_deg_list=az_list,
    elevation_deg=41.5,
)
```

`PatchBeamTODSim` subclasses `limTOD.TODSim` and overrides **only** the
sky-TOD step — gain noise, 1/f noise, white noise, LST handling, and the
`generate_TOD` assembly are inherited unchanged. `simulate_sky_TOD`
mirrors the base signature (HEALPix-specific arguments like
`nside_hires` are accepted and ignored).

## Building blocks

| API | Purpose |
|---|---|
| `MeerKLASSBeam(path, antenna=..., polarizations=("HH","VV"))` | Load the holographic NPZ (only the requested antenna/pols are materialized, as float32 power) |
| `MeerKLASSBeam.from_arrays(...)` | In-memory beams for tests/synthetic studies |
| `synthetic_gaussian_beam(...)` | Circular-Gaussian power beam on the (l, m) grid |
| `integrate_tod(...)` / `integrate_sample(...)` | The sky-TOD core, usable without the simulator class |
| `limTOD.uvbeam.uvbeam_to_patch_beam(...)` | Bridge a pyuvdata UVBeam onto the (l, m) grid ([uvbeam guide](uvbeam.md)) |

## Parallelism

The sample loop parallelizes over time with joblib
(`PatchBeamTODSim(n_jobs=-1)` / `integrate_tod(n_jobs=...)`). joblib is an
optional dependency (`pip install "limTOD[parallel]"`); the default
serial path (`n_jobs=1`) needs nothing extra.

## Accuracy cross-check

`tests/patchbeam/test_against_limtod.py` pins the disc path against the
classic HEALPix spherical-harmonic path on matched Gaussian beams
(agreement at the few-percent level, limited by HEALPix pixelization of
the narrow beam — the regime where the disc path is *more* accurate, not
less).
