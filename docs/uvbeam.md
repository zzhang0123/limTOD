# limTOD.uvbeam — using pyuvdata UVBeam objects

`limTOD.uvbeam` connects measured or simulated beams stored as
[pyuvdata `UVBeam`](https://pyuvdata.readthedocs.io/en/latest/uvbeam.html)
objects (CST/FEKO exports, holography products, `AnalyticBeam.to_uvbeam`
outputs, ...) to both of limTOD's simulation paths.

```bash
pip install "limTOD[uvbeam]"        # pulls pyuvdata
```

## 1. As a `beam_func` for the classic HEALPix path

```python
from pyuvdata import UVBeam
from limTOD import TODSim
from limTOD.uvbeam import uvbeam_beam_func

uvb = UVBeam.from_file("my_beam.beamfits")

sim = TODSim(
    beam_func=uvbeam_beam_func(uvb),   # chromatic: interpolates per freq
    sky_func=my_sky_func,
    beam_nside=256, sky_nside=256,
)
```

`uvbeam_beam_func(uvb, stokes="IQUV")` returns full `(4, npix)`
pseudo-Stokes beam rows (efield beams are converted with pyuvdata's own
`efield_to_pstokes`); the default `stokes="I"` accepts efield beams or
power beams carrying XX and YY (averaged to Stokes I). One-off maps:
`uvbeam_to_healpix_maps(uvb, freq_MHz=..., nside=...)`.

## 2. As a patch beam for the `limTOD.patchbeam` disc path

```python
import numpy as np
from limTOD.patchbeam import PatchBeamTODSim
from limTOD.uvbeam import uvbeam_to_patch_beam

patch = uvbeam_to_patch_beam(
    uvb, margin_deg=np.linspace(-6.0, 6.0, 481), polarization="HH",
)
sim = PatchBeamTODSim(beam=patch, sky_func=my_sky_func, sky_nside=256)
```

This samples the UVBeam onto the `(l, m)` direction-cosine grid of
[`MeerKLASSBeam`](patchbeam.md) — the right choice for narrow beams.

## Conventions (numerically locked)

- UVBeam zenith angle → HEALPix polar angle directly (boresight at the
  pole; limTOD's beam-map convention).
- UVBeam azimuth (pyuvdata convention: East = 0, North = π/2,
  counterclockwise) maps to the HEALPix beam-map azimuth as
  ``az_uvbeam = π/2 − φ_healpix``. This mapping was **locked
  numerically** — a strongly displaced test beam pushed through the
  HEALPix path and the independent `limTOD.patchbeam` disc path agrees at
  0.5%, while every other candidate mapping is 66–90% off
  (`tests/test_uvbeam.py::TestOrientationLock` keeps both directions
  pinned). Hand derivations of such conventions are not trusted in this
  package — an earlier one had a handedness error that only the
  numerical lock caught.
- Pixels beyond the UVBeam's zenith-angle coverage are filled with
  `fill_value` (default 0).
- Frequencies are in MHz on the limTOD side and interpolated on the
  UVBeam frequency axis (`freq_interp_kind`, default linear).

## Scope and limits

- UVBeam objects on regular `(az, za)` grids are supported; HEALPix-
  pixelized UVBeams are rejected with a pointer to pyuvdata's own
  regridding.
- Full-Stokes (`stokes="IQUV"`) needs an efield beam (or a power beam
  already carrying pseudo-Stokes products).
- `uvbeam_to_patch_beam` maps `HH`→XX and `VV`→YY (pyuvdata's default
  `x_orientation="east"` makes X the horizontal feed).
