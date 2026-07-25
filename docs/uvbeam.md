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

## Conventions: UVBeam vs limTOD, and the adapter

**pyuvdata's frame** (per the pyuvdata ≥3.2 source): a UVBeam on an
`az_za` grid lives in a *fixed* antenna-local Cartesian frame with
**x = East, y = North, z = zenith = boresight** — the coordinate-system
registry says *"az runs from East to North"*: `az = 0` along East,
`az = 90°` along North, increasing counterclockwise (the mathematician's
angle in the East–North plane), with `za` measured from the boresight.
This is **not** the astronomer's compass azimuth (North 0° → East 90°);
the two are mirror images: `az_astro = 90° − az_uvbeam`. E-field beams
carry two vector components along this grid's θ̂(za)/φ̂(az) unit
vectors, and `x_orientation="east"` ties the X feed to the `az = 0`
(East) axis. Crucially, **pyuvdata does not define how this frame
rotates when a dish points away from zenith** — that is the consumer's
job.

**limTOD's frame**
([theory](theory.md#beam-coordinate-convention)) is anchored to the
*pointing*, not to fixed compass labels: the beam-map meridian φ = 0
tracks the **up** side (increasing elevation) and φ = 90° the **right**
side (increasing azimuth), at every pointing.

**The adapter** supplies the missing rotation rule through one
identification, applied when the UVBeam is sampled onto the beam map:

```
healpix (θ, φ)   ←   UVBeam (za = θ,  az = 90° − φ)
```

In axis language: **UVBeam's North axis becomes the beam's up side
(φ = 0), and UVBeam's East axis (the X feed for `x_orientation="east"`)
becomes the right side (φ = 90°)**. Note the minus sign: the two
azimuths increase in *opposite* senses (`az`: East → North; φ:
up → right, i.e. North-image → East-image), so the identification is
orientation-reversing at the chart level. This is exactly the kind of
statement hand derivations get wrong — an earlier one had a handedness
error — so the mapping was **locked numerically**: a strongly displaced
test beam pushed through the HEALPix path and the independent
`limTOD.patchbeam` disc path agrees at 0.5%, while every other candidate
mapping is 66–90% off (`tests/test_uvbeam.py::TestOrientationLock` pins
both directions).

`uvbeam_to_patch_beam` applies the same identification via direction
cosines: the UVBeam components
`(sin za · cos az, sin za · sin az) = (East, North)` map onto the patch
grid's `(l, m) = (right, up)` axes.

If your beam file was produced under a different physical mounting
convention (e.g. the feed frame rotated relative to the E/N axes),
rotate the UVBeam yourself before passing it in — the adapter assumes
exactly the identification above.

Other adapter behavior:

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
