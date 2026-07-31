# limTOD.cstbeam — using CST Studio far-field exports

The sibling of [`limTOD.uvbeam`](uvbeam.md) for the other format a measured or
simulated horn arrives in: **CST Microwave Studio far-field ASCII exports**,
one file per frequency. Needs only `healpy` and `scipy`, both base
dependencies — there is no extra to install.

```python
from limTOD.cstbeam import cst_beam_func
from limTOD import TODSim

sim = TODSim(beam_func=cst_beam_func("~/beams/HornDryGround"),
             sky_func=..., ...)
```

## Three entry points

| Function | Gives you |
|---|---|
| `read_cst_farfield(path)` | one file as `(theta_deg, phi_deg, directivity)` on its own grid |
| `cst_beam_maps(directory, freq_MHz, nside=...)` | `(n_freq, npix)` HEALPix maps, frequency-interpolated |
| `cst_beam_func(directory)` | a `beam_func(freq=..., nside=...)` for `TODSim` |

`cst_beam_func` validates the directory and the options **at construction**,
not at the first channel, and caches each file's HEALPix resampling across
calls — a sweep of 200 channels over a 61-file directory parses each file once
rather than hundreds of times.

## Conventions

Stated because getting one wrong returns a finite, correctly shaped, **wrong**
beam.

**Theta.** CST's `Theta` is measured from the model's `+z` axis and maps
directly onto the HEALPix colatitude: the boresight sits at the pole, which is
what limTOD's beam maps mean by beam-local.

**The quantity.** `Abs(Dir.)` is total directivity in dBi — a *power* quantity.
Maps come back as $10^{\mathrm{dBi}/10}$, which is the $B$ of
$\int B T / \int B$. **Nothing is normalized here.** Divide by your own
quadrature $\int B$ downstream; that is the only way the band limit cancels
exactly.

**Frequency** is in MHz throughout, as elsewhere in limTOD, and is read from
the trailing number of each filename's stem — `HornDry70.5.txt` is 70.5 MHz.
Interpolation between bracketing files is linear in linear power.
Extrapolation is refused: a beam invented outside the simulated band is not a
beam.

### Phi, which the file does not contain

CST's `Phi` is measured from the model's `+x` axis. limTOD's beam-map
`phi = 0` is carried to the direction of increasing elevation. **Which physical
direction the CST `+x` axis points is a fact about how the horn was built and
mounted, and it is not in the export** — so it cannot be recovered here.

Two degrees of freedom are exposed instead:

| Option | Meaning |
|---|---|
| `phi0_deg` | the CST azimuth that lands on the beam-map `phi = 0` meridian |
| `phi_sense` | `"ccw"` if CST azimuth increases with beam-map `phi`, `"cw"` if it decreases |

The defaults are the identity mapping, which is **an assumption to check
against the as-built horn, not a result**. For a beam with real azimuthal
structure the handedness is not a detail — RHINO's horn varies by 30–60 %
around the $\theta = 30^\circ$ ring, and getting the sense backwards mirrors
that structure into the wrong half of the sky while leaving every integral,
every peak and every azimuthally-symmetric diagnostic unchanged.

:::{note}
Unlike [`limTOD.uvbeam`](uvbeam.md), whose azimuth convention *is* fixed by
pyuvdata and is therefore locked numerically by a three-way orientation test,
this one cannot be locked: the information is not in the file. What the tests
do lock is that the knobs act correctly — that `phi_sense` is a reflection
about `phi = 0` rather than a relabelling, and that `phi0_deg` is a rotation
that conserves the integral.
:::

## What a CST export looks like

Two header lines, then rows on a regular grid with **theta running fastest**
inside each phi block:

```text
Theta [deg.]  Phi [deg.]  Abs(Dir.)[dBi]  Abs(Theta)[dBi]  ...
--------------------------------------------------------------
     0.000      0.000   1.41421356000000e+01  ...
     2.000      0.000   1.41205511000000e+01  ...
```

Reshaping phi-fastest gives a correctly-shaped array with the samples
transposed — a beam, just not this one — so the reader checks that the rows
fill the complete grid they span and raises `ValueError` if they do not.

## API

```{eval-rst}
.. automodule:: limTOD.cstbeam
   :members:
```
