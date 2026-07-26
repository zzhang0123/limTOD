# Theory and conventions

This page is the authoritative statement of limTOD's conventions —
coordinate systems, the beam orientation, Euler angles, and the noise
model. (It supersedes the retired `conventions.pdf`, whose beam-frame
figure did not pin the local horizontal system and therefore could not
define the orientation uniquely.)

## Signal model

```
TOD(ν,t) = G_bg(ν,t) · [1 + G_noise(ν,t)] · [sky_TOD(ν,t) + Tsys_others(ν,t)] · [1 + η(t)]
```

| Term | Meaning |
|---|---|
| `G_bg(ν,t)` | Background gain pattern |
| `G_noise(ν,t)` | 1/f (flicker) gain fluctuations |
| `sky_TOD(ν,t)` | Sky brightness seen through the pointed beam |
| `Tsys_others(ν,t)` | All other system-temperature components |
| `η(t)` | White noise (dimensionless fractional) |

The sky term is the beam-weighted sum

```
sky_TOD(ν,t) = Σ_p B_t(p, ν) · T_sky(p, ν)
```

over HEALPix pixels `p`, where `B_t` is the beam rotated to the pointing at
time `t`. (This is the discrete counterpart of `∫ B · T dΩ`; limTOD uses
the plain pixel sum, without a solid-angle factor — pair it with
`normalize_beam=True` for a beam-weighted average.)

## Beam coordinate convention

:::{admonition} 🧭 The convention in one test — no rotation ⇔ North Pole, zenith, azimuth 0
:class: important

If you remember one thing from this page, remember this. Multiplying a beam
map by a sky map with no rotation at all,

$$
\sum_p B(p)\, S(p),
$$

is **not** a convention-free operation — it is one specific pointing:

> **an antenna at the terrestrial North Pole (`lat_deg = 90`), looking at
> its zenith (`el_deg = 90`), with `az_deg = 0`** — at `lst_deg = 0`,
> `selfrot = 0`.

**Azimuth 0, not 180.** At $e = 90°$ the boresight is the zenith whatever the
azimuth is, so azimuth no longer selects a *direction* — it only **rolls** the
beam about the boresight, and the identity is the zero roll. (The tempting
`180` comes from the $\varphi = 0$ axis landing toward the **south** point at
$A = 0°$: that is where the axis points, not the mount azimuth.) In general
the identity there is the family $A = \Theta_{\rm LST} + \psi$, because at
`lat = 90°, e = 90°` the whole five-angle chain collapses to
$R_z(\psi - A + \Theta_{\rm LST})$.

**This is also how you *determine* the convention** — the fastest, most
decisive check there is, and it needs three lines. Don't test for equality;
test which azimuth *wins*:

```python
import numpy as np, healpy as hp
from limTOD.simulator import generate_TOD_sky

nside, lmax = 64, 128
theta, phi = hp.pix2ang(nside, np.arange(12 * nside**2))
# The beam MUST be asymmetric under phi -> phi + 180: an even-order feature
# (cos^2 phi, or any beam that is a function of theta alone) is invariant
# under a 180-degree roll and so cannot see the difference at all.
beam = np.exp(-(theta**2) / (2 * np.deg2rad(9.0) ** 2)) * (
    1 + 0.5 * np.sin(theta) * np.cos(phi)
)
beam = hp.alm2map(hp.map2alm(beam, lmax=lmax), nside)   # band-limit it
sky = np.random.default_rng(0).standard_normal(beam.size)

plain = np.sum(beam * sky)                              # the no-rotation product
for az in (0.0, 90.0, 180.0, 270.0):
    tod = generate_TOD_sky(
        beam, sky,
        np.array([0.0]), 90.0,               # LST = 0, latitude = +90
        np.array([az]), np.array([90.0]),    # azimuth, elevation = 90 (zenith)
        np.array([0.0]),                     # selfrot = 0
        normalize_beam=False, truncate_frac_thres=0.0,
    )
    print(f"azimuth {az:5.1f}: {abs(tod[0] - plain) / abs(plain):.1e}")
```

```text
azimuth   0.0: 2.1e-03      <-- the identity
azimuth  90.0: 1.3e-01
azimuth 180.0: 2.7e-01
azimuth 270.0: 1.4e-01
```

Two orders of magnitude, so the verdict is unambiguous — and note the winner
is *not* zero: the 2·10⁻³ residual is HEALPix analysis/synthesis error from
the `map2alm` that `generate_TOD_sky` performs internally, **not** a
convention mismatch. That is exactly why the test is comparative. Pinned at
the alm level (where it *is* roundoff, 3·10⁻¹⁶) in
[`tests/test_beam_orientation.py`](https://github.com/zzhang0123/limTOD/blob/main/tests/test_beam_orientation.py)
`::test_north_pole_zenith_identity_is_azimuth_zero`.
:::

The beam enters limTOD as a HEALPix map (RING ordering;
`(θ, φ) = healpy.pix2ang`). A beam's orientation is only meaningful
relative to the **local horizontal system**, so the convention is
stated through it — first as a physical mount motion, then as formulas
(the formulas are normative). Every claim is pinned numerically by
[`tests/test_beam_orientation.py`](https://github.com/zzhang0123/limTOD/blob/main/tests/test_beam_orientation.py)
(displaced-blob probes through the full pointing chain).

### The convention as a mount motion

Park the dish at the zenith with the azimuth drive reading $0°$. The
beam map is then a chart of the sky around the zenith with its
meridians on the compass:

$$
\varphi = 0,\; 90°,\; 180°,\; 270°
\;\;\longrightarrow\;\;
\text{south, east, north, west points of the horizon,}
$$

and $\theta$ the angle from the zenith. Driving the mount to a
pointing $(A, e, \psi)$ carries the pattern rigidly in three steps:

1. rotate about the vertical by $A$ in the azimuth-increasing sense
   (north → east);
2. tilt the boresight down from the zenith by $90° − e$ — it descends
   along the vertical circle toward compass azimuth $A$, and the
   $\varphi = 0$ side of the chart now faces the direction of
   increasing elevation;
3. rotate by $\psi$ (`selfrot_deg`) about the boresight, in the
   $\varphi$-increasing sense.

![The beam coordinate convention: bird's-eye view of the horizontal
system, the vertical plane through the pointing, and the beam
chart.](_static/beam-convention.svg)

### Formal definition (normative)

**Horizontal system.** Right-handed Cartesian basis
$(\hat E, \hat N, \hat U)$: $\hat E$ toward East, $\hat N$ toward
North, $\hat U = \hat E \times \hat N$ toward the zenith. Azimuth $A$
is measured from North toward East ($\hat N$: $A = 0°$, $\hat E$:
$A = 90°$); elevation $e$ from the horizon ($0°$) toward the zenith
($90°$). The pointing direction is

$$
\hat b(A, e) \;=\; \sin A \cos e\; \hat E \;+\; \cos A \cos e\; \hat N
\;+\; \sin e\; \hat U .
$$

**Tangent basis at the pointing** — defined purely by coordinate
derivatives:

$$
\hat e_{\mathrm{el}} \;\equiv\; \frac{\partial \hat b}{\partial e}
 \;=\; -\sin A \sin e\; \hat E \;-\; \cos A \sin e\; \hat N
 \;+\; \cos e\; \hat U
$$

is the unit tangent along **increasing elevation** (at fixed azimuth),
and

$$
\hat e_{\mathrm{az}} \;\equiv\; \frac{1}{\cos e}
 \frac{\partial \hat b}{\partial A}
 \;=\; \cos A\; \hat E \;-\; \sin A\; \hat N
$$

is the unit tangent along **increasing azimuth** (at fixed elevation);
its normalized form is independent of $e$, so it stays defined at
$e = 90°$. The triad is right-handed:
$\hat e_{\mathrm{el}} \times \hat e_{\mathrm{az}} = \hat b$.

**The convention.** For a pointing $(A, e)$ with self-rotation $\psi$
(`selfrot_deg`):

- the **beam centre (boresight) is the beam map's north pole**,
  $\theta = 0$; $\theta$ is the angular distance from the boresight;
- the map point $(\theta, \varphi)$ is carried to the sky direction at
  angular distance $\theta$ from $\hat b$ along the tangent direction

$$
\hat t(\varphi) \;=\; \cos(\varphi + \psi)\; \hat e_{\mathrm{el}}
              \;+\; \sin(\varphi + \psi)\; \hat e_{\mathrm{az}} .
$$

Equivalently, with $\psi = 0$:

| beam-map meridian | carried to |
|---|---|
| $\varphi = 0$ | $+\hat e_{\mathrm{el}}$ (increasing elevation) |
| $\varphi = 90°$ | $+\hat e_{\mathrm{az}}$ (increasing azimuth) |
| $\varphi = 180°$ | $-\hat e_{\mathrm{el}}$ (decreasing elevation) |
| $\varphi = 270°$ | $-\hat e_{\mathrm{az}}$ (decreasing azimuth) |

Positive `selfrot_deg` rotates the pattern about the boresight in the
$\varphi$-increasing sense: the feature at meridian $\varphi$ is
carried to $\hat t$ evaluated at $\varphi + \psi$ (from
$\hat e_{\mathrm{el}}$ toward $\hat e_{\mathrm{az}}$).

The mount-motion narrative and the figure are exactly equivalent to
this definition; wherever wording could be read two ways, the formulas
win.

### Practical reading

Beam-map pixel $(\theta, \varphi = 0)$ holds the response to a source
at the **same azimuth and elevation $e + \theta$**; pixel
$(\theta, \varphi = 90°)$ to a source at the **same elevation and
larger azimuth** (offset $\approx \theta / \cos e$ for small
$\theta$; exactly, source offsets SIN-project onto
$(\hat e_{\mathrm{az}}, \hat e_{\mathrm{el}})$).

**Worked anchors.** Pointing $A = 0°, e = 0°$: $\hat b = \hat N$,
$\hat e_{\mathrm{el}} = \hat U$, $\hat e_{\mathrm{az}} = \hat E$ — the
$\varphi = 0$ meridian is carried toward the zenith, $\varphi = 90°$
toward East. Pointing $A = 180°, e = 0°$: $\hat b = -\hat N$,
$\hat e_{\mathrm{el}} = \hat U$, $\hat e_{\mathrm{az}} = -\hat E$ —
$\varphi = 90°$ is now carried toward **West**. This is why the
convention is stated through $\hat e_{\mathrm{az}}$/$\hat e_{\mathrm{el}}$
rather than compass or left/right words.

### Special cases

- **Identity / reading the map as equatorial.** For
  `lat = 0°, LST = 0°, A = 0°, e = 0°`, $\psi = 0$, the full rotation
  chain is the identity: the beam map can be read directly as an
  equatorial map — beam centre at the **north celestial pole**,
  $\varphi \equiv$ RA. The $\varphi = 0$ meridian runs along the
  RA $= 0°$ meridian in the direction of decreasing declination —
  which is this observer's zenith direction, since the zenith sits at
  (RA $0°$, Dec $0°$). Equivalently: an unrotated beam is the beam of
  an antenna at the terrestrial North Pole pointing at its zenith
  (the NCP), with the beam's $\varphi = 0$ axis along the
  $\Theta_{\rm LST} = 0$ meridian toward decreasing declination and
  $\varphi = 90°$ along RA $= 90°$. That second reading is
  `lat = 90°, e = 90°`, and its mount azimuth is
  $A = 0°$ — **not** $A = 180°$; more generally the identity there is
  the one-parameter family $A = \Theta_{\rm LST} + \psi$, since at
  `lat = 90°, e = 90°` the whole chain collapses to
  $R_z(\psi - A + \Theta_{\rm LST})$.

  Why the azimuth is easy to get wrong here: at $e = 90°$ the boresight
  is the zenith **whatever** $A$ is, so azimuth no longer selects a
  pointing direction — it only *rolls* the beam about the boresight, and
  the identity is the zero roll. The tempting wrong answer, $A = 180°$,
  comes from conflating that mount azimuth with the compass direction the
  $\varphi = 0$ axis lands on: at $A = 0°$ the $\varphi = 0$ axis points
  toward the **south** point (see the parked configuration below), and
  "south is azimuth 180°" then invites the substitution. Multiplying a
  beam map by a sky map with no rotation at all *is* this configuration
  ($A = 0°$); $A = 180°$ would rotate the beam by 180° in $\varphi$.
  Both statements are pinned in `tests/test_beam_orientation.py`.
- **Zenith pointing** ($e = 90°$, reached with mount azimuth $A$): the
  formulas remain valid and unambiguous —
  $\hat e_{\mathrm{az}} = \cos A\, \hat E - \sin A\, \hat N$ (unchanged)
  and $\hat e_{\mathrm{el}}(A, 90°) = -(\sin A\, \hat E + \cos A\, \hat N)$,
  the continuous carry-over of the mount's approach azimuth.
- **Parked configuration** ($A = 0°, e = 90°$): the compass reading
  used as the anchor of the mount-motion statement above —
  $\varphi = 0, 90°, 180°, 270° \to$ south, east, north, west
  points; equal to the general formulas at $A = 0°, e = 90°$ and
  pinned directly by the parked tests.
- **Symmetric beams** are insensitive to $\varphi$ entirely — which is
  why this section matters only for asymmetric beams and polarization
  work (and why the orientation went unstated for so long: symmetric
  cross-checks cannot detect it).

### Polarization: the basis follows, the handedness does not

Stokes $Q, U$ live in the **same tangent basis** as everything above
($\varphi = 0 \to \hat e_{\mathrm{el}}$), so fixing the beam convention
fixes the polarization basis — there is no separate frame to declare. The
transport is automatic too: a 3- or 4-row beam goes `map2alm` → rotate →
`alm2map`, and `map2alm` decomposes $(I,Q,U)$ into $(T,E,B)$; under a
rotation of the sphere $E$ and $B$ transform as ordinary scalar
$a_{\ell m}$ and do not mix, so synthesis returns $Q, U$ in the correctly
rotated local basis. Verified — the polarization position angle co-rotates
with the pattern, and the `spin-0` mistake (a frozen position angle) is
excluded by two orders of magnitude.

What is *not* automatic is one sign. Because the beam is rotated between
the caller's choice of convention and the dot product, a convention change
is harmless only if it **commutes with the transport** — and $(Q,U)$
transport is itself a rotation in the $(Q,U)$ plane. Measured, with the
change applied to the beam **and** the sky together:

| convention change | effect on the TOD | |
|---|---|---|
| **rotation** of the $(Q,U)$ reference axis | $1.7\times10^{-16}$ | harmless — rotations commute |
| $U \to -U$ (IAU vs CMB **handedness**) | $4.3\times10^{-2}$ | **matters** — a reflection does not |
| $V \to -V$ (IEEE vs IAU circular) | $0$ | harmless — $V$ is spin-0 |

So the reference *axis* is free (any choice, consistently applied, gives
the same answer) but the **handedness of $(Q,U)$ is not**: reflection and
rotation do not commute ($F R F = R^{-1}$), so a beam and a sky built with
opposite $U$-sign conventions reverse the sense in which the position angle
is carried, and the TOD is simply wrong — by $O(\text{polarized fraction})$,
and invisible to every Stokes-$I$ check. $V$'s sign convention, by
contrast, genuinely does not matter.

**Caller contract:** the beam's $Q, U$ and the sky's must share a
handedness. This only bites when mixing provenances — a `pyuvdata` beam
against an externally generated sky model, say — which is the same class of
problem as the UVBeam azimuth adapter below, and wants the same treatment:
lock the relative sign numerically at the boundary. Every number above is
pinned in
[`tests/test_stokes_and_boundaries.py`](https://github.com/zzhang0123/limTOD/blob/main/tests/test_stokes_and_boundaries.py).

The patch-beam path uses the same tangent basis: its direction cosines
are the SIN-projected components $(l, m)$ along
$(\hat e_{\mathrm{az}}, \hat e_{\mathrm{el}})$ — see
[patchbeam.md](patchbeam.md). For the pyuvdata UVBeam frame and the
adapter between the two conventions, see
[uvbeam.md](uvbeam.md#conventions-uvbeam-vs-limtod-and-the-adapter).

## Coordinate chain

Telescope pointing → equatorial beam orientation is expressed as Euler
rotations:

1. **Scan → LST.** UTC timestamps convert to Local Sidereal Time via the
   site location (`generate_LSTs_deg`, astropy).
2. **Pointing → ZYZYZ.** The natural rotation sequence
   `R = R_z(χ) R_y(δ) R_z(γ) R_y(β) R_z(α)` with

   | Angle | Value | Role |
   |---|---|---|
   | α | LST | Earth rotation |
   | β | 90° − latitude | Site location |
   | γ | −azimuth | Local pointing (east-of-north positive; note the sign) |
   | δ | elevation − 90° | Altitude |
   | χ | self-rotation | Antenna rotation about the beam centre |

3. **ZYZYZ → ZYZ.** `zyzyz2zyz` collapses the five-angle sequence to
   `R = R_z(φ) R_y(θ) R_z(ψ)`, returned as `(ψ, θ, φ)` in **radians**
   (the unit `healpy.rotate_alm` expects). Public APIs take **degrees**.
4. **Beam rotation in harmonic space.** `pointing_beam_in_eq_sys` rotates
   the beam's alm coefficients (`healpy.rotate_alm`) and synthesizes the
   pointed beam map — no per-pixel interpolation.
5. **Horizontal mask (optional).** A mask defined in horizontal
   coordinates (pole at the zenith) is rotated with the pointing-independent
   part of the chain — the zenith pointing azimuth = 0, elevation = 90, for
   which δ = 0 and the rotation reduces to ψ′ = α, θ′ = β, φ′ = 0 — then
   thresholded at 0.5 and applied to the pointed beam. (Before v1.3.0 the
   code used elevation = 0 here, tipping the mask 90° onto the horizon.)
6. **Sky integration.** `_beam_weighted_sum` forms the pixel dot product —
   one TOD sample per pointing.

## Numerical conventions

- **Angles**: public function signatures take degrees; internal Euler
  angles `(ψ, θ, φ)` are radians.
- **HEALPix**: RING ordering throughout; `lmax = 3·nside − 1` defaults.
- **`hp.rotate_alm` argument order**: limTOD calls
  `hp.rotate_alm(alm, φ, θ, ψ)` for its own `(ψ, θ, φ)` — the numerically
  locked convention the JAX port reproduces exactly (see
  [limtod-jax.md](limtod-jax.md)).
- **Beam truncation**: after rotation, pixels below
  `truncate_frac_thres × max(beam)` are zeroed (a nonlinear cleanup of
  synthesis ringing; disable with `truncate_frac_thres=0.0` for a strictly
  linear chain).
- **1/f noise**: angular-frequency convention `ω = 2πf`, cutoff
  `fc = 2π/(N·dt)`.

## Flicker (1/f) noise

`flicker_model.sim_noise` draws gain-noise realizations with power spectrum
parametrized by `[f0, fc, alpha]`: `f0` sets the knee scale, `fc` the
low-frequency cutoff, and `alpha` the spectral slope. Realizations are
correlated in time and independent across frequencies; in MPI runs a single
realization is drawn on rank 0 and broadcast so all ranks share it.
