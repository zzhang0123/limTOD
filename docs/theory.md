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

The beam enters limTOD as a HEALPix map (RING ordering;
`(θ, φ) = healpy.pix2ang`). A beam's orientation is only meaningful
relative to the **local horizontal system**, so the convention is
defined through it — using only the coordinate functions azimuth and
elevation, never observer-dependent words like "left/right/up/down"
(which compass direction a beam meridian maps to depends on the
pointing). Every claim below is pinned numerically by
[`tests/test_beam_orientation.py`](https://github.com/zzhang0123/limTOD/blob/main/tests/test_beam_orientation.py)
(displaced-blob probes through the full pointing chain).

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
  $\varphi = 90°$ along RA $= 90°$.
- **Zenith pointing** ($e = 90°$, reached with mount azimuth $A$): the
  formulas remain valid and unambiguous —
  $\hat e_{\mathrm{az}} = \cos A\, \hat E - \sin A\, \hat N$ (unchanged)
  and $\hat e_{\mathrm{el}}(A, 90°) = -(\sin A\, \hat E + \cos A\, \hat N)$,
  the continuous carry-over of the mount's approach azimuth.
- **Symmetric beams** are insensitive to $\varphi$ entirely — which is
  why this section matters only for asymmetric beams and polarization
  work (and why the orientation went unstated for so long: symmetric
  cross-checks cannot detect it).

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
