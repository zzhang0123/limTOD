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
**relative to the local horizontal system** — any definition that does
not reference it is ambiguous — so the convention is stated in those
terms and every claim below is pinned numerically by
[`tests/test_beam_orientation.py`](https://github.com/zzhang0123/limTOD/blob/main/tests/test_beam_orientation.py)
(displaced-blob probes through the full pointing chain).

**Horizontal system.** Azimuth $A$ runs from North ($0°$) through East
($90°$); elevation $E$ from the horizon ($0°$) to the zenith ($90°$).
At the pointing direction $\hat b(A, E)$ define two tangent directions:

- $\hat e_\mathrm{up}$ — direction of **increasing elevation** (along
  the great circle toward the zenith);
- $\hat e_\mathrm{right}$ — direction of **increasing azimuth** (the
  right-hand side when you stand at the antenna facing the pointing,
  head up; East when facing North).

$(\hat e_\mathrm{up}, \hat e_\mathrm{right}, \hat b)$ is right-handed:
$\hat e_\mathrm{up} \times \hat e_\mathrm{right} = \hat b$.

**The convention.** For a pointing $(A, E)$ with self-rotation $\psi$
(`selfrot_deg`):

- the **beam centre (boresight) is the beam map's north pole**,
  $\theta = 0$; $\theta$ is the angular distance from the boresight;
- the map point $(\theta, \varphi)$ is carried to the sky direction at
  angular distance $\theta$ from $\hat b$ along the tangent direction

$$
\hat t(\varphi) \;=\; \cos(\varphi + \psi)\, \hat e_\mathrm{up}
              \;+\; \sin(\varphi + \psi)\, \hat e_\mathrm{right}.
$$

Equivalently, with $\psi = 0$:

| beam-map meridian | side of the beam |
|---|---|
| $\varphi = 0$ | **up** (increasing elevation, toward the zenith) |
| $\varphi = 90°$ | **right** (increasing azimuth; East when facing North) |
| $\varphi = 180°$ | down (toward the horizon) |
| $\varphi = 270°$ | left |

Positive `selfrot_deg` rotates the pattern about the boresight in the
$\varphi$-increasing sense (a feature at meridian $\varphi$ moves from
the up side toward the right side).

### Special cases

- **Identity / reading the map as equatorial.** For
  `lat = 0°, LST = 0°, A = 0° (North), E = 0° (horizon)`, $\psi = 0$,
  the full rotation chain is the identity: the beam map can be read
  directly as an equatorial map — beam centre at the **north celestial
  pole**, $\varphi \equiv$ RA. The $\varphi = 0$ meridian runs along
  RA $= 0$ toward this observer's zenith at (RA $0°$, Dec $0°$).
  Beware the classic trap: at the pole, "chart South" (decreasing Dec)
  is the beam's **up** side in this configuration — sky-chart down and
  antenna down are opposite things here.
- **Zenith pointing** ($E = 90°$, reached with mount azimuth $A$): the
  tangent frame is carried continuously over the top —
  $\hat e_\mathrm{up}$ ends along the horizontal direction of azimuth
  $A + 180°$ and $\hat e_\mathrm{right}$ along azimuth $A + 90°$. The
  formula above stays unambiguous; only the everyday words "up/right"
  lose their meaning.
- **Symmetric beams** are insensitive to $\varphi$ entirely — which is
  why this section matters only for asymmetric beams and polarization
  work (and why the orientation went unstated for so long: symmetric
  cross-checks cannot detect it).

The patch-beam path uses the same two tangent axes: its direction
cosines are $(l, m) = (\sin$-projected $\hat e_\mathrm{right},
\hat e_\mathrm{up})$ components — see
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
