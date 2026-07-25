# Theory and conventions

Full derivations live in
[conventions.pdf](https://github.com/zzhang0123/limTOD/blob/main/conventions.pdf)
(coordinate systems, Euler-angle conventions, spherical-harmonic
formulations, beam convolution, noise models). This page is the working
summary.

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
   coordinates is rotated with the pointing-independent part of the chain
   (azimuth = 0, elevation = 0 ⇒ ψ′ = α, θ′ = β, φ′ = 0), thresholded at
   0.5, and applied to the pointed beam.
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
