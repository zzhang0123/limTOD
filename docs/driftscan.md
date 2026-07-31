# Drift scans in harmonic space — the m-mode path

`limtod_jax.driftscan` is the drift-scan special case of the sky→TOD
simulator: fixed azimuth/elevation/self-rotation at a fixed site, only the
LST advancing. Where the generic path
({func}`limtod_jax.core.generate_tod_sky`) performs one Wigner rotation of
the beam **per time sample** — $O(n_\mathrm{time}\,\ell_\mathrm{max}^3)$ —
the drift-scan geometry collapses the whole scan onto **one** rotation
plus per-$m$ phases. It is pure JAX + [equinox](https://docs.kidger.site/equinox/)
(jit/vmap/grad-safe, an `equinox.Module` operator), designed to drop
straight into
[replicant-telescope](https://github.com/zzhang0123/replicant-telescope)
pipelines.

## Formalism

m-mode analysis is the standard harmonic treatment of drift-scan (transit)
observations: the scan repeats every sidereal day, so the time-ordered data
is a Fourier series in LST whose coefficients — the m-modes — are what the
sky couples to.

The conventions and equation numbering used on this page follow *M-mode RIME
explicit in beam, fringe and sky modes*
([reference note](https://zh-zhang.com/myNotes/MmodeNote.pdf), Jan 2024),
specialized to its "MT interpretation" with the fringe term $\equiv 1$: for
a single-dish autocorrelation the modulated beam **is** the primary beam.

Earth rotation is a rotation about the celestial pole, so the
celestial-frame beam coefficients at any LST differ from those at a
reference LST only by a phase:

$$
B_{\ell m}(\mathrm{lst}) \;=\; B_{\ell m}(\mathrm{lst}_\mathrm{ref})\,
e^{-i m \Delta},
\qquad \Delta = \mathrm{lst} - \mathrm{lst}_\mathrm{ref}\ \text{[rad]} .
$$

The TOD is then a Fourier series in $\Delta$ (eqns 13–15 of the note):

$$
V(\Delta) = \tilde V_0 + 2\,\mathrm{Re}\sum_{m\ge 1} \tilde V_m\,
e^{+i m \Delta},
\qquad
\tilde V_m = \sum_{\ell} \overline{B_{\ell m}(\mathrm{lst}_\mathrm{ref})}\,
\tilde S_{\ell m} .
$$

The $\tilde V_m$ are the **m-modes** — the Fourier coefficients of the
sidereal-day-periodic TOD — and the sky→m-mode projection is a single
per-$(\ell,m)$ product with the reference-frame beam. Both phase signs
above are **locked numerically** against the generic path (never trusted
on paper — see `tests/limtod_jax/test_driftscan.py`); the two paths agree
to ~$10^{-15}$ relative in float64, including the `normalize` branch and
the exact adjoint.

Cost: one $O(\ell_\mathrm{max}^3)$ Wigner rotation total, then an
$O(n_\mathrm{time}\,\ell_\mathrm{max})$ phase synthesis with
$O(n_\mathrm{time}+\ell_\mathrm{max})$ memory.

## Polarisation

Full Stokes is supported, and it costs the formalism **nothing structural**.
Pass the static `npol` (1, 3 or 4) and give the alms a leading Stokes axis
`(npol, n_alm)` holding the packed $T,E,B[,V]$ rows — the healpy analysis of
limTOD's $I,Q,U[,V]$ maps, exactly what `hp.map2alm(beam_map)` returns (with
$V$ analysed separately, as numpy limTOD does, because it is spin-0 and must
not ride along in the spin-2 transform):

```python
beam_alm = jnp.asarray(hp.map2alm(beam_map_iqu, lmax=lmax))   # (3, n_alm)
op  = ltj.DriftScanMmode.from_pointing(beam_alm, lst_deg, lat, az, el, lmax=lmax)
tod = op(sky_alm)          # (n_time,) — the Stokes rows CONTRACT into each sample
```

`npol` is inferred from a 2-D `beam_alm` on `from_pointing` and is explicit
everywhere else (see below).

### Why it is free

Three facts, each locked numerically in
`tests/limtod_jax/test_polarisation.py`:

1. **Spin-weighted alms rotate with the same Wigner-$D$ as scalar ones** — the
   spin index never enters the $m$-mixing, and $E$/$B$ do not mix under
   rotation (they are a *parity* split, not a rotational one). healpy's 3-row
   `rotate_alm` is therefore **bit-identical** to the scalar rotation applied
   per row, and `limtod_jax` needs no spin-2 rotation anywhere.
2. **Spin-2 harmonics are orthonormal**, so the pixel dot splits row-wise,
   $\sum_p (Q_bQ_s + U_bU_s) = \sum_{\ell m}[\overline{E_b}E_s +
   \overline{B_b}B_s]$ — the $E$-$B$ cross terms are purely imaginary under
   the real-field symmetry and drop. The quadrature-alm exactness contract
   carries over unchanged, per row.
3. **A rotation about $z$ gives $D^\ell_{m'm}(\alpha,0,0) = \delta_{m'm}
   e^{-im\alpha}$ independently of spin**, so $E$ and $B$ pick up the *same*
   drift phase as $T$.

Consequently the only change to the formalism is one extra sum:

$$
\tilde V_m = \sum_{\mathrm{row}}\sum_{\ell}
\overline{B^{\mathrm{row}}_{\ell m}(\mathrm{lst}_\mathrm{ref})}\,
\tilde S^{\mathrm{row}}_{\ell m}.
$$

Same phase law, same synthesis, same FFT fast path (the rows are contracted
*before* the synthesis, so it still sees one $(\ell_{\max}+1)$ m-mode series),
same block-diagonal-in-$m$ structure. The $\ell$-rotation is shared across
rows — one Risbo recursion, not $n_\mathrm{pol}$ of them — so a 3-row beam
costs ~1× the $O(\ell_{\max}^3)$ step, not 3×.

`normalize` divides every row by the rotated **Stokes-I** beam's pixel sum,
matching numpy `pointing_beam_in_eq_sys` (and necessarily so: a beam with zero
net $Q$ would make a per-row normalizer singular). The adjoint maps a scalar
TOD back to a full `(npol, n_alm)` sky increment.

### Verification

`driftscan_tod` == generic `generate_tod_sky` == numpy
`limTOD.simulator.generate_TOD_sky` to float64 roundoff, for
$n_\mathrm{pol} \in \{1,3,4\}$, with and without `normalize`, on direct and
FFT synthesis, including a zenith-gimbal pointing, a pole latitude, a
near-horizon pointing, and $E$-only / $B$-only beams:

| leg | npol=1 | npol=3 | npol=4 |
|---|---|---|---|
| m-mode vs generic JAX | ~1e-15 | ~6e-15 | ~1e-15 |
| m-mode vs numpy oracle, nside 8 | 5.9e-16 | 1.3e-15 | 6.0e-16 |
| m-mode vs numpy oracle, nside 16 | 8.8e-16 | 2.4e-15 | 5.7e-16 |
| adjoint dot-test (12 cells) | ≤2.6e-14 | ≤1.9e-14 | ≤6.8e-14 |

(Roundoff, so seed- and resolution-dependent at the stated order; the suite
asserts 1e-12 / 1e-10 bounds rather than these values.)

The numpy leg runs through healpy's own polarised transforms end to end, so
it is a genuinely independent check — and it is the **first independent
oracle numpy limTOD's full-Stokes chain has had** (it was previously pinned
only by physical invariants; see the header of
`tests/test_stokes_and_boundaries.py`).

### Opt-in, and why it is never inferred

At the functional layer `npol` is a static argument defaulting to `None`
(= unpolarised, every leading axis a batch axis, unchanged bit for bit). It is
**not** inferred from the array shape, because a Stokes axis (contracted) and
a frequency axis (passed through) are both leading axes and shape alone cannot
tell them apart — inferring would turn a 3-frequency stack into a silently
wrong Stokes contraction. `DriftScanMmode.from_pointing` *does* infer it, but
only because 2-D `beam_alm` was previously a hard error there, so no existing
shape changes meaning; a row count outside {1, 3, 4} is rejected by name.

### Coverage

| | polarised? | note |
|---|---|---|
| `driftscan_tod` / `mmodes_from_sky` / adjoint / `DriftScanMmode` | ✅ | pure harmonic, row-wise, free |
| `generate_tod_sky` / `generate_tod_sky_adjoint` | ✅ | same |
| `horizon_truncated_beam` | ✅ | map space, but the taper is a real scalar |
| `horizon_beam_fraction` | ✅ | $f_\mathrm{sky}$ is a Stokes-I solid-angle split |
| `horizon_masked_beam_alm` | ✅ | needs genuine spin-2 transforms — read on |
| `generate_projection_rows` | ❌ raises | a projection row *is* a pixel-space beam |

Everything except the projection builder is polarised. The masked beam is the
one place polarisation is **not** free, because it is the one place the chain
leaves harmonic space: the mask multiplies $(I,Q,U)$ *maps*, so carrying it
back to $(T,E,B)$ needs a real spin-2 synthesis and analysis.

### The spin-2 trap (read before touching `limtod_jax.hpx`)

**s2fft's on-the-fly (Price–McEwen) recursion is wrong at spin ≠ 0 on the
HEALPix grid.** It renormalises with $\log|d^\ell_{mn}|$ and $1/|d^\ell_{mn}|$
and accumulates with `nansum`; wherever a Wigner-$d$ is *exactly* zero at a
ring colatitude this gives $\log 0 = -\infty$ and $0\cdot\infty = $ NaN, and
the entire $\ell$ term is silently dropped. HEALPix rings sit at rational
$\cos\theta$, so those exact zeros really occur — at $\cos\theta = \pm 2/\ell$,
killing $\ell = 3, 6, 8, 16, 32, 48\ldots$ depending on nside. MW/GL/DH
sampling never triggers it; neither does spin 0, which is why the rest of the
package was always fine.

Measured cost of using it anyway: **4–6 % on the synthesis**, tens of percent
on the affected multipoles alone, and a silent $10^{-3}$–$10^{-2}$ break of the
pixel-dot = alm-dot exactness contract. `recursion="auto"` routes back to the
same code.

`limtod_jax.hpx` therefore uses the **precompute** transforms with
`recursion="risbo"`, which reproduce healpy to ~$10^{-14}$ (verified against
healpy and against an independent brute-force Wigner-$d$ evaluator). The
price is a dense $(n_\theta, L, 2L-1)$ complex128 kernel — $O(n_\mathrm{side}
\ell_{\max}^2)$:

| nside / $\ell_{\max}$ | 8 / 23 | 16 / 47 | 32 / 63 | 128 / 255 | 256 / 511 |
|---|---|---|---|---|---|
| kernel | 0.5 MB | 4.4 MB | 16 MB | ~1 GB | ~8.6 GB |

It is cached per `(L, spin, nside, forward)` and masking is a one-off beam
preparation, so this is usually paid once — but it does cap the resolution at
which a polarised beam can be masked inside JAX. The spin-2 path also requires
`nside ≥ 2` and `lmax + 1 ≥ 2·nside` (s2fft asserts both without a message;
`limtod_jax` raises a stated `ValueError` instead).

`tests/limtod_jax/test_polarisation.py::test_onthefly_backend_really_is_broken`
pins the defect: if it ever starts failing, s2fft has fixed its recursion and
the precompute kernel — with its memory bill — can be dropped.

At a **zenith** pointing, prefer `horizon_truncated_beam` regardless: also
fully polarised, exact rather than band-limited, ~8× cheaper, and it needs no
spin-2 machinery at all.

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)   # before any jax array

import numpy as np, healpy as hp, jax.numpy as jnp
import limtod_jax as ltj

nside, lmax = 128, 256
beam_alm = jnp.asarray(hp.map2alm(beam_map, lmax=lmax))          # beam-local
sky_alm  = ltj.map2alm_quad(jnp.asarray(sky_map), nside=nside, lmax=lmax)

op = ltj.DriftScanMmode.from_pointing(
    beam_alm,
    lst_deg=np.linspace(0.0, 360.0, 3600, endpoint=False),
    lat_deg=-30.7, az_deg=41.5, el_deg=52.5,
    lmax=lmax,
)
tod    = op(sky_alm)          # (n_time,) — == generate_tod_sky to roundoff
vm     = op.mmodes(sky_alm)   # (lmax+1,) complex m-modes Ṽ_m
sky_at = op.adjoint(tod)      # exact transpose (map-making normal equations)
```

`DriftScanMmode` is an `equinox.Module` (a frozen pytree): jit it with
`eqx.filter_jit`, batch skies with `jax.vmap(op)`, embed it as a field of
larger models, and differentiate through construction (the reference
rotation is traced, so gradients w.r.t. pointing angles and beam alms
work).

:::{admonition} 🧭 The no-rotation sanity check — azimuth 0, not 180
:class: important

The first thing most people try is
multiplying a beam map by a sky map directly, `np.sum(beam * sky)` — and
that plain product is a *specific* pointing in this convention, not a
convention-free operation. It is the identity of the rotation chain:
an antenna at the terrestrial **North Pole** (`lat_deg=90`) looking at its
**zenith** (`el_deg=90`) with **`az_deg=0`**, at `lst_deg=0`, `selfrot=0`.

Azimuth `0`, *not* `180`. At `el = 90°` the boresight is the zenith
whatever the azimuth is, so azimuth no longer picks a direction — it only
*rolls* the beam about the boresight, and the identity is the zero roll.
(The tempting `180` comes from the `φ = 0` axis landing toward the **south**
point at `az = 0`; that is the direction the axis points, not the mount
azimuth.) More generally the identity there is `az_deg = lst_deg + selfrot`.
Pinned in `tests/test_beam_orientation.py`; full statement in
[theory.md](theory.md#special-cases).
:::

### Uniform sampling: FFT synthesis

When the LST grid is uniform over a full sidereal turn, the phase sum is
an inverse real FFT and the m-mode analysis a forward one —
$O(n_t\log n_t)$ **independent of $\ell_{\max}$**. Opt in with the static
`uniform_sampling` flag (or `uniform=True` on the functional API):

```python
op = ltj.DriftScanMmode.from_pointing(
    beam_alm, lst_deg=np.linspace(0.0, 360.0, 8640, endpoint=False),
    lat_deg=-30.7, az_deg=41.5, el_deg=52.5, lmax=lmax,
    uniform_sampling=True,          # requires 2*lmax < n_time
)
```

Measured per-call synthesis cost (float64, CPU), FFT vs direct sum:

| $\ell_{\max}$ | $n_t$ = 8 640 | $n_t$ = 86 400 |
|---|---|---|
| 64 | 32.1 → 1.03 ms (31×) | 37.4 → 1.94 ms (19×) |
| 128 | 39.2 → 1.00 ms (39×) | 47.8 → 1.78 ms (27×) |
| 256 | 54.4 → 1.06 ms (51×) | 59.0 → 1.94 ms (30×) |

(Whole-operator calls, guard included — see below.)

The direct sum stays the **default** because it is exact on *arbitrary*
sampling — real scans have gaps, flags and irregular cadence, where the
FFT identity simply does not hold. The choice is deliberately static and
never auto-detected: dispatching on the *values* of `dphi` would be
impossible under `jit` (they are traced), so uniformity is the caller's
assertion. The contract is enforced in **two layers**, because getting this wrong is
otherwise silent and severe (a uniform *half*-turn grid — the normal shape
of a real observation — produced a 74%-wrong TOD in testing):

1. **A clear `ValueError` while the values are concrete.** Note this is not
   the same as "outside `jit`": any arithmetic inside a trace yields a
   tracer, so deriving `dphi` from a compile-time-constant LST grid already
   hides it. The rheplicant adapter therefore validates the *raw*
   `lst_deg`, which is still concrete inside the trace, and
   `limtod_jax.check_uniform_grid` is public for other adapters to do the
   same at their own boundary.
2. **NaN, never a plausible wrong number.** When the grid genuinely is a
   traced argument, a pure-JAX guard (no host callback, no dispatch on
   traced values, `vmap`-per-row) replaces the output with NaN. It costs
   30–45% of the raw FFT call — leaving a 10–100× net win over the direct
   sum — and is not optional: a fast path that can silently lie is not a
   fast path.

The sampling-theorem condition $2\ell_{\max} < n_t$ is always enforced as a
shape statement, and the uniformity tolerance is `64·eps(dtype)·max(2π, |Δ|)`
— scaled to the input dtype, because a genuinely uniform float32 grid
carries ~3e-7 rad of `deg2rad` representation error which must not read as
irregular. Two consequences worth knowing:

* **Never upcast a grid before checking it.** An f32 degree grid cast to
  f64 deviates ~3e-7 rad — 3·10⁶ times the f64 bound — so the cast turns a
  legitimate grid into a rejection. Check at the native dtype.
* **In float32 the tolerance cannot be tight enough to protect the
  roundoff contract.** The admitted deviation costs ~$\ell_{\max}\cdot$tol
  radians of phase, so an f32 session can carry ~10⁻³ ($\ell_{\max}$ = 256)
  to ~10⁻² ($\ell_{\max}$ = 1024) relative TOD error, and ppm-level grid
  errors sit below its detection floor. That is the same order as the f32
  transform error the module already warns about — enable x64 for
  quantitative work.

**LST gradients on the fast path.** The uniform contract pins the grid to
the one-parameter family $\Delta_0 + 2\pi t/n$, so `dphi`'s Jacobian is a
single column at index 0, equal to the direct path's row sum — i.e. the
derivative with respect to a *global* LST shift is exact, while per-sample
`dphi` derivatives are structurally zero. That is the correct gradient for
the parametrization, not a loss: an off-grid perturbation is not a smaller
gradient but a contract violation, rejected or NaN-poisoned. Fit per-sample
timing models with the direct sum.
`mmodes_from_tod_uniform` is the matching data-side transform: one FFT
carries a measured drift-scan TOD into m-space. The functional layer
({func}`~limtod_jax.driftscan.mmodes_from_sky`,
{func}`~limtod_jax.driftscan.tod_from_mmodes`,
{func}`~limtod_jax.driftscan.driftscan_tod`,
{func}`~limtod_jax.driftscan.driftscan_tod_adjoint`,
{func}`~limtod_jax.driftscan.mmodes_from_tod`) is available when you
don't want the operator object.

Conventions are unchanged from the rest of `limtod_jax`: degrees at the
public pointing boundary, packed healpy alms (m ≥ 0, real fields),
quadrature sky alms for the exactness contract, `lmax`/`nside`/`normalize`
static. Enable x64.

## The horizon mask

Physically, the beam of a ground-based drift scan is the free-space beam
**cut at the horizon** — the antenna cannot see through the ground. The
m-mode formalism handles this naturally: the masked beam is still static
in the antenna frame, so it just replaces the beam alms.
{func}`~limtod_jax.driftscan.horizon_masked_beam_alm` builds it inside
JAX: rotate the beam-local alms into the horizontal frame (where the
horizon is the pure-colatitude circle $\theta = 90°$), multiply by an
elevation taper, re-analyze (healpy-equivalent iterative `map2alm`,
{func}`limtod_jax.hpx.map2alm_iter`), rotate back. The result lives in
the **beam-local frame**, so it drops into either the drift-scan path or
the generic path unchanged:

```python
masked = ltj.horizon_masked_beam_alm(
    beam_alm, az_deg=41.5, el_deg=10.0,
    nside=nside, lmax=lmax, apod_deg=5.0,
)
op = ltj.DriftScanMmode.from_pointing(masked, ...)
# or in one step:
op = ltj.DriftScanMmode.from_pointing(
    beam_alm, ..., nside=nside, horizon_mask=True, apod_deg=5.0,
)
```

Masking is **off by default** so the module reproduces numpy limTOD
(which does not mask) exactly.

### How bad is the ringing? (and when to care)

A sharp cut is not band-limited, so representing the masked beam at
finite $\ell_\mathrm{max}$ produces Gibbs ringing. The study script
[`docs/driftscan_ringing_study.py`](https://github.com/zzhang0123/limTOD/blob/main/docs/driftscan_ringing_study.py)
quantifies the effect end-to-end (reference: pixel-space TOD with the
sharply masked beam at $N_\mathrm{side}=256$; sky band-limited at
$\ell=150$ so the sky side is exact). Headline numbers (relative RMS TOD
error from the harmonic representation of the masked beam):

| scenario | mask matters? | hard cut | apod 2° | apod 5° |
|---|---|---|---|---|
| narrow beam (2° FWHM), any el ≥ 10° | no (~10⁻⁶) | — | — | — |
| wide beam (25° FWHM), el = 41° | ~10⁻⁴ | 6·10⁻⁶ @ ℓ≤192 | 2·10⁻⁶ | 4·10⁻⁷ |
| wide beam (25° FWHM), el = 10° | **yes (~30%)** | **6·10⁻³ @ ℓ≤192, plateaus ~4·10⁻³ @ ℓ≤384** | 1.8·10⁻³ → 3.6·10⁻⁴ @ ℓ≤384 | 1.5·10⁻⁴ → 3.6·10⁻⁵ @ ℓ≤384 |

Reading of the table:

- **Narrow beams don't care.** A few-degree beam at el ≥ 10° has
  ~$10^{-6}$ of its TOD response below the horizon; skip the mask (or
  apply it — it changes nothing).
- **Wide beams at low elevation must mask** (the cut carries ~30% of the
  TOD), and there a **hard cut leaves a ~0.5% TOD error that does not
  converge away with $\ell_\mathrm{max}$** — the discontinuity aliases
  into the analysis step, so raising the band-limit alone stalls.
- **Apodization is the effective mitigation**: a 2–5° cosine taper
  (`apod_deg`) drops the error by 1–2 orders of magnitude at fixed
  $\ell_\mathrm{max}$, at the price of slightly reshaping the beam within
  the taper band (a *physical* modelling choice — real horizons aren't
  knife edges either).
- **Don't over-apodize**: keep `apod_deg` well below
  (elevation − beam radius), or the taper starts eating the main beam
  (visible in the study as a jump in the "mask matters" column for the
  narrow beam at el = 10°, apod 10°).

A compact regression version of this study is pinned in the test suite
(`test_ringing_apodization_mitigates`).

## Where the cost actually sits

Measured per frequency (float64, CPU), so that the right thing gets
optimized:

| stage | $\ell_{\max}$ = 64 | 128 | 256 | frequency |
|---|---|---|---|---|
| beam rotation to $t_{\rm ref}$ | 2.3 ms | 17.0 ms | 117.6 ms | **once** |
| m-mode projection $\tilde V_m$ | 31 µs | 58 µs | 191 µs | per evaluation |
| synthesis, direct ($n_t$ = 86 400) | 5.7 ms | 11.4 ms | 23.1 ms | per evaluation |
| synthesis, FFT | 0.45 ms | 0.42 ms | 0.46 ms | per evaluation |
| sky `map2alm` (if the sky is a map) | 5.1 ms | 29.9 ms | 175 ms | per evaluation |
| generic path, same TOD | 696 s | 5 236 s | 38 115 s | per evaluation |

Three things to read off it:

1. **The rotation is the only $O(\ell_{\max}^3)$ step and it happens once** —
   that is the whole point of the reference-time construction. Build the
   operator (or call `to_reference_frame()` on the rheplicant projector)
   *outside* your inference loop and each evaluation drops by 2–3 orders
   of magnitude.
2. **The projection is genuinely free** (microseconds), but the *direct*
   synthesis is not: at $\ell_{\max}$ = 64 over a full day it costs more
   than the rotation itself. `uniform_sampling=True` removes it.
3. **Once the rotation is amortized, the pixel↔harmonic transform becomes
   the bottleneck** — at $\ell_{\max}$ = 256 a single sky `map2alm` (175 ms)
   exceeds the rotation. The m-mode formalism wants the sky in *harmonic*
   space on both sides (eqn 14 is literally a map from $T_{\ell m}$ to
   $\tilde V_m$); parameterize it there and that cost disappears entirely.

Beyond raw cost, the operator is **block-diagonal in $m$**: $\tilde V_m$
depends only on $\tilde S_{\ell m}$ at the same $m$, so map-making normal
equations decouple into $\ell_{\max}+1$ independent small systems instead
of one large one. `mmodes_from_sky` / `mmodes_from_tod` expose exactly that
structure.

One caveat for large band-limits: the $\ell$ loop in the Wigner kernel is
Python-level, so it unrolls into the jaxpr — **compile time grows linearly
in $\ell_{\max}$** (2.6 / 5.5 / 13.7 s at $\ell_{\max}$ = 64 / 128 / 256),
and reverse-mode AD through the rotation stores $O(\ell_{\max}^3)$ of
intermediates. Neither matters when the rotation is cached; both do when
you differentiate w.r.t. the beam.

## When to use which path

| | generic `generate_tod_sky` | `DriftScanMmode` | `DriftScanMmode(uniform_sampling=True)` |
|---|---|---|---|
| pointing | arbitrary per-sample | fixed az/el/selfrot (drift) | same, plus uniform full-turn LSTs |
| build cost | — | $O(\ell_\mathrm{max}^3)$ once | $O(\ell_\mathrm{max}^3)$ once |
| per-evaluation | $O(n_t\,\ell_\mathrm{max}^3)$ | $O(n_t\,\ell_\mathrm{max})$ | $O(n_t\log n_t)$ |
| agreement | — | equal to roundoff | equal to roundoff |
| polarisation | `npol=1/3/4` | `npol=1/3/4` | `npol=1/3/4` |
| m-modes | not exposed | {func}`~limtod_jax.driftscan.mmodes_from_sky` / `op.mmodes` | same, plus `mmodes_from_tod_uniform` |

Anything that is a genuine drift scan should use this path; tracking or
scanning strategies need the generic one.
