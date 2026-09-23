# `limTOD.mmode` — m-mode solver for drift scans

Earth rotation turns a single dish's beam-weighted sky integral into a Fourier
series in local sidereal time (Zhang, *M-mode RIME explicit in beam, fringe and
sky modes*, eqns 13–14, fringe set to unity):

    d(nu, t) = sum_m d_m(nu) e^{i m (t - t_ref)},      d_m = sum_{l >= |m|} T_lm B_lm

The data are real, so `d_{-m} = conj d_m` and the real form is

    d(t) = d_0 + 2 sum_{m>=1} [ Re d_m cos(m dt) - Im d_m sin(m dt) ],   dt = t - t_ref.

`Re d_m` and `Im d_m` are the cosine and sine amplitudes of the scan: the
quantities linear in the sky. This package solves time-ordered data for them.

## Quick use

```python
import numpy as np
from limTOD import mmode

window = mmode.LSTWindow.uniform_arc(lst_start_deg=0.0, span_deg=360.0, cadence_s=120.0)
kernel = mmode.dpss_kernel(length=181, nw=4.0)          # optional
res = mmode.solve(tod, window, m_trunc=20, kernel=kernel, sigma=0.05)

res.d0                # (n_freq,) the monopole mode
res.std_d0()          # 1-sigma on d_0 per channel; res.std_mode(m) for (Re, Im) of mode m
res.cond              # conditioning of the normalised design: mode confusion
res.modes             # (n_freq, M+1) complex
```

Frequency smoothness is applied *after* the time solve:

```python
basis = mmode.spline_basis(freqs, n_par=12)
d0_smooth = mmode.project_onto_basis(res.d0, basis)
```

## What the window decides

| property | meaning |
|---|---|
| `coverage` | fraction of the sidereal circle with a sample |
| `rayleigh` | `2 pi / T`: the narrowest feature in `m` the arc resolves |
| `shannon_number(M)` | how many of the `2M+1` real unknowns the window constrains |
| `closed` | the arc meets itself: smoothing is periodic, no edge is lost |

On a closed scan every mode is separable (`cond ~ 1.4`) and no kernel is
needed. On a 7 h arc (29% coverage) the low modes are not separable at any
truncation: the weakest singular direction is `d_0` against the low cosines,
because `1, cos t, cos 2t` are nearly the same parabola over a short arc. This
is not a numerical problem and no regularisation supplies the missing
information; `noise_amplification` puts a number on it (`~1e8` at `m <= 8`).
Stacking nights closes the circle at about 260 nights for a 7 h window
(`LSTWindow.stacked_nights`).

## Smoothing

Away from the edges a convolution has the Fourier modes as eigenfunctions,
`k * e^{imt} = K(m) e^{imt}`, so smoothed data obey a truncated model exactly
up to what `K(m)` lets through above the truncation. Half a kernel is dropped
at each end of an open arc; a scan that closes on itself (last sample exactly
one cadence before the first) is convolved periodically and loses nothing.
Choose `m_trunc` at or above the kernel's band edge
(`kernel.band_edge(window, nw)`), otherwise the unmodelled modes are
attributed to the low ones. On a partial arc a longer kernel costs more in
discarded edges than it saves in leakage.

**The estimator is weighted OLS on the smoothed data, on purpose.** Its
covariance is the sandwich `P C P^T` with `C` the covariance the kernel
induces. Generalised least squares with that same `C` is *not* used by
default: whitening with `S S^T` divides `K(m)` back out, and the estimator
collapses to the raw-data solve with the discarded edges removed -- the
kernel's suppression is gone. So the smoothed solve trades some noise for
insensitivity to unmodelled modes; `weighting="gls"` exists for a supplied
correlated covariance (1/f noise) on an unsmoothed solve.

## Error budget

* **noise** — `LinearResult.std_d0()` / `std_mode(m)`, the sandwich
  covariance with `C = S diag(1/w) S^T` for sample weights `w`, or a supplied
  `noise_cov` (e.g. with 1/f noise added) used as given.
* **conditioning** — `LinearResult.cond` on normalised columns.
* **amplification** — `noise_amplification(window, M, kernel)`: the same
  estimator in extended precision, the reference the float64 path is checked
  against. `solve` takes its pseudo-inverse from the SVD of the whitened
  design, never from the normal matrix (whose float64 form has already lost
  the small singular values), and switches to extended precision above a
  condition number of 1e12; above 1e8 it warns that the estimate is
  noise-dominated.
* **leakage** — `leakage_bound(window, M, envelope, kernel)`: the worst-case
  bias on `d_0` from modes above `M`, from an envelope of `|d_m|` you supply.
  The data cannot measure this term.

## Bayesian route (`limTOD.mmode.bayes`, optional)

`pip install limTOD[mmode-bayes]`. Every mode up to `mmax` becomes a latent
with a Gaussian prior of width `envelope[m]`; the raw samples are observed with
independent noise; the prediction is affine, which bayesmith certifies and
solves exactly (`exact_posterior` for the Wiener mean, `posterior_samples` for
constrained-realisation draws and credible intervals). This replaces truncation
and smoothing with a prior. Along directions the window does not constrain the
posterior *is* the prior, which is the honest statement of the same limit.

## Conventions

* `t_ref_deg` is the LST at which the beam harmonics were evaluated.
* Weights are inverse variances (visit counts for stacked nights) and must be positive.
* LST arrays are in time order; one downward jump of more than half a turn is
  read as a 0 h crossing. Anything else is not time-ordered: no kernel can be
  applied, and the geometry falls back to the plain extent of the samples.
* `coverage` counts revisits of one LST once and credits each sample at most
  one nominal spacing (the median gap, at least 1 deg).
* `tod` must be finite; flag bad samples and drop them from the window.
* Modes at or beyond the sampling Nyquist (`window.nyquist_m = 180 / spacing`)
  are refused everywhere: `K(m)` aliases there and so do the modes themselves.
* Above a condition number of ~1e12 the amplified round-off of float64 data
  dominates the estimate; above ~1e16 the float64 design matrix itself no
  longer determines the answer and no extended precision recovers it.
* `weighting="gls"` refuses a covariance conditioned worse than 1e12 and a
  whitened solve that has lost positive definiteness: a long DPSS kernel makes
  `S S^T` singular, and whitening with it is meaningless.
* A kernel that nulls a mode below the truncation to machine precision is
  refused with the mode named; lower `m_trunc` below it.
* Kernels are symmetric and unit-sum; asymmetric kernels are refused because
  they shift every phase.
