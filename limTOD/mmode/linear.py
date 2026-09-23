"""The linear m-mode solve: design, per-channel least squares, error budget.

Real parametrisation. The data are real, so ``d_{-m} = conj(d_m)`` and::

    d(t) = d_0 + 2 sum_{m>=1} [ Re d_m cos(m dt) - Im d_m sin(m dt) ],   dt = t - t_ref

The unknown vector per channel is ``[d_0, Re d_1, Im d_1, ..., Re d_M, Im d_M]``.

Three facts shape what is offered here:

* Ordinary least squares on the *smoothed* data is what suppresses the modes
  above the truncation. Generalised least squares with the covariance the
  kernel induces undoes the kernel (see :mod:`limTOD.mmode.kernel`), so the
  default estimator is weighted OLS and its covariance is the sandwich
  ``P C P^T`` with ``P = (E^T W E)^-1 E^T W`` and ``C`` the true covariance of
  the smoothed samples. ``weighting="gls"`` is there for a supplied
  correlated covariance (1/f noise) on an unsmoothed solve.
* With each ``d_m(nu)`` on a frequency basis ``Phi`` the joint design is a
  Kronecker product ``E (x) Phi``, so the joint solution is the per-channel
  solve *followed by* a projection onto ``span(Phi)``
  (:func:`project_onto_basis`); the basis never enters the design.
* On a partial arc the small singular values of ``E`` sit far below double
  precision. The pseudo-inverse is taken from the SVD of the *whitened design*
  (accurate to ``eps * cond`` relative on those values) and never from the
  normal matrix, whose float64 form has already lost them. Past a condition
  number of 1e12 the normal equations are formed in extended precision.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from .kernel import Kernel, identity_kernel
from .window import LSTWindow

__all__ = [
    "design_matrix", "conditioning", "unpack", "pack", "synthesise",
    "LinearResult", "solve", "noise_amplification", "leakage_bound",
    "project_onto_basis", "legendre_basis", "spline_basis",
]

#: Above this (column-normalised) condition number the normal equations are
#: formed in extended precision; below it the float64 SVD of the whitened
#: design keeps the small singular values to better than 1e-4 relative.
_EXTENDED_PRECISION_COND = 1e12
#: A design column this small relative to the largest is a mode the kernel has
#: removed: dividing it back out would amplify float64 round-off, so it is refused.
_NULLED_COLUMN = 1e-13
#: Whitening (weighting="gls") with a covariance worse conditioned than this is refused.
_GLS_COV_COND = 1e12
#: Above this the estimate itself is noise-dominated at any realistic sigma
#: and a warning says so; the reported error is still right.
_WARN_COND = 1e8


# --------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------

def design_matrix(window: LSTWindow, m_trunc: int, khat: Optional[np.ndarray] = None) -> np.ndarray:
    """``(n, 2M+1)`` real design; column ``2m-1`` is ``2 K(m) cos``, ``2m`` is ``-2 K(m) sin``."""
    if m_trunc < 0:
        raise ValueError("m_trunc must be >= 0.")
    if m_trunc >= window.nyquist_m:
        raise ValueError(f"m_trunc = {m_trunc} is at or beyond the sampling Nyquist mode "
                         f"({window.nyquist_m:.0f}): those modes alias onto lower ones.")
    ph = window.phase
    k = np.ones(m_trunc + 1) if khat is None else np.asarray(khat, dtype=float)
    if k.size < m_trunc + 1:
        raise ValueError("khat must cover m = 0 .. m_trunc.")
    cols = [np.full(ph.size, k[0])]
    for m in range(1, m_trunc + 1):
        cols += [2 * k[m] * np.cos(m * ph), -2 * k[m] * np.sin(m * ph)]
    return np.column_stack(cols)


def conditioning(design: np.ndarray) -> float:
    """Condition number with columns normalised: mode confusion, not ``K(m)`` scaling."""
    norms = np.linalg.norm(design, axis=0)
    if np.any(norms == 0):
        return float("inf")
    return float(np.linalg.cond(design / norms))


def _conditioning_of(e: np.ndarray, w: np.ndarray) -> float:
    """Conditioning of the design the solve actually inverts: weighted, then column-normalised."""
    return conditioning(e * np.sqrt(w)[:, None])


def _warn_if_degenerate(cond: float) -> None:
    if cond > _WARN_COND:
        warnings.warn(f"design condition number {cond:.1e}: the window does not separate these "
                      "modes. The reported error is right, the estimate is noise-dominated, past ~1e12 "
                      "the round-off in float64 data is amplified into it, and past ~1e16 the float64 "
                      "design itself no longer determines the answer.", stacklevel=3)


def pack(d_m: np.ndarray) -> np.ndarray:
    """Complex ``(..., M+1)`` modes -> real ``(..., 2M+1)`` parameters. ``Im d_0`` is dropped."""
    d = np.asarray(d_m)
    out = [d[..., 0].real]
    for m in range(1, d.shape[-1]):
        out += [d[..., m].real, d[..., m].imag]
    return np.stack(out, axis=-1)


def unpack(params: np.ndarray) -> np.ndarray:
    """Real ``(..., 2M+1)`` parameters -> complex ``(..., M+1)`` modes."""
    p = np.asarray(params, dtype=float)
    if p.shape[-1] % 2 == 0:
        raise ValueError("A real parameter vector has odd length 2M+1.")
    mmax = (p.shape[-1] - 1) // 2
    out = np.zeros(p.shape[:-1] + (mmax + 1,), dtype=complex)
    out[..., 0] = p[..., 0]
    for m in range(1, mmax + 1):
        out[..., m] = p[..., 2 * m - 1] + 1j * p[..., 2 * m]
    return out


def synthesise(window: LSTWindow, d_m: np.ndarray) -> np.ndarray:
    """Time-ordered data ``(n, n_freq)`` from modes ``(n_freq, M+1)`` (or ``(M+1,)``)."""
    d = np.atleast_2d(np.asarray(d_m))
    e = design_matrix(window, d.shape[1] - 1)
    return e @ pack(d).T


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _System:
    """Everything one (window, kernel, truncation) determines, built once."""

    window: LSTWindow          # the raw window
    valid: LSTWindow           # what survives the kernel
    kernel: Kernel
    khat: np.ndarray           # K(m) for m = 0 .. mmax_needed
    e: np.ndarray              # (n_valid, 2M+1) design on the valid window
    s: np.ndarray              # (n_valid, n) smoothing operator


def _system(window: LSTWindow, m_trunc: int, kernel: Optional[Kernel], mmax_needed: int) -> _System:
    kern = identity_kernel() if kernel is None else kernel
    khat = kern.transfer(window, mmax_needed) if kern.length > 1 else np.ones(mmax_needed + 1)
    win_v = kern.valid(window)
    e = design_matrix(win_v, m_trunc, khat)
    if e.shape[0] < e.shape[1]:
        raise ValueError(f"{e.shape[0]} valid samples for {e.shape[1]} unknowns: "
                         "lower m_trunc or shorten the kernel.")
    return _System(window, win_v, kern, khat, e, kern.smoothing_operator(window))


def _estimator(e: np.ndarray, w: np.ndarray, cond: float) -> np.ndarray:
    """``P = (E^T W E)^-1 E^T W`` with the small singular values kept.

    Works on the column-normalised design, so that a column the kernel has
    scaled down (``K(m)`` near zero) is divided back out exactly rather than
    floored away as if it were degenerate: ``cond`` measures mode confusion
    on the same normalised matrix, and the singular-value floor applies to
    that alone. A column that is exactly zero is refused.
    """
    norms = np.linalg.norm(e, axis=0)
    dead = norms <= _NULLED_COLUMN * norms.max()
    if np.any(dead):
        m = (int(np.argmax(dead)) + 1) // 2
        raise ValueError(f"the kernel nulls mode m={m} to machine precision (K(m) ~ "
                         f"{norms[dead][0] / norms.max():.1e}); lower m_trunc below it.")
    sw = np.sqrt(w)
    ew = (e / norms) * sw[:, None]
    if cond > _EXTENDED_PRECISION_COND:
        import mpmath as mp

        with mp.workdps(50):
            em = mp.matrix(ew.tolist())
            try:
                inv = mp.inverse(em.T * em)
            except ZeroDivisionError as exc:
                raise np.linalg.LinAlgError(
                    "the normal matrix is singular even in extended precision: some mode "
                    "combination is not constrained by this window at all.") from exc
            p = np.array((inv * em.T).tolist(), dtype=float)
    else:
        u, s, vt = np.linalg.svd(ew, full_matrices=False)
        floor = s[0] * np.finfo(float).eps * max(ew.shape) * 1e-2
        s_inv = np.where(s > floor, 1.0 / np.maximum(s, floor), 0.0)
        p = (vt.T * s_inv) @ u.T
    return (p / norms[:, None]) * sw[None, :]


def _covariance_root(cov: np.ndarray) -> np.ndarray:
    """A matrix ``R`` with ``R R^T = cov``: Cholesky when it exists, else a clipped eigen-root."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
        return vecs * np.sqrt(np.clip(vals, 0.0, None))[None, :]


@dataclass(frozen=True, eq=False)
class LinearResult:
    """What the per-channel least-squares solve returned, and its provenance."""

    modes: np.ndarray            # (n_freq, M+1) complex
    cov: np.ndarray              # (2M+1, 2M+1) parameter covariance for unit sigma
    sigma: np.ndarray            # (n_freq,) noise sigma per raw sample used
    m_trunc: int
    kernel: str
    weighting: str
    n_samples: int
    n_valid: int
    cond: float
    khat: np.ndarray
    window: LSTWindow = field(repr=False)
    residual_rms: Optional[np.ndarray] = field(default=None, repr=False)   # (n_freq,) on the smoothed data

    @property
    def d0(self) -> np.ndarray:
        return self.modes[:, 0].real

    def std_d0(self) -> np.ndarray:
        """1-sigma error on ``d_0``, per channel."""
        return self.sigma * np.sqrt(self.cov[0, 0])

    def std_mode(self, m: int) -> np.ndarray:
        """``(n_freq, 2)`` 1-sigma errors on ``(Re d_m, Im d_m)``; ``m = 0`` gives ``(n_freq, 1)``."""
        if not 0 <= m <= self.m_trunc:
            raise ValueError(f"m must lie in 0 .. {self.m_trunc}.")
        idx = [0] if m == 0 else [2 * m - 1, 2 * m]
        return self.sigma[:, None] * np.sqrt(np.diag(self.cov)[idx])[None, :]


def solve(tod: np.ndarray, window: LSTWindow, m_trunc: int, kernel: Optional[Kernel] = None,
          sigma: Union[float, np.ndarray] = 1.0, noise_cov: Optional[np.ndarray] = None,
          weighting: str = "ols") -> LinearResult:
    """Smooth, trim, and solve every channel for ``d_0 .. d_M``.

    Parameters
    ----------
    tod : (n, n_freq) or (n,)
        Raw time-ordered data on ``window``.
    window : LSTWindow
    m_trunc : int
        Truncation. Choose it at or above the kernel's band edge, or the
        modes above it are attributed to the ones below.
    kernel : Kernel, optional
        Default: none (the plain windowed-Fourier solve).
    sigma : float or (n_freq,)
        Noise per raw sample, per unit weight, per channel. Reported errors
        scale with it; the estimator does not.
    noise_cov : (n_valid, n_valid), optional
        Covariance of the *smoothed, trimmed* samples for unit sigma, already
        including the weights. Default: the kernel acting on independent
        samples of variance ``1 / weight``.
    weighting : {"ols", "gls"}
        ``"ols"`` (default): weighted least squares with the sample weights
        only, errors by the sandwich formula -- the estimator that keeps the
        kernel's suppression. ``"gls"``: whiten with the full covariance;
        undoes any kernel, and is for a supplied correlated ``noise_cov``.
    """
    if weighting not in ("ols", "gls"):
        raise ValueError("weighting must be 'ols' or 'gls'.")
    y = np.asarray(tod, dtype=float)
    y = y[:, None] if y.ndim == 1 else y
    if y.shape[0] != window.n:
        raise ValueError(f"tod has {y.shape[0]} samples, the window {window.n}.")
    nf = y.shape[1]
    sig = np.asarray(sigma, dtype=float)
    if sig.ndim > 1 or (sig.ndim == 1 and sig.size != nf):
        raise ValueError(f"sigma must be a scalar or have one value per channel ({nf}).")
    sig = np.broadcast_to(sig, (nf,)).copy()
    if np.any(sig <= 0) or not np.all(np.isfinite(sig)):
        raise ValueError("sigma must be positive and finite.")
    if not np.all(np.isfinite(y)):
        raise ValueError("tod contains NaN or inf; flag and drop those samples from the window first.")

    sysm = _system(window, m_trunc, kernel, m_trunc)
    e, s, w = sysm.e, sysm.s, sysm.valid.weights
    ys = s @ y
    if noise_cov is None:
        cov_n = (s / window.weights[None, :]) @ s.T
    else:
        cov_n = np.asarray(noise_cov, dtype=float)
        if cov_n.shape != (e.shape[0],) * 2:
            raise ValueError(f"noise_cov has shape {cov_n.shape}, expected {(e.shape[0],) * 2}.")

    cond = _conditioning_of(e, w)
    _warn_if_degenerate(cond)
    if weighting == "ols":
        p = _estimator(e, w, cond)
        # sandwich as G G^T with G = P C^{1/2}: a sum of squares, so a mode the kernel has
        # scaled far down (P ~ 1/K) cannot come out negative through cancellation
        if noise_cov is None:
            g = (p @ s) / np.sqrt(window.weights)[None, :]
        else:
            g = p @ _covariance_root(cov_n)
        cov_p = g @ g.T
    else:
        from scipy.linalg import LinAlgError, cho_factor, cho_solve

        evals = np.linalg.eigvalsh((cov_n + cov_n.T) / 2)
        if evals[0] <= 0 or evals[-1] / evals[0] > _GLS_COV_COND:
            raise LinAlgError(f"noise_cov has condition number {evals[-1] / max(evals[0], 1e-300):.1e}: "
                              "a long kernel makes the smoothed covariance singular, and whitening with "
                              "it is meaningless -- use weighting='ols' there.")
        cf = cho_factor(cov_n)
        ninv_e = cho_solve(cf, e)
        normal = e.T @ ninv_e
        cov_p = np.linalg.inv(normal)
        p = cov_p @ ninv_e.T
        if not np.all(np.isfinite(cov_p)) or np.any(np.diag(cov_p) <= 0):
            raise LinAlgError("the whitened normal equations lost positive definiteness in float64; "
                              "use weighting='ols' or a better-conditioned noise_cov.")
    params = p @ ys
    resid = ys - e @ params
    return LinearResult(
        modes=unpack(params.T), cov=cov_p, sigma=sig, m_trunc=m_trunc, kernel=sysm.kernel.label,
        weighting=weighting, n_samples=window.n, n_valid=e.shape[0], cond=cond,
        khat=sysm.khat, window=window, residual_rms=np.sqrt(np.mean(resid ** 2, axis=0)),
    )


# --------------------------------------------------------------------------
# error budget
# --------------------------------------------------------------------------

def noise_amplification(window: LSTWindow, m_trunc: int, kernel: Optional[Kernel] = None,
                        dps: int = 60) -> float:
    """Noise on ``d_0`` per unit noise per unit-weight sample, in extended precision.

    The same estimator :func:`solve` uses (weighted OLS on the smoothed data,
    sandwich covariance), evaluated with ``dps`` digits so that nothing is
    lost on a partial arc. Agrees with ``solve(...).std_d0()`` at unit sigma;
    this is the reference the float64 path is checked against.
    """
    import mpmath as mp

    sysm = _system(window, m_trunc, kernel, m_trunc)
    e, s, w = sysm.e, sysm.s, sysm.valid.weights
    _warn_if_degenerate(_conditioning_of(e, w))
    with mp.workdps(dps):
        em = mp.matrix(e.tolist())
        wm = mp.diag([mp.mpf(float(v)) for v in w])
        unit = mp.matrix(e.shape[1], 1)
        unit[0] = 1
        row = mp.lu_solve(em.T * wm * em, unit)                 # (E^T W E)^-1 e_0
        p0 = wm * em * row                                      # W E (E^T W E)^-1 e_0 = P^T e_0
        p0 = np.array([float(v) for v in p0])
    g0 = (p0 @ s) / np.sqrt(window.weights)
    return float(np.sqrt(g0 @ g0))


def leakage_bound(window: LSTWindow, m_trunc: int, envelope: Sequence[float],
                  kernel: Optional[Kernel] = None) -> float:
    """Worst-case bias on ``d_0`` from modes above the truncation.

    The data cannot tell you what the unmodelled modes did; this bounds it
    from an ``envelope[m]`` of ``|d_m|`` for ``m = 0 .. mmax`` you supply
    (a sky+beam model, or the solve's own high-m estimate at full coverage).
    Each unmodelled mode is passed through the same estimator :func:`solve`
    uses at its worst phase; the bound sums the magnitudes.
    """
    env = np.asarray(envelope, dtype=float)
    mmax = env.size - 1
    if mmax <= m_trunc:
        return 0.0
    sysm = _system(window, m_trunc, kernel, mmax)
    cond = _conditioning_of(sysm.e, sysm.valid.weights)
    _warn_if_degenerate(cond)
    p0 = _estimator(sysm.e, sysm.valid.weights, cond)[0]
    ph = sysm.valid.phase
    total = 0.0
    for m in range(m_trunc + 1, mmax + 1):
        c = p0 @ (2 * sysm.khat[m] * np.cos(m * ph))
        s = p0 @ (-2 * sysm.khat[m] * np.sin(m * ph))
        total += env[m] * np.hypot(c, s)              # worst phase of a complex d_m
    return float(total)


# --------------------------------------------------------------------------
# frequency bases, applied after the time solve
# --------------------------------------------------------------------------

def legendre_basis(freqs: np.ndarray, n_par: int) -> np.ndarray:
    f = np.asarray(freqs, dtype=float)
    u = 2 * (f - f.mean()) / np.ptp(f)
    return np.polynomial.legendre.legvander(u, n_par - 1)


def spline_basis(freqs: np.ndarray, n_par: int) -> np.ndarray:
    """Cubic B-splines with ``n_par - 4`` uniform interior knots."""
    from scipy.interpolate import BSpline

    f = np.asarray(freqs, dtype=float)
    if n_par < 4:
        raise ValueError("A cubic spline basis needs at least 4 parameters.")
    interior = np.linspace(f[0], f[-1], n_par - 4 + 2)[1:-1]
    knots = np.r_[[f[0]] * 4, interior, [f[-1]] * 4]
    return np.asarray(BSpline.design_matrix(f, knots, 3, extrapolate=False).todense())


def project_onto_basis(spectrum: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Least-squares projection of a spectrum onto ``span(basis)``.

    Because the joint design is a Kronecker product this *is* the joint
    solve. Rank-deficient bases are handled: the projection is onto the span,
    whatever the columns.
    """
    b = np.asarray(basis, dtype=float)
    s = np.asarray(spectrum)
    coef = np.linalg.lstsq(b, s, rcond=None)[0]
    return b @ coef
