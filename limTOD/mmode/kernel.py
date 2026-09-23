"""Smoothing along sidereal time, and what it does to each ``m``.

Away from the edges a convolution has the Fourier modes as eigenfunctions::

    (k * e^{imt})(t_i) = K(m) e^{i m t_i},      K(m) = sum_j k_j e^{-i m tau_j}

so smoothed data obey a *truncated* linear model exactly, up to whatever
``K(m)`` lets through above the truncation. Half a kernel is discarded at each
end of an open arc; a closed scan is convolved periodically and keeps every
sample. The kernel needs uniform, time-ordered sampling and checks for it.

What smoothing buys, and what it does not
-----------------------------------------
Ordinary least squares on the smoothed data is insensitive to the modes the
kernel suppresses: that is the point. Generalised least squares with the
covariance the kernel induces (``S S^T``) is *not* -- whitening divides
``K(m)`` back out, and the estimator collapses to the raw-data solve with the
discarded edges removed. So the smoothed solve is a deliberately non-optimal
linear estimator, robust to unmodelled modes at some cost in noise, and its
errors are reported with the sandwich formula rather than ``(E^T C^-1 E)^-1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .window import LSTWindow

__all__ = ["Kernel", "dpss_kernel", "boxcar_kernel", "identity_kernel"]


@dataclass(frozen=True, eq=False)
class Kernel:
    """A symmetric, unit-sum smoothing kernel and how it was made."""

    taps: np.ndarray
    label: str = "kernel"

    def __post_init__(self) -> None:
        k = np.asarray(self.taps, dtype=float).ravel()
        if k.size % 2 == 0:
            raise ValueError("Use an odd number of taps so the kernel has a centre sample.")
        if not np.allclose(k, k[::-1], rtol=1e-12, atol=1e-12):
            raise ValueError("Only symmetric kernels: an asymmetric one shifts every phase.")
        if abs(k.sum()) < 1e-12:
            raise ValueError("The kernel sums to zero and would erase the monopole.")
        object.__setattr__(self, "taps", k / k.sum())

    @property
    def length(self) -> int:
        return self.taps.size

    @property
    def half(self) -> int:
        return (self.taps.size - 1) // 2

    def transfer(self, window: LSTWindow, mmax: int) -> np.ndarray:
        """``K(m)`` for ``m = 0 .. mmax``; real, because the kernel is symmetric."""
        dt = self._step_rad(window)
        if self.length > 1 and mmax >= window.nyquist_m:
            raise ValueError(f"m = {mmax} is at or beyond the sampling Nyquist mode "
                             f"({window.nyquist_m:.0f}); K(m) aliases there and means nothing.")
        offsets = (np.arange(self.length) - self.half) * dt
        return np.cos(np.outer(np.arange(mmax + 1), offsets)) @ self.taps

    def band_edge(self, window: LSTWindow, nw: float) -> float:
        """``NW * 2 pi / T_kernel``: where a DPSS kernel's response starts to fall."""
        return nw * 2 * np.pi / (self.length * self._step_rad(window))

    def smoothing_operator(self, window: LSTWindow) -> np.ndarray:
        """The ``(n_valid, n)`` matrix ``S`` with ``smooth(y) = S y``."""
        self._step_rad(window)
        n = window.n
        if self.length == 1:
            return np.eye(n)
        h = self.half
        n_out = n if window.closed else n - 2 * h
        rows = np.arange(n_out)
        s = np.zeros((n_out, n))
        for j, tap in enumerate(self.taps):                       # one vectorised diagonal per tap
            cols = (rows + j - h) % n if window.closed else rows + j
            s[rows, cols] += tap
        return s

    def smooth(self, tod: np.ndarray, window: LSTWindow) -> np.ndarray:
        """Convolve along axis 0. Trims half a kernel at each end of an open arc."""
        y = np.asarray(tod, dtype=float)
        if y.shape[0] != window.n:
            raise ValueError(f"tod has {y.shape[0]} samples, the window {window.n}.")
        return self.smoothing_operator(window) @ y

    def valid(self, window: LSTWindow) -> LSTWindow:
        """The window of samples that survive :meth:`smooth`."""
        self._step_rad(window)
        if window.closed or self.length == 1:
            return window
        h = self.half
        return LSTWindow(window.lst_deg[h:window.n - h], window.weights[h:window.n - h], window.t_ref_deg)

    def noise_covariance(self, window: LSTWindow, sigma: Union[float, np.ndarray] = 1.0) -> np.ndarray:
        """Covariance of the smoothed samples for independent input noise ``sigma``.

        ``sigma`` is a scalar or per-*input*-sample array -- for a weighted
        window, ``sigma / sqrt(weights)``. Smoothing correlates neighbours
        over one kernel length.
        """
        sig = np.broadcast_to(np.asarray(sigma, dtype=float), (window.n,))
        s = self.smoothing_operator(window)
        return (s * sig ** 2) @ s.T

    def _step_rad(self, window: LSTWindow) -> float:
        if self.length == 1:
            return 0.0
        c = window.cadence_deg
        if c is None:
            raise ValueError("Smoothing needs a uniformly sampled window in time order; this one "
                             "is not. Bin or resample first, or solve without a kernel.")
        need = self.length if window.closed else 2 * self.half + 2
        if window.n < need:
            raise ValueError(f"A {self.length}-tap kernel needs at least {need} samples; "
                             f"the window has {window.n}.")
        return float(np.deg2rad(c))


def dpss_kernel(length: int, nw: float) -> Kernel:
    """Zeroth-order Slepian sequence: least response outside ``|m| <= NW 2pi/T_k`` for its length."""
    from scipy.signal.windows import dpss

    if length % 2 == 0:
        raise ValueError("Use an odd length.")
    if nw <= 0 or nw >= length / 2:
        raise ValueError("NW must lie in (0, length/2).")
    return Kernel(dpss(length, nw), label=f"dpss(L={length}, NW={nw:g})")


def boxcar_kernel(length: int) -> Kernel:
    return Kernel(np.ones(length), label=f"boxcar(L={length})")


def identity_kernel() -> Kernel:
    """No smoothing: the plain windowed-Fourier solve."""
    return Kernel(np.ones(1), label="none")
