"""m-mode solver for single-dish drift scans.

Earth rotation turns the beam-weighted sky integral into a Fourier series in
local sidereal time (Zhang, *M-mode RIME explicit in beam, fringe and sky
modes*, eqns 13-14, with the fringe set to unity)::

    d(nu, t) = sum_m d_m(nu) e^{i m (t - t_ref)},    d_m = sum_{l >= |m|} T_lm B_lm

This package solves time-ordered data for the ``d_m(nu)``:

``window``
    :class:`LSTWindow` -- the sidereal times visited, with weights for
    stacked nights; coverage, Rayleigh width and Shannon number.
``kernel``
    Smoothing along LST (:func:`dpss_kernel`, or any symmetric taps), its
    transfer ``K(m)``, the valid region, and the noise covariance it induces.
``linear``
    The per-channel least-squares solve (:func:`solve`), conditioning, the
    error budget (:func:`noise_amplification`, :func:`leakage_bound`), and
    frequency bases applied after the time solve.
``bayes``
    Exact Gaussian posteriors and samples through bayesmith (optional).

The estimator on smoothed data is weighted OLS with sandwich errors: GLS
with the kernel's own covariance would undo the kernel (see ``kernel``).

What the window allows is decided before any solve: a 7 h arc covering 29% of
the day cannot separate the low modes at all, whatever the kernel or the
frequency basis (see the MERS explorer and ``docs/mmode.md``).
"""

from .window import LSTWindow
from .kernel import Kernel, dpss_kernel, boxcar_kernel, identity_kernel
from .linear import (
    LinearResult, solve, design_matrix, conditioning, synthesise, pack, unpack,
    noise_amplification, leakage_bound, project_onto_basis, legendre_basis, spline_basis,
)

__all__ = [
    "LSTWindow", "Kernel", "dpss_kernel", "boxcar_kernel", "identity_kernel",
    "LinearResult", "solve", "design_matrix", "conditioning", "synthesise", "pack", "unpack",
    "noise_amplification", "leakage_bound", "project_onto_basis", "legendre_basis", "spline_basis",
]
