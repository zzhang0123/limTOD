"""Exact, unconditionally stable noise model for a TRIS ring.

The TRIS covariance has the form

.. math:: C = \\mathrm{diag}(\\sigma^2) + \\sigma_c^2\\, \\mathbf{1}\\mathbf{1}^T

-- independent per-sample statistical errors plus one fully correlated
zero-level term.  Forming that matrix densely and factorizing it is a trap:
for a 120-sample ring with 0.004 K statistical errors, ``numpy.linalg.cholesky``
keeps succeeding well past the point where ``C`` has lost positive definiteness
in float64, so a caller who marginalizes the zero level with a large
``sigma_c`` gets silently wrong coefficients *and* a wrong covariance, with a
healthy-looking condition number.  At ``sigma_c = 1e5`` the coefficients were
off by 193 sigma with no exception raised.

This module never builds ``C``.  It applies the rank-1 term analytically:
with :math:`u_i = 1/\\sigma_i`, :math:`\\nu = \\sum u_i^2` and
:math:`\\alpha = \\sqrt{1 + \\sigma_c^2 \\nu} - 1`, the exact Cholesky-like
factor is :math:`L = D^{1/2}(I + \\alpha\\, uu^T/\\nu)`, so

.. math:: L^{-1}v = D^{-1/2}v - \\frac{\\alpha}{1+\\alpha}\\,
          \\frac{u\\,(u^T D^{-1/2} v)}{\\nu}.

That is stable for every ``sigma_c``, and in the limit ``sigma_c -> inf`` it
degenerates gracefully into projecting the common mode out entirely, which is
exactly what "marginalize the zero level" should mean.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._validate import (
    _readonly_finite_array,
    _RealScalarInput,
    _validate_optional_nonnegative_scalar,
    _validate_optional_positive_scalar,
    _VectorLike,
)


@dataclass(frozen=True, init=False, eq=False)
class TRISNoiseModel:
    """Diagonal statistical noise plus an optional common zero-level mode.

    Parameters
    ----------
    statistical_uncertainty_k
        Per-sample statistical errors in kelvin.  Must be strictly positive
        after ``uncertainty_floor_k`` is applied; the published rings each
        contain exactly one row with a zero entry, so a floor is not optional
        for real data.
    common_mode_sigma_k
        Standard deviation of one fully correlated offset, or ``None`` for a
        purely diagonal model.  Appropriate for the *symmetric* 600-MHz zero
        level (0.066 K).  The 820-MHz zero level is asymmetric
        (+0.430/-0.300 K) and is never symmetrized for you: choose and record
        an approximation yourself, or handle it outside this API.
    uncertainty_floor_k
        Positive floor applied with ``numpy.maximum`` before anything else.
    """

    statistical_sigma_k: np.ndarray
    common_mode_sigma_k: float

    def __init__(
        self,
        statistical_uncertainty_k: _VectorLike,
        *,
        common_mode_sigma_k: Optional[_RealScalarInput] = None,
        uncertainty_floor_k: Optional[_RealScalarInput] = None,
    ) -> None:
        floor = _validate_optional_positive_scalar(
            uncertainty_floor_k, "uncertainty_floor_k"
        )
        common = _validate_optional_nonnegative_scalar(
            common_mode_sigma_k, "common_mode_sigma_k"
        )
        sigma = _readonly_finite_array(
            statistical_uncertainty_k, "statistical_uncertainty_k"
        )
        if np.any(sigma < 0.0):
            raise ValueError("statistical_uncertainty_k must be non-negative")
        if np.any(sigma == 0.0) and floor is None:
            raise ValueError(
                "zero statistical uncertainty requires a positive uncertainty_floor_k"
            )
        if floor is not None:
            sigma = np.maximum(sigma, floor)
            sigma.setflags(write=False)
        object.__setattr__(self, "statistical_sigma_k", sigma)
        object.__setattr__(
            self, "common_mode_sigma_k", 0.0 if common is None else common
        )

    @property
    def size(self) -> int:
        """Number of samples the model describes."""
        return int(self.statistical_sigma_k.size)

    @property
    def variance_k2(self) -> np.ndarray:
        """Per-sample statistical variance, ignoring the common mode.

        This is what a diagonal-only consumer such as
        :func:`limTOD.wiener_filter_map` can accept.  If a common mode is
        present, do not silently drop it -- carry it as an explicit nuisance
        parameter instead (see
        :func:`limTOD.tris.build_tris_mapmaking_inputs`).
        """
        return self.statistical_sigma_k**2

    def _common_mode_suppression(self) -> float:
        """Return ``1/(1+alpha) = 1/sqrt(1 + sigma_c^2 nu)``.

        This is the factor by which whitening shrinks the common-mode
        direction; it tends to zero as ``sigma_c`` grows, which is the
        projection limit.  It is returned directly rather than as
        ``1 - alpha/(1+alpha)`` because that subtraction cancels
        catastrophically once ``alpha`` is large -- at ``sigma_c = 1e12`` it
        costs several per cent of the (tiny) surviving component.
        """
        if self.common_mode_sigma_k == 0.0:
            return 1.0
        nu = float(np.sum(self.statistical_sigma_k**-2.0))
        return float(1.0 / np.sqrt(1.0 + self.common_mode_sigma_k**2 * nu))

    def whiten(self, values: np.ndarray) -> np.ndarray:
        """Apply ``L**-1`` for some factor with ``C = L L^T``.

        Accepts a vector or a matrix.  Whitening a design matrix and its data
        with this turns the generalized least-squares problem into an ordinary
        one, exactly and stably.

        The factor used here is the *symmetric* one, not LAPACK's
        lower-triangular Cholesky.  They differ by an orthogonal transform,
        which no downstream quantity -- solution, covariance, chi-square --
        can distinguish.

        Implemented as "keep the component orthogonal to the common mode,
        shrink the parallel one", which makes the large-``sigma_c`` limit a
        clean projection instead of a difference of nearly equal numbers.
        """
        array = np.asarray(values, dtype=float)
        if array.ndim not in (1, 2) or array.shape[0] != self.size:
            raise ValueError(
                "values must have {} rows to be whitened by this noise model".format(
                    self.size
                )
            )
        sigma = self.statistical_sigma_k
        scaled = array / (sigma if array.ndim == 1 else sigma[:, None])
        if self.common_mode_sigma_k == 0.0:
            return scaled
        unit = 1.0 / sigma
        nu = float(unit @ unit)
        projection = unit @ scaled
        parallel = (
            np.outer(unit, projection) if array.ndim == 2 else unit * projection
        ) / nu
        return (scaled - parallel) + self._common_mode_suppression() * parallel

    def inverse_apply(self, values: np.ndarray) -> np.ndarray:
        """Apply ``C**-1`` to a vector or matrix without ever forming ``C``."""
        array = np.asarray(values, dtype=float)
        if array.ndim not in (1, 2) or array.shape[0] != self.size:
            raise ValueError(
                "values must have {} rows for this noise model".format(self.size)
            )
        sigma = self.statistical_sigma_k
        inverse_variance = sigma**-2.0
        scaled = array * (
            inverse_variance if array.ndim == 1 else inverse_variance[:, None]
        )
        if self.common_mode_sigma_k == 0.0:
            return scaled
        nu = float(np.sum(inverse_variance))
        weight = self.common_mode_sigma_k**2 / (1.0 + self.common_mode_sigma_k**2 * nu)
        projection = inverse_variance @ array
        correction = (
            inverse_variance * projection
            if array.ndim == 1
            else np.outer(inverse_variance, projection)
        )
        return scaled - weight * correction

    def dense_covariance(self) -> np.ndarray:
        """Return ``C`` densely.

        Provided for tests and diagnostics only.  Solving with this matrix is
        the failure mode this class exists to avoid -- see the module
        docstring.
        """
        covariance = np.diag(self.variance_k2)
        if self.common_mode_sigma_k != 0.0:
            covariance = covariance + self.common_mode_sigma_k**2 * np.ones(
                (self.size, self.size)
            )
        return covariance
