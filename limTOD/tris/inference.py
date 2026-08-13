"""Rank-gated low-dimensional inference on a single TRIS ring.

A fixed-declination drift ring constrains, per temporal Fourier mode, only the
single combination :math:`V_m = \\sum_l B_{lm}^* S_{lm}`.  Many sky harmonics
therefore project onto the same datum, and no amount of regularization inside
*this* module can create the missing information.  What lives here is
deliberately restricted to caller-supplied low-dimensional models: Fourier
representations of the profile, or amplitudes of external templates.

For actual sky reconstruction -- where an informative prior legitimately
supplies the missing directions -- use :mod:`limTOD.tris.mapmaking`, which
builds the operator and noise objects that limTOD's Wiener/MAP map-makers
consume.
"""

from dataclasses import dataclass
import typing as _typing
from typing import Optional

import numpy as np

from ._validate import (
    _coerce_real_scalar,
    _IntegerScalarInput,
    _MatrixLike,
    _readonly_finite_array,
    _readonly_finite_matrix,
    _RealScalarInput,
    _validate_optional_positive_scalar,
    _VectorLike,
)
from .archive import TRISRing
from .noise import TRISNoiseModel


def build_tris_fourier_design(
    ra_deg: _VectorLike, m_max: int, *, include_constant: bool = True
) -> np.ndarray:
    """Build a finite low-dimensional Fourier design at the supplied TRIS RAs.

    With ``include_constant=True``, columns are ordered as ``[1, cos(alpha),
    sin(alpha), ..., cos(m alpha), sin(m alpha)]`` with ``alpha`` in radians.
    The supplied sampling is used verbatim; no uniform-grid approximation is
    applied -- and that matters, because the published RA labels are *not* on
    an exact grid (the real spacings are 2.75, 3.00 and 3.25 degrees).

    Choosing ``m_max``: the beam is about 23.4 degrees wide in the RA
    direction, so it suppresses harmonics above roughly ``180/23.4 ~ 8``.
    Fitting far beyond that fits noise and model mismatch, not sky.  The
    ``p < n`` gate in :func:`fit_tris_linear_model` is a hard backstop, not a
    statement that everything below it is well posed: on the real 120-sample
    ring, ``m_max=59`` passes the gate and reproduces the data to 1e-4 K,
    which is a restatement of the data rather than an inference from it.
    """
    if not isinstance(m_max, (int, np.integer)) or isinstance(m_max, bool) or m_max < 0:
        raise ValueError("m_max must be a non-negative integer")
    if not isinstance(include_constant, (bool, np.bool_)):
        raise ValueError("include_constant must be boolean")
    ra = _readonly_finite_array(ra_deg, "ra_deg")
    if np.any(ra < 0.0) or np.any(ra >= 360.0):
        raise ValueError("ra_deg must be in [0, 360)")

    alpha = np.deg2rad(ra)
    columns: _typing.List[np.ndarray] = []
    if include_constant:
        columns.append(np.ones(ra.size))
    for mode in range(1, int(m_max) + 1):
        columns.extend((np.cos(mode * alpha), np.sin(mode * alpha)))
    if not columns:
        empty = np.empty((ra.size, 0), dtype=float)
        empty.setflags(write=False)
        return empty
    design = np.column_stack(columns)
    design.setflags(write=False)
    return design


@dataclass(frozen=True, init=False, eq=False)
class TRISRankDiagnostic:
    """Immutable SVD identifiability diagnostic for a whitened design."""

    singular_values: np.ndarray
    numerical_rank: int
    parameter_count: int
    tolerance: float
    rank_rtol: float
    condition_number: float

    def __init__(
        self,
        singular_values: _VectorLike,
        numerical_rank: _IntegerScalarInput,
        parameter_count: _IntegerScalarInput,
        tolerance: _RealScalarInput,
        rank_rtol: _RealScalarInput,
        condition_number: _RealScalarInput,
    ) -> None:
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "numerical_rank", numerical_rank)
        object.__setattr__(self, "parameter_count", parameter_count)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "rank_rtol", rank_rtol)
        object.__setattr__(self, "condition_number", condition_number)
        self.__post_init__()

    def __post_init__(self) -> None:
        singular_values = _readonly_finite_array(
            self.singular_values, "singular_values"
        )
        if (
            not isinstance(self.numerical_rank, (int, np.integer))
            or isinstance(self.numerical_rank, bool)
            or self.numerical_rank < 0
            or self.numerical_rank > singular_values.size
        ):
            raise ValueError("numerical_rank must be a valid non-negative integer")
        if (
            not isinstance(self.parameter_count, (int, np.integer))
            or isinstance(self.parameter_count, bool)
            or self.parameter_count <= 0
        ):
            raise ValueError("parameter_count must be a positive integer")
        tolerance = _coerce_real_scalar(self.tolerance, "tolerance")
        rank_rtol = _coerce_real_scalar(self.rank_rtol, "rank_rtol")
        condition_number = _coerce_real_scalar(
            self.condition_number, "condition_number", finite=False
        )
        if tolerance < 0.0 or rank_rtol <= 0.0:
            raise ValueError("tolerance must be non-negative and rank_rtol positive")
        if condition_number < 1.0:
            raise ValueError("condition_number must be at least one")
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "numerical_rank", int(self.numerical_rank))
        object.__setattr__(self, "parameter_count", int(self.parameter_count))
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "rank_rtol", rank_rtol)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True, init=False, eq=False)
class TRISLinearFit:
    """Immutable rank-gated generalized least-squares fit to one TRIS ring."""

    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    prediction_k: np.ndarray
    residual_k: np.ndarray
    rank_diagnostic: TRISRankDiagnostic
    chi_square: float
    degrees_of_freedom: int

    def __init__(
        self,
        coefficients: _VectorLike,
        coefficient_covariance: _MatrixLike,
        prediction_k: _VectorLike,
        residual_k: _VectorLike,
        rank_diagnostic: TRISRankDiagnostic,
        chi_square: _RealScalarInput,
        degrees_of_freedom: _IntegerScalarInput,
    ) -> None:
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "coefficient_covariance", coefficient_covariance)
        object.__setattr__(self, "prediction_k", prediction_k)
        object.__setattr__(self, "residual_k", residual_k)
        object.__setattr__(self, "rank_diagnostic", rank_diagnostic)
        object.__setattr__(self, "chi_square", chi_square)
        object.__setattr__(self, "degrees_of_freedom", degrees_of_freedom)
        self.__post_init__()

    def __post_init__(self) -> None:
        coefficients = _readonly_finite_array(self.coefficients, "coefficients")
        covariance = _readonly_finite_matrix(
            self.coefficient_covariance, "coefficient_covariance"
        )
        if covariance.shape != (coefficients.size, coefficients.size):
            raise ValueError(
                "coefficient_covariance must be square in coefficient count"
            )
        prediction = _readonly_finite_array(self.prediction_k, "prediction_k")
        residual = _readonly_finite_array(self.residual_k, "residual_k")
        if prediction.size != residual.size:
            raise ValueError("prediction_k and residual_k must have the same length")
        if not isinstance(self.rank_diagnostic, TRISRankDiagnostic):
            raise ValueError("rank_diagnostic must be a TRISRankDiagnostic")
        if self.rank_diagnostic.parameter_count != coefficients.size:
            raise ValueError("rank diagnostic and coefficient count must agree")
        chi_square = _coerce_real_scalar(self.chi_square, "chi_square")
        if chi_square < 0.0:
            raise ValueError("chi_square must be non-negative")
        if (
            not isinstance(self.degrees_of_freedom, (int, np.integer))
            or isinstance(self.degrees_of_freedom, bool)
            or self.degrees_of_freedom <= 0
        ):
            raise ValueError("degrees_of_freedom must be a positive integer")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "coefficient_covariance", covariance)
        object.__setattr__(self, "prediction_k", prediction)
        object.__setattr__(self, "residual_k", residual)
        object.__setattr__(self, "chi_square", chi_square)
        object.__setattr__(self, "degrees_of_freedom", int(self.degrees_of_freedom))

    @property
    def reduced_chi_square(self) -> float:
        """``chi^2`` per degree of freedom under the supplied noise model.

        A value far from 1 means the reported ``coefficient_covariance`` is not
        a believable error bar: the model, not the noise, dominates the
        residual.  On the real 600-MHz ring an ``m_max=3`` Fourier model gives
        about 1.1e4 here, i.e. formal errors too small by a factor of ~100.
        """
        return self.chi_square / self.degrees_of_freedom

    @property
    def singular_values(self) -> np.ndarray:
        """Whitened-design singular values from the rank diagnostic."""
        return self.rank_diagnostic.singular_values

    @property
    def numerical_rank(self) -> int:
        """Whitened-design numerical rank."""
        return self.rank_diagnostic.numerical_rank

    @property
    def parameter_count(self) -> int:
        """Number of fitted template coefficients."""
        return self.rank_diagnostic.parameter_count

    @property
    def tolerance(self) -> float:
        """SVD rank tolerance."""
        return self.rank_diagnostic.tolerance

    @property
    def rank_rtol(self) -> float:
        """Relative SVD rank tolerance."""
        return self.rank_diagnostic.rank_rtol

    @property
    def condition_number(self) -> float:
        """Whitened-design condition number."""
        return self.rank_diagnostic.condition_number


def fit_tris_linear_model(
    ring: TRISRing,
    design_matrix: _MatrixLike,
    *,
    uncertainty_floor_k: Optional[_RealScalarInput] = None,
    common_mode_sigma_k: Optional[_RealScalarInput] = None,
    rank_rtol: Optional[_RealScalarInput] = None,
    design_ra_deg: Optional[_VectorLike] = None,
) -> TRISLinearFit:
    """Fit an identifiable caller-supplied template model to one ``TRISRing``.

    The design and data are whitened by :class:`~limTOD.tris.TRISNoiseModel`,
    which applies the optional common-mode term analytically instead of
    factorizing a dense covariance.  The whitened design's SVD rank is checked
    before any coefficient is solved, so this API cannot be turned into a
    free-pixel map-maker.  Archive zero-level metadata is deliberately not read
    or converted here.  An explicit ``rank_rtol`` must be at least machine
    epsilon times the largest whitened-design dimension.

    Pass ``design_ra_deg`` (normally ``ring.ra_deg``) to have the design's
    provenance checked against the ring: only the row *count* is otherwise
    verified, and a design built from a different RA sampling of the same
    length is accepted silently and returns confidently wrong coefficients.
    """
    if not isinstance(ring, TRISRing):
        raise TypeError("fit_tris_linear_model requires a TRISRing")
    design = _readonly_finite_matrix(design_matrix, "design_matrix")
    sample_count, parameter_count = design.shape
    if parameter_count == 0:
        raise ValueError("design_matrix must contain at least one parameter column")
    if sample_count != ring.temperature_k.size:
        raise ValueError("design_matrix row count must match the TRISRing samples")
    if parameter_count >= sample_count:
        raise ValueError(
            "design must be low-dimensional: parameter count={}, sample count={}; "
            "reduce the model before fitting".format(parameter_count, sample_count)
        )
    if design_ra_deg is not None:
        supplied = _readonly_finite_array(design_ra_deg, "design_ra_deg")
        if supplied.size != ring.ra_deg.size or not np.allclose(
            supplied, ring.ra_deg, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "design_ra_deg does not match the ring's own RA sampling; the "
                "design must be built at ring.ra_deg"
            )

    supplied_rtol = _validate_optional_positive_scalar(rank_rtol, "rank_rtol")
    noise = TRISNoiseModel(
        ring.statistical_uncertainty_k,
        common_mode_sigma_k=common_mode_sigma_k,
        uncertainty_floor_k=uncertainty_floor_k,
    )

    whitened_design = noise.whiten(design)
    whitened_data = noise.whiten(ring.temperature_k)
    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        whitened_design, full_matrices=False
    )
    safe_rank_rtol = np.finfo(float).eps * max(whitened_design.shape)
    if supplied_rtol is not None and supplied_rtol < safe_rank_rtol:
        raise ValueError(
            "rank_rtol must be at least {:.17g} for this whitened design".format(
                safe_rank_rtol
            )
        )
    applied_rtol = safe_rank_rtol if supplied_rtol is None else supplied_rtol
    tolerance = applied_rtol * singular_values[0]
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number = (
        float("inf")
        if numerical_rank < parameter_count
        else float(singular_values[0] / singular_values[-1])
    )
    diagnostic = TRISRankDiagnostic(
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        parameter_count=parameter_count,
        tolerance=tolerance,
        rank_rtol=applied_rtol,
        condition_number=condition_number,
    )
    if numerical_rank < parameter_count:
        raise ValueError(
            "rank-deficient whitened design: rank={}, parameter count={}; "
            "reduce the model before fitting".format(numerical_rank, parameter_count)
        )

    coefficients = right_vectors_t.T @ (
        (left_vectors.T @ whitened_data) / singular_values
    )
    coefficient_covariance = (
        right_vectors_t.T / (singular_values**2)
    ) @ right_vectors_t
    prediction = design @ coefficients
    whitened_residual = whitened_data - whitened_design @ coefficients
    return TRISLinearFit(
        coefficients=coefficients,
        coefficient_covariance=coefficient_covariance,
        prediction_k=prediction,
        residual_k=ring.temperature_k - prediction,
        rank_diagnostic=diagnostic,
        chi_square=float(whitened_residual @ whitened_residual),
        degrees_of_freedom=sample_count - parameter_count,
    )
