"""Map-making inputs for the TRIS drift ring.

What this does and does not claim
---------------------------------
A single fixed-declination ring cannot determine a two-dimensional sky on its
own: per temporal Fourier mode it constrains one combination
:math:`V_m = \\sum_l B_{lm}^* S_{lm}`, and the ~23-degree beam suppresses
everything above :math:`m \\sim 8`, so about 15 numbers are measured, not 120.
Reconstruction here is therefore **explicitly prior-driven**: the likelihood
supplies the directions the ring constrains, and the prior supplies the rest.
That is a legitimate MAP estimate, and it is what
:func:`limTOD.wiener_filter_map` computes -- but the posterior is only as good
as the prior in the unconstrained directions, and this module reports enough
diagnostics (per-sample beam coverage, posterior uncertainty) to see where
that is happening.  Nothing here turns one ring into a measured full-sky map.

The zero level
--------------
The published zero-level uncertainty is one fully correlated offset.  It can
be carried either as a rank-1 term in the noise covariance or as an explicit
nuisance parameter with a Gaussian prior -- the two are *exactly* equivalent
(marginalizing a ones-column parameter with prior variance
:math:`\\sigma_c^2` reproduces :math:`C + \\sigma_c^2\\mathbf{1}\\mathbf{1}^T`).
The nuisance-parameter route is the default here because it keeps the noise
covariance diagonal, which is what limTOD's Wiener solver accepts, and because
it is unconditionally stable where the dense rank-1 covariance is not.  It
also hands back the fitted offset, which is a physically interesting number.

The 820-MHz zero level is asymmetric (+0.430/-0.300 K).  Nothing here
symmetrizes it for you; pick and record an approximation, or model it outside.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import healpy as hp
import numpy as np

from ._validate import (
    _readonly_finite_array,
    _RealScalarInput,
    _validate_finite_scalar,
    _validate_optional_nonnegative_scalar,
    _validate_optional_positive_scalar,
    _validate_positive_scalar,
    _VectorLike,
)
from .archive import TRISPrincipalPlaneCuts, TRISRing
from .beam import tris_cut_beam_map, tris_horizon_mask
from .geometry import TRISZenithGeometry, tris_zenith_geometry
from .noise import TRISNoiseModel


def tris_ring_pixels(
    nside: int,
    *,
    dec_deg: _RealScalarInput,
    half_width_deg: _RealScalarInput = 45.0,
) -> np.ndarray:
    """Return RING pixel indices in a declination band around the ring.

    The TRIS ring sweeps every right ascension at one declination, so the
    region it constrains is a band, not a disc.  ``half_width_deg`` should be
    comfortably wider than the beam: 45 degrees keeps essentially all of the
    main lobe and shoulders for the 19x23-degree TRIS beam.  Widen it and
    check ``TRISMapMakingInputs.beam_coverage`` -- pixels outside the band are
    not merely down-weighted, their sky contribution is dropped from the
    model entirely.
    """
    declination = _validate_finite_scalar(dec_deg, "dec_deg")
    half_width = _validate_positive_scalar(half_width_deg, "half_width_deg")
    theta, _phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=False)
    pixel_dec = 90.0 - np.rad2deg(theta)
    selected = np.flatnonzero(np.abs(pixel_dec - declination) <= half_width)
    if selected.size == 0:
        raise ValueError(
            "no pixels fall within {} degrees of declination {}".format(
                half_width, declination
            )
        )
    return selected


@dataclass(frozen=True, init=False, eq=False)
class TRISMapSolution:
    """Posterior of a prior-regularized TRIS reconstruction."""

    pixel_indices: np.ndarray
    nside: int
    sky_k: np.ndarray
    sky_uncertainty_k: np.ndarray
    zero_level_k: Optional[float]
    zero_level_uncertainty_k: Optional[float]
    residual_k: np.ndarray
    chi_square: float
    degrees_of_freedom: int

    def __init__(
        self,
        pixel_indices: np.ndarray,
        nside: int,
        sky_k: _VectorLike,
        sky_uncertainty_k: _VectorLike,
        zero_level_k: Optional[float] = None,
        zero_level_uncertainty_k: Optional[float] = None,
        residual_k: Optional[np.ndarray] = None,
        chi_square: float = 0.0,
        degrees_of_freedom: int = 1,
    ) -> None:
        object.__setattr__(self, "pixel_indices", np.asarray(pixel_indices, dtype=int))
        object.__setattr__(self, "nside", int(nside))
        object.__setattr__(self, "sky_k", _readonly_finite_array(sky_k, "sky_k"))
        object.__setattr__(
            self,
            "sky_uncertainty_k",
            np.asarray(sky_uncertainty_k, dtype=float),
        )
        object.__setattr__(self, "zero_level_k", zero_level_k)
        object.__setattr__(self, "zero_level_uncertainty_k", zero_level_uncertainty_k)
        object.__setattr__(
            self,
            "residual_k",
            np.zeros(0) if residual_k is None else np.asarray(residual_k, dtype=float),
        )
        object.__setattr__(self, "chi_square", float(chi_square))
        object.__setattr__(self, "degrees_of_freedom", max(int(degrees_of_freedom), 1))

    @property
    def reduced_chi_square(self) -> float:
        """Data chi-square per sample under the ring's own noise model.

        This is the first number to look at.  A value far above 1 means the
        prior and the operator together cannot describe the ring: usually a sky
        template that is wrong at a level the 0.01 K statistical errors can
        see, in which case a fitted ``zero_level_k`` is absorbing template
        mismatch rather than measuring the archive's zero point.
        """
        return self.chi_square / self.degrees_of_freedom

    def healpix_map(self, fill: float = hp.UNSEEN) -> np.ndarray:
        """Scatter the solved pixels back onto a full-sky RING map."""
        full = np.full(hp.nside2npix(self.nside), fill, dtype=float)
        full[self.pixel_indices] = self.sky_k
        return full

    def healpix_uncertainty(self, fill: float = hp.UNSEEN) -> np.ndarray:
        """Scatter the posterior standard deviation onto a full-sky RING map."""
        full = np.full(hp.nside2npix(self.nside), fill, dtype=float)
        full[self.pixel_indices] = self.sky_uncertainty_k
        return full


@dataclass(frozen=True, init=False, eq=False)
class TRISMapMakingInputs:
    """Everything a prior-regularized map-maker needs for one TRIS ring.

    ``operator @ parameters`` predicts ``data_k``.  The first
    ``sky_parameter_count`` parameters are sky pixels (in ``pixel_indices``
    order); if ``has_zero_level`` is true, one trailing parameter is the
    common zero-level offset and its operator column is all ones.
    """

    ring: TRISRing
    geometry: TRISZenithGeometry
    nside: int
    pixel_indices: np.ndarray
    beam_map: np.ndarray
    operator: np.ndarray
    data_k: np.ndarray
    noise: TRISNoiseModel
    beam_coverage: np.ndarray
    sky_parameter_count: int
    zero_level_sigma_k: Optional[float]

    def __init__(
        self,
        ring: TRISRing,
        geometry: TRISZenithGeometry,
        nside: int,
        pixel_indices: np.ndarray,
        beam_map: np.ndarray,
        operator: np.ndarray,
        data_k: np.ndarray,
        noise: TRISNoiseModel,
        beam_coverage: np.ndarray,
        sky_parameter_count: int,
        zero_level_sigma_k: Optional[float],
    ) -> None:
        for name, value in (
            ("ring", ring),
            ("geometry", geometry),
            ("noise", noise),
            ("zero_level_sigma_k", zero_level_sigma_k),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "nside", int(nside))
        object.__setattr__(self, "pixel_indices", np.asarray(pixel_indices, dtype=int))
        object.__setattr__(self, "beam_map", beam_map)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "data_k", data_k)
        object.__setattr__(self, "beam_coverage", beam_coverage)
        object.__setattr__(self, "sky_parameter_count", int(sky_parameter_count))

    @property
    def has_zero_level(self) -> bool:
        """Whether a trailing common-offset nuisance parameter is present."""
        return self.operator.shape[1] > self.sky_parameter_count

    @property
    def parameter_count(self) -> int:
        """Total number of fitted parameters, nuisance offset included."""
        return int(self.operator.shape[1])

    @property
    def monopole_degeneracy(self) -> float:
        """How degenerate a free zero level is with the sky monopole, in [0, 1].

        The zero-level column is all ones.  The sky monopole's response is
        ``A @ 1``, which equals :attr:`beam_coverage` -- and that is nearly
        constant, because a normalized beam integrates to the same number at
        every sample of a zenith drift scan.  This property returns the cosine
        between those two directions, and for TRIS it is ``1 - 1e-9`` or
        better.

        The consequence is not subtle: **a free zero level and a free sky
        monopole are not separately measurable from one ring.**  Whatever
        splits them comes from the priors, not from the data.  If you enable
        ``zero_level_sigma_k``, make sure the sky prior's *implied* monopole
        width (roughly the per-pixel width divided by the square root of the
        pixel count) is much tighter than the zero-level prior, or the fit will
        trade one against the other and report a confident, wrong offset.
        """
        response = self.beam_coverage
        ones = np.ones_like(response)
        denominator = np.linalg.norm(response) * np.linalg.norm(ones)
        if denominator == 0.0:
            return 0.0
        return float(abs(response @ ones) / denominator)

    def implied_monopole_prior_sigma_k(
        self, prior_sigma_k: Union[float, np.ndarray]
    ) -> float:
        """Monopole width implied by an independent per-pixel sky prior.

        The quantity the ring is actually sensitive to is the beam-weighted
        sky average ``m = sum_i w_i s_i`` with ``w`` the (normalized) mean
        operator row.  For an independent prior of width ``sigma_i`` per pixel
        that average has width ``sqrt(sum_i w_i^2 sigma_i^2)`` -- roughly
        ``sigma/sqrt(N_eff)``, which for a few thousand pixels is a small
        number even when each pixel is loosely constrained.

        Compare it with ``zero_level_sigma_k`` before trusting a fitted
        offset: if the two are comparable, the split between them is set by
        the priors, not by the data.  See :attr:`monopole_degeneracy`.
        """
        sigma = np.asarray(prior_sigma_k, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(self.sky_parameter_count, float(sigma))
        if sigma.size != self.sky_parameter_count:
            raise ValueError(
                "prior_sigma_k must be a scalar or one value per selected pixel"
            )
        if np.any(sigma <= 0.0):
            raise ValueError("prior_sigma_k must be strictly positive")
        weights = self.operator[:, : self.sky_parameter_count].mean(axis=0)
        total = weights.sum()
        if total <= 0.0:
            raise ValueError("the operator has no positive sky response")
        weights = weights / total
        return float(np.sqrt(np.sum((weights * sigma) ** 2)))

    def with_data(self, data_k: _VectorLike) -> "TRISMapMakingInputs":
        """Return a copy carrying different samples, e.g. a simulated ring.

        Everything expensive (the operator, the beam, the pixel set) is shared,
        so this is the cheap way to run a closure test or a null test against
        the same geometry.
        """
        replacement = _readonly_finite_array(data_k, "data_k")
        if replacement.size != self.data_k.size:
            raise ValueError("data_k must have {} samples".format(self.data_k.size))
        return TRISMapMakingInputs(
            ring=self.ring,
            geometry=self.geometry,
            nside=self.nside,
            pixel_indices=self.pixel_indices,
            beam_map=self.beam_map,
            operator=self.operator,
            data_k=replacement,
            noise=self.noise,
            beam_coverage=self.beam_coverage,
            sky_parameter_count=self.sky_parameter_count,
            zero_level_sigma_k=self.zero_level_sigma_k,
        )

    def predict(self, sky_k: _VectorLike, zero_level_k: float = 0.0) -> np.ndarray:
        """Forward-model a sky vector (and optional offset) into ring samples."""
        sky = _readonly_finite_array(sky_k, "sky_k")
        if sky.size != self.sky_parameter_count:
            raise ValueError(
                "sky_k must have {} entries, one per selected pixel".format(
                    self.sky_parameter_count
                )
            )
        parameters = sky
        if self.has_zero_level:
            parameters = np.concatenate(
                [sky, [_validate_finite_scalar(zero_level_k, "zero_level_k")]]
            )
        return self.operator @ parameters

    def solve(
        self,
        *,
        prior_map: Optional[np.ndarray] = None,
        prior_sigma_k: Optional[Union[float, np.ndarray]] = None,
        regularization: float = 1e-12,
    ) -> TRISMapSolution:
        """Solve the MAP/Wiener problem with :func:`limTOD.wiener_filter_map`.

        ``prior_map`` may be a full-sky HEALPix RING map at this ``nside`` or a
        vector already restricted to ``pixel_indices``; it must be in the TRIS
        temperature convention (Rayleigh-Jeans, CMB monopole included -- see
        :func:`limTOD.tris.to_tris_temperature_convention`).  ``prior_sigma_k``
        is the prior standard deviation, scalar or per selected pixel.

        With no prior this is an ordinary least-squares solve of a
        rank-deficient system and the result is meaningless; a prior is
        required unless you have restricted the pixel set to something the
        ring actually determines.
        """
        from ..HPW_filter import wiener_filter_map

        n_parameters = self.parameter_count
        guess = np.zeros(n_parameters)
        prior_inv_cov: Optional[np.ndarray] = None

        if prior_map is not None:
            values = np.asarray(prior_map, dtype=float)
            if values.size == hp.nside2npix(self.nside):
                values = values[self.pixel_indices]
            if values.size != self.sky_parameter_count:
                raise ValueError(
                    "prior_map must be a full-sky map at nside={} or a vector "
                    "of {} selected pixels".format(self.nside, self.sky_parameter_count)
                )
            guess[: self.sky_parameter_count] = values

        if prior_sigma_k is not None:
            sigma = np.asarray(prior_sigma_k, dtype=float)
            if sigma.ndim == 0:
                sigma = np.full(self.sky_parameter_count, float(sigma))
            if sigma.size != self.sky_parameter_count:
                raise ValueError(
                    "prior_sigma_k must be a scalar or one value per selected pixel"
                )
            if np.any(sigma <= 0.0):
                raise ValueError("prior_sigma_k must be strictly positive")
            prior_inv_cov = np.zeros(n_parameters)
            prior_inv_cov[: self.sky_parameter_count] = sigma**-2.0
        elif self.has_zero_level:
            prior_inv_cov = np.zeros(n_parameters)

        if self.has_zero_level:
            assert prior_inv_cov is not None  # set in both branches above
            zero_sigma = self.zero_level_sigma_k
            if zero_sigma is None or zero_sigma <= 0.0:
                raise ValueError(
                    "a zero-level column needs a positive zero_level_sigma_k prior"
                )
            prior_inv_cov[-1] = zero_sigma**-2.0

        # wiener_filter_map's return type is a union over return_full_cov;
        # with the default it is always the two-element form.
        solved = wiener_filter_map(
            self.data_k,
            self.operator,
            noise_variance=self.noise.variance_k2,
            prior_inv_cov=prior_inv_cov,
            guess=guess,
            regularization=regularization,
        )
        sky_map, uncertainty = solved[0], solved[1]
        zero_level = float(sky_map[-1]) if self.has_zero_level else None
        zero_uncertainty = float(uncertainty[-1]) if self.has_zero_level else None
        residual = self.data_k - self.operator @ sky_map
        whitened = self.noise.whiten(residual)
        return TRISMapSolution(
            pixel_indices=self.pixel_indices,
            nside=self.nside,
            sky_k=sky_map[: self.sky_parameter_count],
            sky_uncertainty_k=uncertainty[: self.sky_parameter_count],
            zero_level_k=zero_level,
            zero_level_uncertainty_k=zero_uncertainty,
            residual_k=residual,
            chi_square=float(whitened @ whitened),
            degrees_of_freedom=self.data_k.size,
        )


def tris_prior_from_template(
    template_map: np.ndarray,
    pixel_indices: np.ndarray,
    *,
    relative_sigma: Optional[_RealScalarInput] = None,
    absolute_sigma_k: Optional[_RealScalarInput] = None,
    floor_sigma_k: _RealScalarInput = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(guess, sigma)`` for a prior taken from an external sky model.

    ``relative_sigma`` expresses "trust this template to within N per cent",
    which is the realistic statement for a Haslam/GSM extrapolation: the prior
    width then tracks the local sky brightness, so the Galactic plane is not
    pinned as tightly as the cold regions.  ``absolute_sigma_k`` adds a flat
    term in quadrature.  ``floor_sigma_k`` keeps the prior from becoming a
    delta function where the template happens to be near zero.

    The template must already be in the TRIS temperature convention.
    """
    template = np.asarray(template_map, dtype=float)
    indices = np.asarray(pixel_indices, dtype=int)
    if template.ndim != 1 or template.size == 0:
        raise ValueError("template_map must be a non-empty one-dimensional map")
    if not np.all(np.isfinite(template)):
        raise ValueError("template_map must contain only finite values")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("pixel_indices must be a non-empty one-dimensional array")
    if indices.min() < 0 or indices.max() >= template.size:
        raise ValueError("pixel_indices fall outside template_map")
    relative = _validate_optional_positive_scalar(relative_sigma, "relative_sigma")
    absolute = _validate_optional_nonnegative_scalar(
        absolute_sigma_k, "absolute_sigma_k"
    )
    floor = _validate_positive_scalar(floor_sigma_k, "floor_sigma_k")
    if relative is None and absolute is None:
        raise ValueError(
            "supply relative_sigma and/or absolute_sigma_k to define a prior width"
        )

    guess = template[indices]
    variance = np.zeros(guess.size)
    if relative is not None:
        variance = variance + (relative * np.abs(guess)) ** 2
    if absolute is not None:
        variance = variance + absolute**2
    sigma = np.sqrt(np.maximum(variance, floor**2))
    return guess, sigma


def build_tris_mapmaking_inputs(
    ring: TRISRing,
    *,
    nside: int,
    cuts: Optional[TRISPrincipalPlaneCuts] = None,
    beam_map: Optional[np.ndarray] = None,
    geometry: Optional[TRISZenithGeometry] = None,
    pixel_indices: Optional[np.ndarray] = None,
    dec_half_width_deg: _RealScalarInput = 45.0,
    uncertainty_floor_k: Optional[_RealScalarInput] = None,
    zero_level_sigma_k: Optional[_RealScalarInput] = None,
    apply_horizon_mask: bool = True,
    nside_hires: Optional[int] = None,
) -> TRISMapMakingInputs:
    """Assemble the operator, data and noise objects for one TRIS ring.

    Supply either ``cuts`` (recommended: the beam is then built from the
    archive's own principal-plane cuts) or an explicit ``beam_map`` in the
    limTOD beam convention.  The geometry defaults to
    :func:`limTOD.tris.tris_zenith_geometry` at the ring's own RA labels, which
    also owns the 7-degree E-plane roll -- do not pre-rotate the beam.

    ``apply_horizon_mask`` is on by default and matters: the cut-based beam has
    genuine response below the horizon (about 1.2e-4 of its power, worth
    ~0.035 K against a 300 K ground, comparable to the published 0.066 K
    zero-level systematic).  Without a mask those pixels are fed sky
    brightness where the horn saw ground.

    ``zero_level_sigma_k`` adds the common offset as a nuisance parameter with
    that prior width -- exactly equivalent to a rank-1 noise term, but stable
    and it reports the fitted offset.  For the 600-MHz ring, 0.066.  The
    820-MHz value is asymmetric and is not converted for you.

    ``uncertainty_floor_k`` is required in practice: each published ring
    contains exactly one row whose statistical uncertainty is 0.000 K.
    """
    from ..simulator import generate_sky2sys_projection

    if not isinstance(ring, TRISRing):
        raise TypeError("build_tris_mapmaking_inputs requires a TRISRing")
    if (beam_map is None) == (cuts is None):
        raise ValueError("supply exactly one of cuts or beam_map")
    if beam_map is None:
        assert cuts is not None  # guaranteed by the exclusive check above
        beam = tris_cut_beam_map(cuts, nside=nside, normalization="peak")
    else:
        beam = np.asarray(beam_map, dtype=float)
        if beam.ndim != 1:
            raise ValueError("beam_map must be a scalar HEALPix map")
        if not np.all(np.isfinite(beam)):
            raise ValueError("beam_map must contain only finite values")

    if geometry is None:
        geometry = tris_zenith_geometry(ring.ra_deg)
    elif not isinstance(geometry, TRISZenithGeometry):
        raise TypeError("geometry must be a TRISZenithGeometry")
    if geometry.lst_deg.size != ring.temperature_k.size:
        raise ValueError("geometry and ring must have the same number of samples")

    if pixel_indices is None:
        pixels = tris_ring_pixels(
            nside,
            dec_deg=geometry.boresight_dec_deg,
            half_width_deg=dec_half_width_deg,
        )
    else:
        pixels = np.asarray(pixel_indices, dtype=int)
        if pixels.ndim != 1 or pixels.size == 0:
            raise ValueError("pixel_indices must be a non-empty one-dimensional array")
        if pixels.min() < 0 or pixels.max() >= hp.nside2npix(nside):
            raise ValueError(
                "pixel_indices fall outside the nside={} map".format(nside)
            )

    horizontal_mask = tris_horizon_mask(nside) if apply_horizon_mask else None

    sky_operator = generate_sky2sys_projection(
        beam,
        geometry.lst_deg,
        geometry.latitude_deg,
        geometry.azimuth_deg,
        geometry.elevation_deg,
        geometry.selfrot_deg,
        pixels,
        horizontal_mask=horizontal_mask,
        normalize_beam=True,
        nside_hires=nside_hires,
        nside_target=nside,
    )
    coverage = sky_operator.sum(axis=1)

    zero_sigma = _validate_optional_positive_scalar(
        zero_level_sigma_k, "zero_level_sigma_k"
    )
    if zero_sigma is None:
        operator = sky_operator
    else:
        operator = np.column_stack([sky_operator, np.ones(sky_operator.shape[0])])

    noise = TRISNoiseModel(
        ring.statistical_uncertainty_k, uncertainty_floor_k=uncertainty_floor_k
    )
    return TRISMapMakingInputs(
        ring=ring,
        geometry=geometry,
        nside=nside,
        pixel_indices=pixels,
        beam_map=beam,
        operator=operator,
        data_k=np.asarray(ring.temperature_k, dtype=float),
        noise=noise,
        beam_coverage=coverage,
        sky_parameter_count=int(sky_operator.shape[1]),
        zero_level_sigma_k=zero_sigma,
    )
