"""TRIS beam models in the limTOD beam convention.

Two models are provided, and the choice matters far more than it looks:

* :func:`tris_cut_beam_map` builds the beam from the archive's own E- and
  H-plane cuts, which run all the way to 176 degrees, with nulls to -60 dB and about
  -48 dB at the anti-boresight.  This
  is the model to use for any quantitative sky comparison or map-making.
* :func:`approximate_tris_gaussian_beam_map` is a main-lobe-only elliptical
  Gaussian.  It is cheap and smooth, but it under-predicts the response
  outside roughly 20 degrees by up to an order of magnitude and puts
  essentially zero power below the horizon.  Forward-modelling a realistic
  600-MHz sky with it, instead of the cut-based beam, moves the predicted
  ring by about 0.9 K rms -- roughly 90 times the published statistical error
  and 14 times the published zero-level systematic.  See ``docs/tris.md``.

Orientation (verified numerically against ``limTOD.simulator``, not merely
asserted): the boresight is the map's north pole, the intrinsic **E** plane is
``phi = 0/180`` and the intrinsic **H** plane is ``phi = 90/270``.  Under
:func:`limTOD.tris.tris_zenith_geometry` the E plane lands along the meridian
and is rolled 7 degrees east of it, matching the archive's own note.  The roll
belongs to the geometry, never to the beam: do not pre-rotate these maps.
"""

import typing as _typing

import healpy as hp
import numpy as np

from ._validate import (
    _RealScalarInput,
    _validate_finite_scalar,
    _validate_positive_scalar,
)
from .archive import TRISPrincipalPlaneCuts

_BLENDS = ("db", "power")
_NORMALIZATIONS = ("peak", "sum", "none")


class _BeamFunc(_typing.Protocol):
    def __call__(self, *, freq: _RealScalarInput, nside: int) -> np.ndarray: ...


def _validate_normalization(normalization: str) -> str:
    if normalization not in _NORMALIZATIONS:
        raise ValueError('normalization must be "peak", "sum", or "none"')
    return normalization


def _apply_normalization(beam_map: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "peak":
        return beam_map / np.max(beam_map)
    if normalization == "sum":
        return beam_map / np.sum(beam_map)
    return beam_map


def tris_cut_beam_response(
    cuts: TRISPrincipalPlaneCuts,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    *,
    blend: str = "db",
) -> np.ndarray:
    """Evaluate the cut-interpolated TRIS beam at ``(theta, phi)`` in degrees.

    ``theta`` is the angle from boresight and ``phi`` the intrinsic azimuth,
    with the E plane at ``phi = 0/180`` and the H plane at ``phi = 90/270``.
    The two principal-plane cuts are interpolated in ``theta`` and blended in
    ``phi`` with ``cos^2 phi`` / ``sin^2 phi`` weights.

    ``blend="db"`` (default) interpolates in decibels.  That is the standard
    horn-pattern construction and it is the exact generalization of an
    elliptical Gaussian: if both cuts were Gaussian, this reproduces
    :func:`approximate_tris_gaussian_beam_map` identically.  ``blend="power"``
    blends the linear powers instead.  Both are exact on the principal planes
    and differ only in between; the spread between them is a usable estimate
    of the interpolation error, which no public TRIS product can remove.

    Beyond the last tabulated angle (176 degrees) the cut is held constant at
    its final value; the archive supplies nothing further.
    """
    if not isinstance(cuts, TRISPrincipalPlaneCuts):
        raise TypeError("cuts must be a TRISPrincipalPlaneCuts")
    if blend not in _BLENDS:
        raise ValueError('blend must be "db" or "power"')
    theta = np.asarray(theta_deg, dtype=float)
    phi = np.asarray(phi_deg, dtype=float)
    if theta.shape != phi.shape:
        raise ValueError("theta_deg and phi_deg must have the same shape")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
        raise ValueError("theta_deg and phi_deg must contain only finite values")
    if np.any(theta < 0.0) or np.any(theta > 180.0):
        raise ValueError("theta_deg must be in [0, 180] degrees")

    order = np.argsort(cuts.angle_deg)
    angle = cuts.angle_deg[order]
    e_db = cuts.e_plane_db[order]
    h_db = cuts.h_plane_db[order]

    e_at = np.interp(theta, angle, e_db, left=e_db[0], right=e_db[-1])
    h_at = np.interp(theta, angle, h_db, left=h_db[0], right=h_db[-1])

    phi_rad = np.deg2rad(phi)
    weight_e = np.cos(phi_rad) ** 2
    weight_h = np.sin(phi_rad) ** 2

    if blend == "db":
        return 10.0 ** ((e_at * weight_e + h_at * weight_h) / 10.0)
    return 10.0 ** (e_at / 10.0) * weight_e + 10.0 ** (h_at / 10.0) * weight_h


def tris_cut_beam_map(
    cuts: TRISPrincipalPlaneCuts,
    *,
    nside: int,
    blend: str = "db",
    normalization: str = "peak",
) -> np.ndarray:
    """Return the cut-interpolated TRIS beam as a HEALPix RING map.

    This is the recommended beam for forward modelling and map-making.  Unlike
    the Gaussian it is strictly positive over the whole sphere (the archive
    floor is about -60 dB), so limTOD's default beam truncation never removes
    any of it -- and that means a **horizon treatment becomes mandatory**:
    without one, the roughly 1.2e-4 of beam power below the horizon is fed
    sky brightness instead of ground.  Pass
    :func:`limTOD.tris.tris_horizon_mask` as ``horizontal_mask`` to
    :func:`limTOD.generate_TOD_sky`.
    """
    _validate_normalization(normalization)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
    beam_map = tris_cut_beam_response(
        cuts, np.rad2deg(theta), np.rad2deg(phi), blend=blend
    )
    return _apply_normalization(beam_map, normalization)


def tris_cut_beam_func(
    cuts: TRISPrincipalPlaneCuts,
    *,
    blend: str = "db",
    normalization: str = "peak",
) -> _BeamFunc:
    """Wrap :func:`tris_cut_beam_map` as limTOD's ``beam_func(*, freq, nside)``.

    Achromatic by construction, and that is the archive's own statement, not an
    assumption: ``TRIS_Beam_Profile.txt`` says the profile "is the same at the
    three frequencies ... since the feed horns are scaled versions of a 8 GHz
    prototype".  The frequency is still validated so that a wrong-unit or
    non-finite value fails loudly rather than being silently ignored.
    """
    if not isinstance(cuts, TRISPrincipalPlaneCuts):
        raise TypeError("cuts must be a TRISPrincipalPlaneCuts")
    if blend not in _BLENDS:
        raise ValueError('blend must be "db" or "power"')
    _validate_normalization(normalization)

    def beam_func(*, freq: _RealScalarInput, nside: int) -> np.ndarray:
        _validate_positive_scalar(freq, "freq")
        return tris_cut_beam_map(
            cuts, nside=nside, blend=blend, normalization=normalization
        )

    return beam_func


def approximate_tris_gaussian_beam_map(
    *,
    nside: int,
    fwhm_e_deg: _RealScalarInput = 18.0,
    fwhm_h_deg: _RealScalarInput = 23.0,
    normalization: str = "peak",
) -> np.ndarray:
    """Return an approximate scalar TRIS main-lobe HEALPix RING beam map.

    Its intrinsic E axis is ``phi=0/180`` and its H axis is ``phi=90/270``.
    ``normalization`` may be ``"peak"``, ``"sum"``, or ``"none"``; ``"sum"``
    uses limTOD's discrete HEALPix sum.

    .. warning::

       The default ``18.0`` is the ring headers' rounded prose figure.  The
       archive's own E-plane cut gives ``19.155`` degrees at half power (and
       ``23.366`` for H) -- see
       :meth:`~limTOD.tris.TRISPrincipalPlaneCuts.half_power_full_width_deg`.
       More importantly the Gaussian *shape* is wrong in the shoulders: it
       carries about a tenth of the measured power beyond 30 degrees and
       essentially none below the horizon.  Prefer :func:`tris_cut_beam_map`
       for anything quantitative; this function is for quick looks and for
       reproducing the earlier behaviour.
    """
    normalized_e = _validate_positive_scalar(fwhm_e_deg, "fwhm_e_deg")
    normalized_h = _validate_positive_scalar(fwhm_h_deg, "fwhm_h_deg")
    _validate_normalization(normalization)

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
    sigma_e = np.deg2rad(normalized_e / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_h = np.deg2rad(normalized_h / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    e_offset = theta * np.cos(phi)
    h_offset = theta * np.sin(phi)
    if normalization == "none":
        beam_map = np.exp(
            -0.5 * ((e_offset / sigma_e) ** 2 + (h_offset / sigma_h) ** 2)
        )
        return beam_map

    # Shift the exponent by its maximum before exponentiating so that a very
    # narrow FWHM cannot underflow the whole map to zero before normalization.
    extended = np.longdouble
    sigma_e_extended = np.deg2rad(extended(normalized_e)) / (
        extended(2.0) * np.sqrt(extended(2.0) * np.log(extended(2.0)))
    )
    sigma_h_extended = np.deg2rad(extended(normalized_h)) / (
        extended(2.0) * np.sqrt(extended(2.0) * np.log(extended(2.0)))
    )
    log_response = -extended(0.5) * (
        (e_offset.astype(extended) / sigma_e_extended) ** 2
        + (h_offset.astype(extended) / sigma_h_extended) ** 2
    )
    beam_map = np.asarray(np.exp(log_response - np.max(log_response)), dtype=float)
    if normalization == "sum":
        beam_map /= np.sum(beam_map)
    return beam_map


def tris_beam_func(
    *,
    fwhm_e_deg: _RealScalarInput = 18.0,
    fwhm_h_deg: _RealScalarInput = 23.0,
    normalization: str = "peak",
) -> _BeamFunc:
    """Return an achromatic callable for the approximate scalar Gaussian beam.

    The returned ``beam_func(*, freq, nside)`` follows limTOD's existing
    keyword-only protocol.  It validates a positive finite MHz frequency but
    deliberately does not use it: the public archive states one common beam.

    Every argument is validated here, at the factory boundary, so a typo in
    ``normalization`` fails immediately rather than inside a simulation loop.
    Carries the same accuracy caveat as
    :func:`approximate_tris_gaussian_beam_map`; prefer
    :func:`tris_cut_beam_func`.
    """
    normalized_e = _validate_positive_scalar(fwhm_e_deg, "fwhm_e_deg")
    normalized_h = _validate_positive_scalar(fwhm_h_deg, "fwhm_h_deg")
    _validate_normalization(normalization)

    def beam_func(*, freq: _RealScalarInput, nside: int) -> np.ndarray:
        _validate_positive_scalar(freq, "freq")
        return approximate_tris_gaussian_beam_map(
            nside=nside,
            fwhm_e_deg=normalized_e,
            fwhm_h_deg=normalized_h,
            normalization=normalization,
        )

    return beam_func


def tris_horizon_mask(
    nside: int, *, min_elevation_deg: _RealScalarInput = 0.0
) -> np.ndarray:
    """Return a local-horizontal HEALPix RING mask: 1 above horizon, 0 below.

    The mask is defined in the frame limTOD expects for ``horizontal_mask``:
    pole at the **zenith**, so ``theta`` is the zenith angle and elevation is
    ``90 - theta``.  :func:`limTOD.pointing_beam_in_eq_sys` rotates it into
    equatorial coordinates itself.

    This matters for :func:`tris_cut_beam_map`, which -- unlike the Gaussian --
    has real response below the horizon.  Without a mask those pixels are fed
    sky brightness where the horn actually saw ground.
    """
    elevation_floor = _validate_finite_scalar(min_elevation_deg, "min_elevation_deg")
    if elevation_floor < -90.0 or elevation_floor >= 90.0:
        raise ValueError("min_elevation_deg must be in [-90, 90) degrees")
    theta, _phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=False)
    elevation_deg = 90.0 - np.rad2deg(theta)
    return np.where(elevation_deg >= elevation_floor, 1.0, 0.0)
