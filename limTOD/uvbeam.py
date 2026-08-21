"""pyuvdata ``UVBeam`` adapters: use measured/simulated beams in limTOD.

Three entry points, all lazy on the optional ``pyuvdata`` dependency
(install with ``pip install "limTOD[uvbeam]"``):

* :func:`uvbeam_to_healpix_maps` — sample a UVBeam onto a HEALPix grid in
  limTOD's beam-map convention (boresight at the pole, RING ordering),
  as Stokes I or full (4, npix) pseudo-Stokes [I, Q, U, V] rows.
* :func:`uvbeam_beam_func` — wrap a UVBeam as a ``beam_func(freq=...,
  nside=...)`` callable satisfying :class:`limTOD.TODSim`'s contract
  (chromatic: each frequency interpolates the UVBeam).
* :func:`uvbeam_to_patch_beam` — sample a UVBeam onto the (l, m)
  direction-cosine grid of :class:`limTOD.patchbeam.beam.MeerKLASSBeam`,
  bridging measured beams into the disc-restricted
  :mod:`limTOD.patchbeam` path.

UVBeam conventions
------------------

* UVBeam zenith angle maps directly to the HEALPix polar angle
  (``za = theta``: boresight at the pole).
* UVBeam azimuth follows pyuvdata's convention: a FIXED antenna-local
  frame (x = East, y = North, z = boresight) with az measured from East
  toward North ("az runs from East to North") — pyuvdata does not define
  how that frame rotates with pointing. limTOD's beam-map phi is instead
  anchored to the pointing: phi = 0 is carried to the direction of
  increasing elevation (e_el) and phi = pi/2 to the direction of
  increasing azimuth (e_az); see docs/theory.md ("Beam coordinate
  convention"). The adapter identifies the two via
  ``az_uvbeam = pi/2 - phi_healpix`` — UVBeam's North axis lands on the
  phi = 0 meridian (-> e_el), its East axis (the X feed for
  x_orientation="east") on phi = pi/2 (-> e_az); the sign flip reflects
  the two azimuths increasing in opposite senses. This mapping is LOCKED
  NUMERICALLY by the three-way orientation test in ``tests/test_uvbeam.py``
  (HEALPix path vs the patch-beam (l, m) disc path, discrimination via a
  strongly displaced beam: winner 0.5% agreement, all other candidate
  mappings 66-90% off). Conventions are never trusted on paper here —
  the hand derivation of this mapping had a handedness error that only
  the numerical lock caught.
* Direction cosines for the patch bridge: ``l = sin(za) cos(az_uvbeam)``
  (UVBeam East -> patch l axis, along e_az), ``m = sin(za)
  sin(az_uvbeam)`` (UVBeam North -> patch m axis, along e_el) — the SIN
  convention of :mod:`limTOD.patchbeam.projection`.
* Pixels beyond the UVBeam's zenith-angle coverage are filled with
  ``fill_value`` (default 0 — no response outside the measured domain).

Efield beams are converted with pyuvdata's own machinery
(``efield_to_pstokes`` for Stokes maps, ``efield_to_power`` for the
patch bridge), so no polarization algebra is re-implemented here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Union

import healpy as hp
import numpy as np

# Pseudo-Stokes polarization numbers in pyuvdata (pI, pQ, pU, pV).
_PSTOKES_NUM = {"I": 1, "Q": 2, "U": 3, "V": 4}
# AIPS auto-correlation polarization numbers.
_POL_XX, _POL_YY = -5, -6
# patch-beam polarization label -> UVBeam power polarization.
_PATCH_POL = {"HH": _POL_XX, "VV": _POL_YY}


def _require_pyuvdata() -> Any:
    try:
        import pyuvdata
    except ImportError as exc:  # pragma: no cover — depends on install extras
        raise ImportError(
            "limTOD.uvbeam needs the optional pyuvdata package; install it "
            'with: pip install "limTOD[uvbeam]" (or pip install pyuvdata).'
        ) from exc
    return pyuvdata


def healpix_phi_to_uvbeam_az(phi_rad: np.ndarray) -> np.ndarray:
    """HEALPix beam-map azimuth -> UVBeam azimuth [radians].

    ``az_uvbeam = pi/2 - phi_healpix``: UVBeam's North axis (az = pi/2)
    maps to the beam-map meridian phi = 0 (carried to the direction of
    increasing elevation at the pointing), its East axis (az = 0) to
    phi = pi/2 (increasing azimuth). Locked by the three-way orientation
    test in ``tests/test_uvbeam.py``; see the module docstring and
    docs/theory.md.
    """
    return (0.5 * np.pi - np.asarray(phi_rad, dtype=np.float64)) % (2.0 * np.pi)


def _validate_az_za(uvb: Any) -> None:
    if uvb.pixel_coordinate_system != "az_za":
        raise NotImplementedError(
            "limTOD.uvbeam supports UVBeam objects on regular (az, za) grids; "
            f"got pixel_coordinate_system={uvb.pixel_coordinate_system!r}. "
            "For HEALPix-pixelized UVBeams, interpolate to an az_za grid "
            "first (see pyuvdata's UVBeam.interp with az_za_grid=True)."
        )


def _validate_domain(uvb: Any) -> None:
    """The zero-fill contract assumes the grid starts at za=0 and covers the
    full azimuth circle; anything else would silently reach pyuvdata's own
    domain error instead of the documented fill behaviour."""
    az = np.asarray(uvb.axis1_array, dtype=np.float64)
    za = np.asarray(uvb.axis2_array, dtype=np.float64)
    if za[0] > 1e-6:
        raise ValueError(
            f"UVBeam zenith-angle grid starts at {za[0]:.4f} rad, not 0; "
            "limTOD.uvbeam assumes boresight coverage."
        )
    span = float(az.max() - az.min())
    step = float(np.median(np.diff(az))) if az.size > 1 else 0.0
    if span < 2.0 * np.pi - 2.0 * step - 1e-9:
        raise ValueError(
            f"UVBeam azimuth grid spans only {span:.4f} rad; limTOD.uvbeam "
            "assumes full 2*pi azimuth coverage."
        )


def _stokes_tuple(stokes: str) -> Sequence[str]:
    if stokes == "I":
        return ("I",)
    if stokes == "IQUV":
        return ("I", "Q", "U", "V")
    raise ValueError(f"stokes must be 'I' or 'IQUV', got {stokes!r}")


def _prepare_stokes_beam(uvb: Any, stokes: str, peak_normalize: bool) -> tuple[Any, list[int], bool]:
    """Return ``(power_beam, polarization_numbers, average_copols)``.

    ``average_copols=True`` means the caller must average the returned
    XX/YY polarizations to form Stokes I (power-only beams without
    pseudo-Stokes products); otherwise the polarization numbers select
    pseudo-Stokes entries directly.
    """
    wants = _stokes_tuple(stokes)

    if uvb.beam_type == "efield":
        wb = uvb.copy()
        wb.efield_to_pstokes()
    else:
        wb = uvb

    pols = {int(p) for p in np.asarray(wb.polarization_array)}
    have_pstokes = all(_PSTOKES_NUM[s] in pols for s in wants)

    if have_pstokes:
        pol_nums = [_PSTOKES_NUM[s] for s in wants]
        average = False
    elif stokes == "I" and {_POL_XX, _POL_YY} <= pols:
        pol_nums = [_POL_XX, _POL_YY]
        average = True
    else:
        raise ValueError(
            f"UVBeam (beam_type={uvb.beam_type!r}, polarizations={sorted(pols)}) "
            f"cannot provide stokes={stokes!r}: full Stokes needs an efield "
            "beam or a power beam carrying pseudo-Stokes products; Stokes I "
            "additionally accepts a power beam with both XX and YY."
        )

    if peak_normalize and wb.data_normalization != "peak":
        if wb is uvb:
            wb = uvb.copy()
        wb.peak_normalize()
    return wb, pol_nums, average


def _interp_beam(
    wb: Any,
    az_rad: np.ndarray,
    za_rad: np.ndarray,
    freq_hz: np.ndarray,
    pol_nums: Sequence[int],
    freq_interp_kind: str,
    interp_kwargs: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Interpolate a power beam; returns real values, shape (npol, nfreq, npts)."""
    from pyuvdata import utils as _uvutils

    if int(wb.Naxes_vec) != 1:
        raise ValueError(
            f"Expected a power beam with Naxes_vec=1, got {wb.Naxes_vec}; "
            "vector-valued power beams are not supported."
        )
    kwargs: Dict[str, Any] = {} if interp_kwargs is None else dict(interp_kwargs)
    # UVBeam.interp's `polarizations` selector takes polarization STRINGS
    # ('xx', 'pI', ...), not AIPS numbers.
    pol_strs = [_uvutils.polnum2str(p) for p in pol_nums]
    vals = wb.interp(
        az_array=az_rad,
        za_array=za_rad,
        freq_array=freq_hz,
        freq_interp_kind=freq_interp_kind,
        polarizations=pol_strs,
        return_basis_vector=False,  # power beams need no basis rotation
        **kwargs,
    )[0]
    # Power-beam interp returns (Naxes_vec=1, Npols, Nfreq, Npts) in a
    # complex container; auto/pseudo-Stokes powers are real up to roundoff.
    return np.real(vals[0])


def uvbeam_to_healpix_maps(
    uvb: Any,
    *,
    freq_MHz: float,
    nside: int,
    stokes: str = "I",
    peak_normalize: bool = False,
    freq_interp_kind: str = "linear",
    fill_value: float = 0.0,
    interp_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Sample a UVBeam onto a HEALPix map in limTOD's beam convention.

    Parameters
    ----------
    uvb : pyuvdata.UVBeam
        Efield or power beam on a regular (az, za) grid covering the full
        azimuth range.
    freq_MHz : float
        Output frequency in MHz (interpolated on the UVBeam frequency
        axis with ``freq_interp_kind``).
    nside : int
        HEALPix resolution of the output map (RING ordering).
    stokes : {'I', 'IQUV'}
        ``'I'`` returns a ``(npix,)`` map; ``'IQUV'`` returns
        ``(4, npix)`` pseudo-Stokes rows in limTOD's multi-row layout.
        Full Stokes requires an efield beam (or a power beam already
        carrying pseudo-Stokes products).
    peak_normalize : bool
        Peak-normalize the beam (per frequency, via pyuvdata) before
        sampling.
    fill_value : float
        Value for pixels beyond the UVBeam zenith-angle coverage.
    interp_kwargs : dict, optional
        Extra keyword arguments forwarded to ``UVBeam.interp`` (e.g.
        ``spline_opts``).

    Returns
    -------
    ndarray
        ``(npix,)`` for Stokes I, ``(4, npix)`` for IQUV; float64.
    """
    _require_pyuvdata()
    _validate_az_za(uvb)
    _validate_domain(uvb)
    wb, pol_nums, average = _prepare_stokes_beam(uvb, stokes, peak_normalize)
    return _sample_healpix(
        wb, pol_nums, average, stokes,
        freq_MHz=freq_MHz, nside=nside,
        freq_interp_kind=freq_interp_kind,
        fill_value=fill_value, interp_kwargs=interp_kwargs,
    )


def _sample_healpix(
    wb: Any,
    pol_nums: Sequence[int],
    average: bool,
    stokes: str,
    *,
    freq_MHz: float,
    nside: int,
    freq_interp_kind: str,
    fill_value: float,
    interp_kwargs: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Sampling kernel over an already-prepared power beam."""
    wants = _stokes_tuple(stokes)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    az_uv = healpix_phi_to_uvbeam_az(phi)

    za_max = float(np.max(wb.axis2_array))
    inside = theta <= za_max + 1e-12

    out = np.full((len(wants), npix), float(fill_value), dtype=np.float64)
    if np.any(inside):
        vals = _interp_beam(
            wb,
            az_uv[inside],
            theta[inside],
            np.atleast_1d(float(freq_MHz)) * 1e6,
            pol_nums,
            freq_interp_kind,
            interp_kwargs,
        )  # (npol, 1, n_inside)
        if average:
            stokes_vals = vals.mean(axis=0, keepdims=True)  # I = (XX + YY) / 2
        else:
            stokes_vals = vals
        out[:, inside] = stokes_vals[:, 0, :]

    if stokes == "I":
        return out[0]
    return out


def uvbeam_beam_func(
    uvb: Any,
    *,
    stokes: str = "I",
    peak_normalize: bool = False,
    freq_interp_kind: str = "linear",
    fill_value: float = 0.0,
    interp_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[..., np.ndarray]:
    """Wrap a UVBeam as a limTOD ``beam_func``.

    The returned callable satisfies :class:`limTOD.TODSim`'s contract
    ``beam_func(freq=..., nside=...) -> ndarray`` (``(npix,)`` for
    Stokes I, ``(4, npix)`` for IQUV) and can be passed directly::

        sim = TODSim(beam_func=uvbeam_beam_func(uvb), sky_func=..., ...)

    Configuration errors (wrong grid type, unsupported Stokes request)
    surface at construction time rather than at the first call — and the
    (possibly expensive) efield conversion / peak normalization runs ONCE
    here, not once per frequency channel of the simulation.
    """
    _require_pyuvdata()
    _validate_az_za(uvb)
    _validate_domain(uvb)
    wb, pol_nums, average = _prepare_stokes_beam(uvb, stokes, peak_normalize)

    def beam_func(*, freq: float, nside: int) -> np.ndarray:
        return _sample_healpix(
            wb, pol_nums, average, stokes,
            freq_MHz=freq,
            nside=nside,
            freq_interp_kind=freq_interp_kind,
            fill_value=fill_value,
            interp_kwargs=interp_kwargs,
        )

    return beam_func


def uvbeam_to_patch_beam(
    uvb: Any,
    *,
    margin_deg: np.ndarray,
    freq_MHz: Optional[np.ndarray] = None,
    polarization: str = "HH",
    peak_normalize: bool = False,
    freq_interp_kind: str = "linear",
    fill_value: float = 0.0,
    interp_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Sample a UVBeam onto a (l, m) grid as a patch beam.

    Bridges measured/simulated UVBeams into the disc-restricted
    :mod:`limTOD.patchbeam` path: the returned
    :class:`limTOD.patchbeam.beam.MeerKLASSBeam` plugs straight into
    :class:`limTOD.patchbeam.PatchBeamTODSim`.

    Parameters
    ----------
    uvb : pyuvdata.UVBeam
        Efield or power beam on a regular (az, za) grid. Efield beams are
        converted with pyuvdata's ``efield_to_power``.
    margin_deg : ndarray
        Uniform 1D grid for both the l and m axes, in degrees of
        direction cosine (``l_deg = rad2deg(l)``), matching the MeerKLASS
        holography format.
    freq_MHz : ndarray, optional
        Output frequency grid in MHz. Default: the UVBeam's own frequency
        grid.
    polarization : {'HH', 'VV'}
        Patch-beam polarization label. Mapped to the UVBeam XX / YY power
        products respectively (pyuvdata's default ``x_orientation:
        'east'`` makes X the horizontal feed).
    fill_value : float
        Value for grid points beyond the UVBeam zenith-angle coverage.

    Returns
    -------
    limTOD.patchbeam.beam.MeerKLASSBeam
    """
    _require_pyuvdata()
    from limTOD.patchbeam.beam import MeerKLASSBeam

    _validate_az_za(uvb)
    _validate_domain(uvb)
    pol = polarization.upper()
    if pol not in _PATCH_POL:
        raise ValueError(f"polarization must be one of {tuple(_PATCH_POL)}, got {polarization!r}")

    if uvb.beam_type == "efield":
        wb = uvb.copy()
        wb.efield_to_power(calc_cross_pols=False)
    else:
        wb = uvb
    pols = {int(p) for p in np.asarray(wb.polarization_array)}
    if _PATCH_POL[pol] not in pols:
        raise ValueError(
            f"UVBeam power products {sorted(pols)} do not include the "
            f"{pol} (AIPS {_PATCH_POL[pol]}) polarization."
        )
    if peak_normalize and wb.data_normalization != "peak":
        if wb is uvb:
            wb = uvb.copy()
        wb.peak_normalize()

    margin = np.asarray(margin_deg, dtype=np.float64)
    freqs_out = (
        np.asarray(uvb.freq_array, dtype=np.float64) / 1e6
        if freq_MHz is None
        else np.asarray(freq_MHz, dtype=np.float64)
    )

    # (m, l) grids in direction cosines; SIN inverse to (az, za).
    m_dc, l_dc = np.meshgrid(np.deg2rad(margin), np.deg2rad(margin), indexing="ij")
    r = np.hypot(l_dc, m_dc)
    az_uv = np.arctan2(m_dc, l_dc) % (2.0 * np.pi)
    za = np.arcsin(np.clip(r, 0.0, 1.0))

    za_max = float(np.max(wb.axis2_array))
    inside = (za <= za_max + 1e-12) & (r <= 1.0)

    n_m = n_l = margin.size
    cube = np.full((freqs_out.size, n_m, n_l), float(fill_value), dtype=np.float64)
    if np.any(inside):
        vals = _interp_beam(
            wb,
            az_uv[inside],
            za[inside],
            freqs_out * 1e6,
            [_PATCH_POL[pol]],
            freq_interp_kind,
            interp_kwargs,
        )  # (1, n_freq, n_inside)
        cube[:, inside] = vals[0]

    return MeerKLASSBeam.from_arrays(
        freq_MHz=freqs_out,
        margin_deg=margin,
        power={pol: cube.astype(np.float32)},
    )
