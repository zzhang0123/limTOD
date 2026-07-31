"""Core sky -> TOD chain: alm rotation, beam-weighted dot, TOD generation.

Pure-JAX port of the linear part of ``limTOD.simulator.generate_TOD_sky``,
Stokes I or full Stokes. Everything here is jit/vmap/grad-safe: ``lmax``,
``normalize`` and ``npol`` are static; alms, angles, and TODs are traced.

POLARISATION. Pass the static ``npol`` (1, 3 or 4) and give the alms a
leading Stokes axis ``(..., npol, n_alm)`` holding the packed ``T``/``E``/
``B``/``V`` rows — the healpy analysis of limTOD's ``I``/``Q``/``U``/``V``
maps. The axis is CONTRACTED (``V_t = Σ_row ⟨R_t b_row, s̃_row⟩``), which is
numpy limTOD's ``np.sum(beam_map * sky_map)``; ``normalize`` divides every
row by the rotated **Stokes-I** beam's pixel sum, as
``pointing_beam_in_eq_sys`` does. The rotation is the same scalar Wigner
rotation applied per row (bit-identical to healpy's 3-row ``rotate_alm``),
sharing one Risbo recursion. See :mod:`limtod_jax.stokes` for why the whole
chain is row-wise, and ``tests/limtod_jax/test_polarisation.py`` for the
numerical locks.

Exactness contract (the reason this matches numpy limTOD to float64
roundoff): the rotated beam is exactly bandlimited, so limTOD's pixel-space
sample ``np.sum(B_rot * s)`` equals the weighted harmonic dot
``⟨R b, s̃⟩`` EXACTLY when ``s̃`` holds QUADRATURE alms
``(npix/4π)·map2alm(s, iter=0)`` (see :func:`limtod_jax.hpx.map2alm_quad`).
Likewise ``normalize`` divides by the rotated beam's PIXEL SUM, computed
exactly as ``⟨R b, ones_alm⟩`` with ``ones_alm`` the quadrature alms of the
ones map — identical semantics to numpy limTOD's ``normalize_beam``.

Out of scope (nonlinear cleanups, not part of the linear chain):
``_truncate_map`` (``truncate_frac_thres``) and horizontal masks — the
numpy oracle reproduces this module with ``truncate_frac_thres=0.0``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from limtod_jax.alm import alm_dot, packed_from_2d, packed_to_2d
from limtod_jax.stokes import (
    match_npol,
    stokes_i,
    validate_npol,
    validate_pol_alm,
    validate_unpolarised,
)
from limtod_jax.wigner import angles_to_alpha_beta_gamma, rotate_flm_2d


def _full_sphere_dot(a2d: jnp.ndarray, b2d: jnp.ndarray) -> jnp.ndarray:
    """Re Σ_{l,m∈[−l,l]} conj(a)·b on dense flm — equals the weighted packed
    inner product :func:`limtod_jax.alm.alm_dot` for real fields."""
    return jnp.real(jnp.sum(jnp.conj(a2d) * b2d, axis=(-2, -1)))


def _stokes_dot(a2d: jnp.ndarray, b2d: jnp.ndarray, npol: int | None) -> jnp.ndarray:
    """:func:`_full_sphere_dot`, additionally summed over the Stokes axis.

    The Stokes rows CONTRACT into one number per sample — total detected
    power, unit Mueller response, no leakage terms — which is exactly numpy
    limTOD's ``np.sum(beam_map * sky_map)`` over a multi-row map.
    """
    dot = _full_sphere_dot(a2d, b2d)
    return dot if npol is None else jnp.sum(dot, axis=-1)


def _validate_angles(zyz_angles: jnp.ndarray) -> None:
    if zyz_angles.ndim != 2 or zyz_angles.shape[-1] != 3:
        raise ValueError(
            f"zyz_angles must have shape (n_time, 3) [(psi, theta, phi) rows, "
            f"radians]; got {zyz_angles.shape}"
        )


def _require_ones(normalize: bool, ones_alm) -> None:
    if normalize and ones_alm is None:
        raise ValueError(
            "normalize=True requires ones_alm — the quadrature alms of the "
            "ones map ((npix/4π)·map2alm(1, iter=0); see "
            "limtod_jax.hpx.ones_quadrature_alm) — so that the normalizer is "
            "the rotated beam's exact pixel sum, as in numpy limTOD."
        )
    if ones_alm is not None:
        validate_unpolarised("ones_alm", ones_alm)


def _require_single_beam(name: str, alm: jnp.ndarray, npol: int | None) -> None:
    """This chain rotates ONE beam; batch anything else with ``jax.vmap``.

    Stated here rather than left to :func:`limtod_jax.wigner.rotate_flm_2d`,
    which now accepts one leading axis for the Stokes stack and would
    therefore quietly treat a frequency stack as a batch — returning
    ``(n_time, n_freq)`` where the docstring promises ``(n_time,)``.
    """
    want = 1 if npol is None else 2
    if alm.ndim != want:
        shape = "(n_alm,)" if npol is None else f"({npol}, n_alm)"
        raise ValueError(
            f"{name} must be {want}-D {shape} for npol={npol}, got shape "
            f"{alm.shape}. Batch frequencies with jax.vmap — a leading axis "
            f"here is a Stokes axis or nothing."
        )


def rotate_alm(
    alm: jnp.ndarray,
    psi: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    *,
    lmax: int,
    dl_array: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Wigner rotation of packed real-field alms; angles traced, radians.

    Reproduces the alm operation of ``limTOD.simulator._rotate_healpix_map``
    for the same ``(psi, theta, phi)`` — which is
    ``hp.rotate_alm(alm, phi, theta, psi)``: limTOD passes its phi into
    healpy's first slot (convention locked numerically, see
    ``tests/limtod_jax/test_rotation_convention.py``).

    Accepts ``(n_alm,)`` or a Stokes stack ``(npol, n_alm)``. The stacked form
    reproduces healpy's 3-row (T,E,B) ``rotate_alm`` BIT FOR BIT: spin-weighted
    alms rotate with the same Wigner-D as scalar ones, so there is no separate
    spin-2 rotation to get wrong (locked in
    ``tests/limtod_jax/test_polarisation.py``). All rows share one Risbo
    recursion.

    Args:
        dl_array: optional precomputed ``(lmax+1, 2lmax+1, 2lmax+1)`` Wigner-d
            plane from :func:`~limtod_jax.wigner.generate_rotate_dls`, in which
            case ``theta`` is unused. The plane depends on the polar angle
            ALONE, so a caller whose ``theta`` is fixed — a drift scan, where
            only ``psi`` advances with LST — can build it once and skip the
            Risbo recursion on every call. Measured at lmax=127: 18.6 ms ->
            2.58 ms, with 58 % less HLO. Passing a plane built for a different
            ``theta`` silently rotates by that other angle; it is the caller's
            job to keep them together.
    """
    flm = packed_to_2d(alm, lmax)
    a, b, g = angles_to_alpha_beta_gamma(psi, theta, phi)
    return packed_from_2d(
        rotate_flm_2d(flm, lmax + 1, a, b, g, dl_array=dl_array), lmax
    )


def beam_weighted_sum(
    beam_alm: jnp.ndarray,
    sky_alm: jnp.ndarray,
    *,
    normalize: bool = False,
    ones_alm: jnp.ndarray | None = None,
    npol: int | None = None,
) -> jnp.ndarray:
    """Harmonic-space beam-weighted sum -> scalar antenna temperature.

    Equals limTOD's pixel-space ``_beam_weighted_sum(B, s)`` exactly when
    ``sky_alm`` (and ``ones_alm``) hold quadrature alms (module docstring),
    for Stokes I and for full Stokes alike.

    Args:
        npol: static 1/3/4 to contract a leading Stokes axis
            ``(..., npol, n_alm)``; ``None`` (default) for the unpolarised
            case, where leading axes stay batch axes. ``ones_alm`` is always
            a single unpolarised row: the normalizer is the **Stokes-I**
            beam's pixel sum, applied to every row.
    """
    _require_ones(normalize, ones_alm)
    npol = validate_npol(npol)
    match_npol("beam_alm", beam_alm, "sky_alm", sky_alm, npol)
    if beam_alm.shape[-1] != sky_alm.shape[-1]:
        raise ValueError(
            f"alm length mismatch: {beam_alm.shape[-1]} vs {sky_alm.shape[-1]}"
        )
    num = alm_dot(beam_alm, sky_alm)
    if npol is not None:
        num = jnp.sum(num, axis=-1)
    if not normalize:
        return num
    assert ones_alm is not None  # _require_ones guarantees; narrows the type
    return num / alm_dot(stokes_i(beam_alm, npol), ones_alm)


def generate_tod_sky(
    beam_alm: jnp.ndarray,
    sky_alm: jnp.ndarray,
    zyz_angles: jnp.ndarray,
    *,
    lmax: int,
    normalize: bool = False,
    ones_alm: jnp.ndarray | None = None,
    npol: int | None = None,
) -> jnp.ndarray:
    """Sky TOD: rotate the beam to each pointing, dot with the sky.

    Args:
        beam_alm: packed beam alms ``(n_alm,)`` — or ``(npol, n_alm)`` when
            ``npol`` is set (as ``hp.map2alm(beam_map)`` computes them in
            numpy limTOD).
        sky_alm: packed QUADRATURE sky alms, same shape (module docstring).
        zyz_angles: ``(n_time, 3)`` rows of ``(psi, theta, phi)`` [radians],
            i.e. stacked :func:`limtod_jax.angles.zyz_of_pointing` output.
        lmax: static band-limit matching the alm lengths.
        normalize: static; divide each sample by the rotated beam's pixel
            sum (numpy limTOD's ``normalize_beam`` semantics) — the
            **Stokes-I** row's sum when polarised, applied to every row.
        ones_alm: quadrature alms of the ones map; required iff normalize.
            Always a single unpolarised row ``(n_alm,)``.
        npol: static 1/3/4 — the Stokes axis to CONTRACT (see
            :mod:`limtod_jax.stokes`); ``None`` (default) is the unpolarised
            path, unchanged bit for bit.

    Returns
    -------
        ``(n_time,)`` real TOD for either case — the Stokes rows are summed
        into each sample, not returned separately. Pointings are iterated
        with ``lax.map`` (sequential — per-step Wigner memory is O(lmax^3),
        so batching the time axis would multiply that by n_time); the whole
        function is vmappable over a leading frequency axis of
        ``beam_alm``/``sky_alm``.
    """
    _validate_angles(zyz_angles)
    _require_ones(normalize, ones_alm)
    npol = validate_npol(npol)
    validate_pol_alm("beam_alm", beam_alm, lmax, npol)
    validate_pol_alm("sky_alm", sky_alm, lmax, npol)
    match_npol("beam_alm", beam_alm, "sky_alm", sky_alm, npol)
    _require_single_beam("beam_alm", beam_alm, npol)
    _require_single_beam("sky_alm", sky_alm, npol)
    L = lmax + 1
    beam_flm = packed_to_2d(beam_alm, lmax)
    sky_flm = packed_to_2d(sky_alm, lmax)
    ones_flm = None
    if normalize:
        assert ones_alm is not None  # _require_ones guarantees; narrows the type
        ones_flm = packed_to_2d(ones_alm, lmax)

    def sample(angles: jnp.ndarray) -> jnp.ndarray:
        a, b, g = angles_to_alpha_beta_gamma(angles[0], angles[1], angles[2])
        rot = rotate_flm_2d(beam_flm, L, a, b, g)
        num = _stokes_dot(rot, sky_flm, npol)
        if not normalize:
            return num
        assert ones_flm is not None  # set above whenever normalize is True
        rot_i = rot[0] if npol is not None else rot
        return num / _full_sphere_dot(rot_i, ones_flm)

    return jax.lax.map(sample, zyz_angles)


def generate_tod_sky_adjoint(
    tod: jnp.ndarray,
    beam_alm: jnp.ndarray,
    zyz_angles: jnp.ndarray,
    *,
    lmax: int,
    normalize: bool = False,
    ones_alm: jnp.ndarray | None = None,
    npol: int | None = None,
) -> jnp.ndarray:
    """Exact transpose of :func:`generate_tod_sky` in the sky slot.

    With the weighted real inner product ``⟨·,·⟩`` of
    :func:`limtod_jax.alm.alm_dot` on the sky side and the plain dot on the
    TOD side, the forward map ``s̃ ↦ (Σ_row ⟨R_t b_row, s̃_row⟩)_t`` (optionally
    divided by ``d_t = ⟨R_t b_I, ones⟩``) has adjoint

        ``y ↦ Σ_t (y_t / d_t) · (R_t b)``

    — an accumulation of rotated beam alms, no synthesis/analysis involved.
    Satisfies ``⟨forward(x), y⟩_R == ⟨x, adjoint(y)⟩_w`` to roundoff
    (dot-test in the suite); this is what map-making normal equations use.

    Polarised (``npol`` set) the statement is unchanged, with the sky-side
    inner product summed over the Stokes axis as well: a scalar TOD maps back
    to a FULL ``(npol, n_alm)`` sky increment, because every row contributed
    to every sample.
    """
    _validate_angles(zyz_angles)
    _require_ones(normalize, ones_alm)
    npol = validate_npol(npol)
    validate_pol_alm("beam_alm", beam_alm, lmax, npol)
    _require_single_beam("beam_alm", beam_alm, npol)
    if tod.ndim != 1 or tod.shape[0] != zyz_angles.shape[0]:
        raise ValueError(
            f"tod must be 1D of length n_time={zyz_angles.shape[0]} "
            f"(batch with jax.vmap), got shape {tod.shape}"
        )
    L = lmax + 1
    beam_flm = packed_to_2d(beam_alm, lmax)
    ones_flm = None
    if normalize:
        assert ones_alm is not None  # _require_ones guarantees; narrows the type
        ones_flm = packed_to_2d(ones_alm, lmax)
    acc_dtype = jnp.result_type(beam_flm.dtype, tod.dtype)

    def step(accum, inputs):
        angles, y = inputs
        a, b, g = angles_to_alpha_beta_gamma(angles[0], angles[1], angles[2])
        rot = rotate_flm_2d(beam_flm, L, a, b, g)
        if normalize:
            y = y / _full_sphere_dot(rot[0] if npol is not None else rot, ones_flm)
        return accum + y * rot, None

    accum0 = jnp.zeros(beam_flm.shape, dtype=acc_dtype)
    accum, _ = jax.lax.scan(step, accum0, (zyz_angles, tod))
    return packed_from_2d(accum, lmax)
