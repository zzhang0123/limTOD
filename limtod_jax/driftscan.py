"""Drift-scan special case in harmonic space: the m-mode formalism.

For a drift scan — fixed azimuth/elevation/self-rotation at a fixed site,
only LST advancing — the generic chain (one Wigner rotation of the beam per
time sample, O(n_time·lmax³), :func:`limtod_jax.core.generate_tod_sky`)
collapses. Earth rotation is a rotation about the celestial pole, so the
celestial-frame beam alms at any LST differ from those at a reference LST
only by a per-m phase:

    ``B_lm(lst) = B_lm(lst_ref) · exp(−i·m·Δ)``,  ``Δ = lst − lst_ref`` [rad]

(sign locked NUMERICALLY in ``tests/limtod_jax/test_driftscan.py``, never on
paper, per the port contract). The TOD then reduces to a Fourier series in Δ:

    ``V(Δ) = Ṽ_0 + 2·Re Σ_{m≥1} Ṽ_m·exp(+i·m·Δ)``,
    ``Ṽ_m = Σ_l conj(B_lm(lst_ref)) · S̃_lm``

— eqns (13)–(15) of Z. Zhang, *M-mode RIME explicit in beam, fringe and sky
modes* (Jan 2024, https://zh-zhang.com/myNotes/MmodeNote.pdf), in the "MT
interpretation" with the fringe ≡ 1 (single-dish autocorrelation): the
modulated beam IS the primary beam. The ``Ṽ_m`` are the **m-modes** — the
Fourier coefficients of the sidereal-day-periodic TOD — and the projection
sky → m-modes is a single per-(l,m) product with the reference-frame beam.

Cost: ONE Wigner rotation total (O(lmax³)) plus an O(n_time·lmax) phase
synthesis, against O(n_time·lmax³) for the generic path; equality to the
generic path is exact (roundoff — the R_z Wigner-D is exactly a phase).

When the LST grid is UNIFORM over a full sidereal turn, the phase
synthesis is an inverse real FFT and the m-mode analysis a forward one:
O(n_time·log n_time) independent of lmax, measured 19-51x faster than
the direct sum end-to-end (guard included; see below). Opt in with the STATIC
``uniform=True`` / ``uniform_sampling=True`` flags (never auto-detected:
the choice must not depend on traced values). The direct sum remains the
default because real scans have gaps and irregular sampling, where the
FFT identity does not hold.

The beam that enters here is, physically, the beam AFTER a horizon cut
(the antenna cannot see below the ground): :func:`horizon_masked_beam_alm`
applies that cut in the horizontal frame, with optional apodization to
tame the Gibbs ringing of a sharp cut at finite band-limit. It is kept
optional/off by default so the module reproduces numpy limTOD (which does
not mask) exactly; see ``docs/driftscan.md`` for the ringing study.

Conventions (identical to the rest of :mod:`limtod_jax`): degrees in public
pointing APIs, radians internally; healpy packed alm layout (m ≥ 0, real
fields); ``lmax``/``nside``/``normalize`` are static Python ints/bools;
alms, LSTs, and TODs are traced. healpy never enters — it stays the test
oracle. Enable ``jax_enable_x64`` for quantitative work (the masked-beam
path uses the s2fft HEALPix transforms; see :mod:`limtod_jax.hpx`).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from limtod_jax.alm import nalm_of_lmax, packed_lm_arrays
from limtod_jax.angles import zyz_of_pointing, zyzyz2zyz
from limtod_jax.core import rotate_alm
from limtod_jax.hpx import alm2map, map2alm_iter, ones_quadrature_alm


def _validate_alm(name: str, alm: jnp.ndarray, lmax: int) -> None:
    if alm.shape[-1] != nalm_of_lmax(lmax):
        raise ValueError(
            f"{name} length {alm.shape[-1]} does not match lmax={lmax} "
            f"(expected {nalm_of_lmax(lmax)})"
        )


def _validate_dphi(dphi: jnp.ndarray) -> None:
    if dphi.ndim != 1:
        raise ValueError(
            f"dphi must be 1D (n_time,) LST offsets in radians "
            f"(batch with jax.vmap), got shape {dphi.shape}"
        )


def _validate_nyquist(n_time: int, lmax: int) -> None:
    """The FFT fast path needs every m in [1, lmax] off the Nyquist bin.

    ``irfft`` weights bin 0 and (for even n) the Nyquist bin by 1 while the
    m-mode expansion needs weight 2 for every m >= 1, so the grid must
    satisfy ``2·lmax < n_time`` — which is also just the sampling theorem
    for the highest retained mode. Static (shapes only).
    """
    if 2 * lmax >= n_time:
        raise ValueError(
            f"uniform FFT synthesis needs 2*lmax < n_time (sampling theorem; "
            f"m = lmax must stay off the Nyquist bin), got lmax={lmax}, "
            f"n_time={n_time}. Use the default direct sum instead."
        )


def _uniform_tolerance(dtype, scale, xp):
    """Deviation a uniform grid may show, scaled to ``dtype``'s precision.

    ``64·eps(dtype)·max(2π, |Δ|max)`` — it must follow the INPUT dtype, not
    float64: in a float32 session an exactly uniform grid still carries
    ~3e-7 rad of representation error (``deg2rad`` of a degree grid), which
    must not read as non-uniform. Measured headroom over the worst
    legitimate representation error is ~40x (f32) and ~100x (f64).

    Do NOT pass a grid cast UP from a narrower dtype: the cast hides the
    real precision and this bound then rejects a legitimate grid (an f32
    degree grid upcast to f64 deviates ~3e-7, 3e6x the f64 bound). Check at
    the native dtype instead.

    PRECISION CAVEAT, measured: the admitted deviation costs ~lmax·tol rad
    of phase, so in a float32 session the uniform path can carry ~1e-3
    (lmax = 256) to ~1e-2 (lmax = 1024) relative TOD error — the same order
    as that session's own transform error, NOT orders below it. The f32
    branch also cannot detect ppm-level grid errors (its representation
    floor, ~1e-6 rad, exceeds the ~7e-6 rad such an error produces). x64 is
    required for the roundoff contract.

    ``xp`` selects the array namespace and is not cosmetic: ``jnp`` ops on
    concrete scalars still return TRACERS when an outer ``jit`` trace is
    active, so the eager checker must pass ``np`` or its ``float()`` would
    raise ConcretizationTypeError inside somebody's jitted call.
    """
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(
            f"dphi must be a floating-point array to check the uniform-grid "
            f"contract, got dtype {dtype}. (An integer or low-precision "
            f"custom dtype has no meaningful eps, which would silently give "
            f"a zero tolerance.)"
        )
    eps = float(np.finfo(dtype).eps)
    return 64.0 * eps * xp.maximum(2.0 * np.pi, scale)


def check_uniform_grid(dphi: jnp.ndarray) -> None:
    """Raise unless ``dphi`` is a uniform full-turn grid ``Δ_0 + 2πt/n``.

    The eager half of the ``uniform=True`` contract: a clear ValueError
    while the values are still CONCRETE. Under ``jit`` they are traced and
    unavailable — note that this includes the case where the grid is a
    compile-time constant, since any arithmetic inside a trace (e.g.
    ``jnp.deg2rad(lst - ref)``) yields a tracer. Enforcement there is
    :func:`_poison_if_nonuniform`, which turns a violated contract into
    NaN rather than a plausible-looking wrong TOD.

    Public so that downstream adapters (which receive their LST grid per
    call rather than at construction) can validate at their own boundary,
    where the raw array is typically still concrete.
    """
    try:
        raw = np.asarray(dphi)
    except jax.errors.TracerArrayConversionError:
        return  # traced: the pure-JAX guard takes over
    arr = raw.astype(np.float64)
    n_time = arr.shape[0]
    expected = arr[0] + 2.0 * np.pi * np.arange(n_time) / n_time
    err = float(np.max(np.abs(arr - expected)))
    tol = float(_uniform_tolerance(raw.dtype, float(np.max(np.abs(arr))), np))
    if err > tol:
        raise ValueError(
            f"uniform=True requires dphi to be a uniform grid over a FULL "
            f"sidereal turn, dphi[t] = dphi[0] + 2*pi*t/n_time (n_time="
            f"{n_time}); the given grid deviates by {err:.3e} rad "
            f"(tolerance {tol:.1e} for dtype {raw.dtype}). Drop the flag to "
            f"use the exact direct sum on arbitrary sampling."
        )


def _poison_if_nonuniform(dphi: jnp.ndarray, out: jnp.ndarray) -> jnp.ndarray:
    """Return ``out``, or NaN if ``dphi`` violates the uniform contract.

    The traced half of the enforcement, and the reason ``uniform=True``
    cannot silently produce a wrong answer: the FFT path reads only
    ``dphi[0]`` and ``n_time``, so a partial-turn or irregular grid would
    otherwise yield a finite, plausible, badly wrong TOD under ``jit``
    (measured 74% error on a uniform HALF-turn grid). This is pure JAX —
    no host callback, no dispatch on traced values, jit/vmap/grad-safe, and
    under ``vmap`` it poisons only the offending row. Costs one pass over
    ``dphi`` (~15% of an FFT synthesis that is ~50x faster than the sum).
    """
    n_time = dphi.shape[0]
    implied = dphi[0] + 2.0 * np.pi * jnp.arange(n_time, dtype=dphi.dtype) / n_time
    err = jnp.max(jnp.abs(dphi - implied))
    tol = _uniform_tolerance(dphi.dtype, jnp.max(jnp.abs(dphi)), jnp)
    return jnp.where(err > tol, jnp.asarray(jnp.nan, out.dtype), out)


def beam_alm_at_reference(
    beam_alm: jnp.ndarray,
    lst_ref_deg: jnp.ndarray,
    lat_deg: jnp.ndarray,
    az_deg: jnp.ndarray,
    el_deg: jnp.ndarray,
    selfrot_deg: jnp.ndarray = 0.0,
    *,
    lmax: int,
) -> jnp.ndarray:
    """Beam-local packed alms -> celestial-frame packed alms at ``lst_ref``.

    The single Wigner rotation of the drift-scan chain (differentiable in
    every angle). Equivalent to what the generic path applies at the sample
    with ``lst = lst_ref``; every other sample follows by phases.
    """
    _validate_alm("beam_alm", beam_alm, lmax)
    psi, theta, phi = zyz_of_pointing(lst_ref_deg, lat_deg, az_deg, el_deg, selfrot_deg)
    return rotate_alm(beam_alm, psi, theta, phi, lmax=lmax)


def mmodes_from_sky(
    beam_ref_alm: jnp.ndarray,
    sky_alm: jnp.ndarray,
    *,
    lmax: int | None = None,
) -> jnp.ndarray:
    """m-modes of the drift-scan TOD: ``Ṽ_m = Σ_l conj(B_lm(ref))·S̃_lm``.

    Args:
        beam_ref_alm: packed celestial-frame beam alms at the reference LST
            (:func:`beam_alm_at_reference` output).
        sky_alm: packed QUADRATURE sky alms (see :mod:`limtod_jax.core` for
            the exactness contract with numpy limTOD).
        lmax: static band-limit; inferred from the alm length when omitted.

    Returns
    -------
        Complex ``(lmax+1,)`` array, entry ``m`` holding ``Ṽ_m`` for
        ``m ≥ 0`` (the note's eqn (14); real fields make
        ``Ṽ_{−m} = conj(Ṽ_m)`` redundant). These are the Fourier coefficients
        of the sidereal-day TOD, eqn (15). Leading batch dims pass through.
    """
    if beam_ref_alm.shape[-1] != sky_alm.shape[-1]:
        raise ValueError(
            f"alm length mismatch: {beam_ref_alm.shape[-1]} vs {sky_alm.shape[-1]}"
        )
    if lmax is None:
        from limtod_jax.alm import lmax_of_nalm

        lmax = lmax_of_nalm(beam_ref_alm.shape[-1])
    else:
        _validate_alm("beam_ref_alm", beam_ref_alm, lmax)
    _, ms = packed_lm_arrays(lmax)
    prod = jnp.conj(beam_ref_alm) * sky_alm
    out = jnp.zeros(prod.shape[:-1] + (lmax + 1,), dtype=prod.dtype)
    return out.at[..., ms].add(prod)


def tod_from_mmodes(mmodes: jnp.ndarray, dphi: jnp.ndarray) -> jnp.ndarray:
    """Synthesize the TOD from its m-modes at arbitrary LST offsets.

    ``V(Δ_t) = Ṽ_0 + 2·Re Σ_{m≥1} Ṽ_m·exp(+i·m·Δ_t)`` — the note's eqn
    (13) (phase sign locked numerically against the generic path). ``dphi``
    holds ``lst − lst_ref`` in RADIANS; any sampling, uniform or not.

    Accumulates with ``lax.scan`` over m — memory O(n_time + lmax), never
    materializing the (n_time, lmax+1) phase matrix — so large TODs at
    large lmax stay cheap. The scan step is ``jax.checkpoint``-ed so this
    bound holds under reverse-mode AD too: the backward pass recomputes
    the per-m cos/sin planes instead of storing them (a ~2x trig-FLOP
    price; without it, grad residuals are exactly the phase matrix).
    Differentiable in both arguments.
    """
    _validate_dphi(dphi)
    n_m = mmodes.shape[-1]
    if mmodes.ndim != 1:
        raise ValueError(
            f"mmodes must be 1D (lmax+1,) (batch with jax.vmap), got {mmodes.shape}"
        )
    real_dtype = jnp.result_type(jnp.real(mmodes).dtype, dphi.dtype)
    weights = jnp.asarray(np.where(np.arange(n_m) > 0, 2.0, 1.0), dtype=real_dtype)
    m_values = jnp.arange(n_m, dtype=real_dtype)

    @jax.checkpoint
    def step(acc, inputs):
        c_m, w_m, m = inputs
        phase = m * dphi
        acc = acc + w_m * (
            jnp.real(c_m) * jnp.cos(phase) - jnp.imag(c_m) * jnp.sin(phase)
        )
        return acc, None

    acc0 = jnp.zeros(dphi.shape, dtype=real_dtype)
    tod, _ = jax.lax.scan(step, acc0, (mmodes, weights, m_values))
    return tod


def tod_from_mmodes_uniform(
    mmodes: jnp.ndarray, n_time: int, *, phase0: jnp.ndarray = 0.0
) -> jnp.ndarray:
    """FFT synthesis of :func:`tod_from_mmodes` on a uniform full-turn grid.

    Exactly equal to ``tod_from_mmodes(mmodes, phase0 + 2π·arange(n_time)/
    n_time)`` (roundoff), in O(n_time·log n_time) independent of lmax.

    ``irfft`` already applies the Hermitian ``+2·Re`` expansion and a 1/n
    factor, so the m-modes only need the reference-phase rotation
    ``Ṽ_m·e^{+imφ₀}`` before zero-padding — and one factor of ``n_time``
    back afterwards.

    Args:
        mmodes: complex ``(lmax+1,)`` m-modes (:func:`mmodes_from_sky`).
        n_time: STATIC number of samples covering the full turn; must
            satisfy ``2·lmax < n_time``.
        phase0: traced scalar ``Δ_0`` [rad] — the grid's first LST offset.
    """
    mmodes = jnp.asarray(mmodes)
    if mmodes.ndim != 1:
        raise ValueError(
            f"mmodes must be 1D (lmax+1,) (batch with jax.vmap), got {mmodes.shape}"
        )
    n_m = mmodes.shape[-1]
    _validate_nyquist(n_time, n_m - 1)
    # phase0 must enter the promotion (passed by VALUE so its weak type does
    # not force complex128 on the 0.0 default): the direct path promotes on
    # both traced inputs, and dropping the time-axis dtype here would make
    # the FFT path silently less precise than the sum it must reproduce.
    dtype = jnp.result_type(mmodes.dtype, np.complex64, phase0)
    m = jnp.arange(n_m, dtype=jnp.result_type(jnp.real(mmodes).dtype, np.float32))
    rotated = (mmodes * jnp.exp(1j * m * jnp.asarray(phase0))).astype(dtype)
    pad = jnp.zeros(n_time // 2 + 1, dtype=dtype).at[:n_m].set(rotated)
    return jnp.fft.irfft(pad, n=n_time) * n_time


def _zeta_uniform(tod: jnp.ndarray, lmax: int, phase0: jnp.ndarray) -> jnp.ndarray:
    """``ζ_m = Σ_t y_t·e^{−imΔ_t}`` on a uniform full-turn grid, via rfft.

    ``rfft(y)[m] = Σ_t y_t·e^{−2πimt/n}`` is precisely the ζ sum once the
    reference phase is factored out, so this is the exact FFT counterpart
    of :func:`_zeta` (and hence keeps the adjoint an exact transpose).
    """
    spectrum = jnp.fft.rfft(tod)[: lmax + 1]
    m = jnp.arange(lmax + 1, dtype=jnp.result_type(tod.dtype, np.float32))
    return spectrum * jnp.exp(-1j * m * jnp.asarray(phase0))


def mmodes_from_tod_uniform(
    tod: jnp.ndarray, *, lmax: int, phase0: jnp.ndarray = 0.0
) -> jnp.ndarray:
    """FFT form of :func:`mmodes_from_tod` on a uniform full-turn grid.

    ``Ṽ_m = rfft(V)[m]·e^{−imφ₀}/n_time`` — the exact inverse of
    :func:`tod_from_mmodes_uniform` when ``2·lmax < n_time``, and the
    natural way to carry real drift-scan DATA into m-space (one FFT
    instead of an O(n_time·lmax) sum).
    """
    tod = jnp.asarray(tod)
    if tod.ndim != 1:
        raise ValueError(
            f"tod must be 1D (n_time,) (batch with jax.vmap), got {tod.shape}"
        )
    _validate_nyquist(tod.shape[0], lmax)
    return _zeta_uniform(tod, lmax, phase0) / tod.shape[0]


def mmodes_from_tod(tod: jnp.ndarray, dphi: jnp.ndarray, *, lmax: int) -> jnp.ndarray:
    """Estimate m-modes from a TOD: ``Ṽ_m ≈ (1/n_t)·Σ_t V_t·exp(−i·m·Δ_t)``.

    The discrete form of the note's eqn (15). This inverts
    :func:`tod_from_mmodes` EXACTLY only when the sampling is uniform over
    a full sidereal circle (``Δ_t = 2π·t/n_t + const``) with
    ``n_time > 2·lmax``; otherwise it is the plain DFT estimate (aliased /
    biased by gaps). For rigorous non-uniform analysis, build the forward
    operator and solve least squares instead.
    """
    _validate_dphi(dphi)
    if tod.ndim != 1 or tod.shape[0] != dphi.shape[0]:
        raise ValueError(
            f"tod must be 1D of length n_time={dphi.shape[0]} "
            f"(batch with jax.vmap), got shape {tod.shape}"
        )
    n_t = dphi.shape[0]
    real_dtype = jnp.result_type(tod.dtype, dphi.dtype)
    m_values = jnp.arange(lmax + 1, dtype=real_dtype)

    def one_m(m):
        phase = m * dphi
        return (
            jnp.sum(tod * jnp.cos(phase)) - 1j * jnp.sum(tod * jnp.sin(phase))
        ) / n_t

    return jax.lax.map(one_m, m_values)


def _zeta(tod: jnp.ndarray, dphi: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """``ζ_m = Σ_t y_t·exp(−i·m·Δ_t)`` for m in [0, lmax] — adjoint phases."""
    real_dtype = jnp.result_type(tod.dtype, dphi.dtype)
    m_values = jnp.arange(lmax + 1, dtype=real_dtype)

    def one_m(m):
        phase = m * dphi
        return jnp.sum(tod * jnp.cos(phase)) - 1j * jnp.sum(tod * jnp.sin(phase))

    return jax.lax.map(one_m, m_values)


def _synthesize(
    mmodes: jnp.ndarray, dphi: jnp.ndarray, uniform: bool
) -> jnp.ndarray:
    """Dispatch the m-mode -> TOD synthesis on the STATIC ``uniform`` flag."""
    if uniform:
        tod = tod_from_mmodes_uniform(mmodes, dphi.shape[0], phase0=dphi[0])
        return _poison_if_nonuniform(dphi, tod)
    return tod_from_mmodes(mmodes, dphi)


def driftscan_tod(
    beam_ref_alm: jnp.ndarray,
    sky_alm: jnp.ndarray,
    dphi: jnp.ndarray,
    *,
    lmax: int,
    normalize: bool = False,
    ones_alm: jnp.ndarray | None = None,
    uniform: bool = False,
) -> jnp.ndarray:
    """Drift-scan sky TOD via m-modes — the fast exact special case of
    :func:`limtod_jax.core.generate_tod_sky` for fixed az/el/selfrot.

    Args:
        beam_ref_alm: packed celestial-frame beam alms at the reference LST
            (:func:`beam_alm_at_reference`).
        sky_alm: packed QUADRATURE sky alms (exactness contract of
            :mod:`limtod_jax.core`).
        dphi: ``(n_time,)`` LST offsets ``deg2rad(lst_deg − lst_ref_deg)``.
        lmax: static band-limit matching the alm lengths.
        normalize: static; divide each sample by the rotated beam's pixel
            sum (numpy limTOD's ``normalize_beam`` semantics). Along a
            drift the denominator is constant up to the ones-map's tiny
            m ≠ 0 quadrature residues; it is computed exactly anyway.
        ones_alm: quadrature alms of the ones map; required iff normalize.
        uniform: STATIC opt-in to the FFT synthesis — assert that ``dphi``
            is a uniform grid over a full sidereal turn
            (``dphi[t] = dphi[0] + 2π·t/n_time``, ``2·lmax < n_time``).
            Same result to roundoff, O(n_time·log n_time) instead of
            O(n_time·lmax). Verified when ``dphi`` is concrete; on traced
            input it is the caller's contract. Never auto-detected — the
            dispatch must not depend on traced values.

    Returns
    -------
        ``(n_time,)`` real TOD, equal to the generic per-sample-rotation
        path to float64 roundoff (oracle-locked in the test suite).
    """
    _validate_alm("beam_ref_alm", beam_ref_alm, lmax)
    _validate_alm("sky_alm", sky_alm, lmax)
    _validate_dphi(dphi)
    if normalize and ones_alm is None:
        raise ValueError(
            "normalize=True requires ones_alm — the quadrature alms of the "
            "ones map (limtod_jax.hpx.ones_quadrature_alm)"
        )
    if uniform:
        _validate_nyquist(dphi.shape[0], lmax)
        check_uniform_grid(dphi)
    num = _synthesize(mmodes_from_sky(beam_ref_alm, sky_alm, lmax=lmax), dphi, uniform)
    if not normalize:
        return num
    assert ones_alm is not None  # checked above; narrows the type
    den = _synthesize(mmodes_from_sky(beam_ref_alm, ones_alm, lmax=lmax), dphi, uniform)
    return num / den


def driftscan_tod_adjoint(
    tod: jnp.ndarray,
    beam_ref_alm: jnp.ndarray,
    dphi: jnp.ndarray,
    *,
    lmax: int,
    normalize: bool = False,
    ones_alm: jnp.ndarray | None = None,
    uniform: bool = False,
) -> jnp.ndarray:
    """Exact transpose of :func:`driftscan_tod` in the sky slot.

    Same inner products as :func:`limtod_jax.core.generate_tod_sky_adjoint`
    (weighted ``alm_dot`` on the sky side, plain dot on the TOD side), so
    ``⟨forward(x), y⟩ == ⟨x, adjoint(y)⟩_w`` to roundoff. In m-mode form the
    accumulation of rotated beams collapses to

        ``adj_lm = B_lm(ref) · ζ_m``,  ``ζ_m = Σ_t y'_t·exp(−i·m·Δ_t)``

    with ``y' = y / den`` when ``normalize`` (den = the forward
    denominator). O(n_time·lmax) like the forward pass — or
    O(n_time·log n_time) with ``uniform=True``, where ζ is a forward real
    FFT (the exact counterpart of the synthesis ``irfft``, so the transpose
    property is preserved; dot-tested in both modes).
    """
    _validate_alm("beam_ref_alm", beam_ref_alm, lmax)
    _validate_dphi(dphi)
    if tod.ndim != 1 or tod.shape[0] != dphi.shape[0]:
        raise ValueError(
            f"tod must be 1D of length n_time={dphi.shape[0]} "
            f"(batch with jax.vmap), got shape {tod.shape}"
        )
    if uniform:
        _validate_nyquist(dphi.shape[0], lmax)
        check_uniform_grid(dphi)
    if normalize:
        if ones_alm is None:
            raise ValueError(
                "normalize=True requires ones_alm — the quadrature alms of "
                "the ones map (limtod_jax.hpx.ones_quadrature_alm)"
            )
        den = _synthesize(
            mmodes_from_sky(beam_ref_alm, ones_alm, lmax=lmax), dphi, uniform
        )
        tod = tod / den
    zeta = (
        _zeta_uniform(tod, lmax, dphi[0]) if uniform else _zeta(tod, dphi, lmax)
    )
    _, ms = packed_lm_arrays(lmax)
    out = beam_ref_alm * zeta[ms]
    return _poison_if_nonuniform(dphi, out) if uniform else out


def _pixel_thetas(nside: int) -> np.ndarray:
    """Colatitude θ of every RING-ordered HEALPix pixel (numpy, static).

    Built from s2fft's own HEALPix ring geometry (the same functions its
    transforms use), so it is consistent-by-construction with
    :mod:`limtod_jax.hpx` — and oracle-locked against ``healpy.pix2ang``
    in the test suite. healpy itself stays banned from this package.
    """
    from s2fft.sampling.s2_samples import nphi_ring, thetas

    ring_theta = thetas(sampling="healpix", nside=nside)
    counts = np.array([nphi_ring(t, nside) for t in range(ring_theta.shape[0])])
    return np.repeat(ring_theta, counts)


def horizon_weights(nside: int, apod_deg: float = 0.0) -> np.ndarray:
    """Per-pixel horizon weights in the HORIZONTAL-frame HEALPix chart.

    In limTOD's horizontal chart the pole is the zenith, so a pixel at
    colatitude θ sits at elevation ``90° − θ`` regardless of φ (chart
    convention pinned in ``tests/test_beam_orientation.py``); the horizon
    cut is a pure θ function. ``apod_deg = 0`` gives the hard cut
    ``el > 0``; positive values a cosine ramp — 0 at the horizon rising to
    1 at ``el = apod_deg`` — the mitigation knob for the Gibbs ringing a
    hard cut produces at finite lmax (see ``docs/driftscan.md``).

    Static numpy: weights depend only on (nside, apod_deg), never traced.
    """
    el = 90.0 - np.rad2deg(_pixel_thetas(nside))
    if apod_deg == 0.0:
        return (el > 0.0).astype(np.float64)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip(el, 0.0, apod_deg) / apod_deg))
    return np.where(el <= 0.0, 0.0, np.where(el >= apod_deg, 1.0, ramp))


def horizon_masked_beam_alm(
    beam_alm: jnp.ndarray,
    az_deg: jnp.ndarray,
    el_deg: jnp.ndarray,
    selfrot_deg: jnp.ndarray = 0.0,
    *,
    nside: int,
    lmax: int,
    apod_deg: float = 0.0,
    iterations: int = 3,
) -> jnp.ndarray:
    """Horizon-masked beam, returned as BEAM-LOCAL packed alms.

    Rotates the beam into the horizontal frame with the (az, el, selfrot)
    sub-chain (no LST/latitude), multiplies by the static
    :func:`horizon_weights` elevation taper, re-analyzes with the
    healpy-equivalent :func:`limtod_jax.hpx.map2alm_iter`, and rotates
    back. The output lives in the same beam-local frame as the input, so
    it drops into EITHER the drift-scan path or the generic
    :func:`limtod_jax.core.generate_tod_sky` unchanged.

    The result is the best band-limited representation of the sharply (or
    apodized-)masked beam — a non-band-limited target, so Gibbs ringing
    appears near the cut at small lmax. See ``docs/driftscan.md`` for
    magnitudes and the ``apod_deg`` mitigation study.

    Requires x64 (s2fft HEALPix transforms; :mod:`limtod_jax.hpx`).
    """
    _validate_alm("beam_alm", beam_alm, lmax)
    psi, theta, phi = zyzyz2zyz(
        0.0,
        0.0,
        -jnp.asarray(az_deg),
        jnp.asarray(el_deg) - 90.0,
        jnp.asarray(selfrot_deg),
    )
    beam_h = rotate_alm(beam_alm, psi, theta, phi, lmax=lmax)
    map_h = alm2map(beam_h, nside=nside, lmax=lmax)
    weights = jnp.asarray(horizon_weights(nside, apod_deg))
    masked_alm = map2alm_iter(
        map_h * weights, nside=nside, lmax=lmax, iterations=iterations
    )
    # Inverse rotation: (psi, theta, phi) -> (-phi, -theta, -psi), locked
    # numerically (roundtrip test in the suite).
    return rotate_alm(masked_alm, -phi, -theta, -psi, lmax=lmax)


class DriftScanMmode(eqx.Module):
    """Drift-scan m-mode operator: sky alms -> TOD, one beam rotation total.

    An :class:`equinox.Module` (frozen pytree) bundling the reference-frame
    beam with the LST sampling, ready for jit/vmap/grad and for use as a
    field of larger rheplicant models. Build it with
    :meth:`from_pointing` (friendly constructor: beam-local alms +
    pointing) or directly from a precomputed ``beam_ref_alm``.

    Attributes:
        beam_ref_alm: ``(n_alm,)`` packed celestial-frame beam alms at the
            reference LST (traced — differentiable beam).
        dphi: ``(n_time,)`` LST offsets from the reference, RADIANS (traced).
        lmax: static band-limit matching ``beam_ref_alm``.
        normalize: static; numpy limTOD ``normalize_beam`` semantics.
        ones_alm: quadrature alms of the ones map; required iff ``normalize``.
        uniform_sampling: static; assert that ``dphi`` is a uniform grid over
            a full sidereal turn and use the FFT synthesis
            (see :func:`driftscan_tod`). Validated at construction whenever
            ``dphi`` is concrete.
    """

    beam_ref_alm: jax.Array
    dphi: jax.Array
    lmax: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True, default=False)
    ones_alm: jax.Array | None = None
    uniform_sampling: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        if self.beam_ref_alm.ndim != 1:
            raise ValueError(
                f"beam_ref_alm must be 1D (n_alm,) — the operator is "
                f"single-beam; batch frequencies by vmapping the constructor "
                f"or the call — got shape {self.beam_ref_alm.shape}"
            )
        _validate_alm("beam_ref_alm", self.beam_ref_alm, self.lmax)
        _validate_dphi(self.dphi)
        if self.normalize and self.ones_alm is None:
            raise ValueError(
                "normalize=True requires ones_alm "
                "(limtod_jax.hpx.ones_quadrature_alm)"
            )
        if self.uniform_sampling:
            _validate_nyquist(self.dphi.shape[0], self.lmax)
            check_uniform_grid(self.dphi)

    @classmethod
    def from_pointing(
        cls,
        beam_alm: jnp.ndarray,
        lst_deg: jnp.ndarray,
        lat_deg: float,
        az_deg: float,
        el_deg: float,
        selfrot_deg: float = 0.0,
        *,
        lmax: int,
        lst_ref_deg: float | jnp.ndarray | None = None,
        normalize: bool = False,
        nside: int | None = None,
        horizon_mask: bool = False,
        apod_deg: float = 0.0,
        mask_iterations: int = 3,
        uniform_sampling: bool = False,
    ) -> "DriftScanMmode":
        """Build the operator from beam-local alms and drift-scan pointing.

        Args:
            beam_alm: ``(n_alm,)`` packed beam alms in the beam-local frame
                (as ``hp.map2alm(beam_map)`` computes them in numpy limTOD).
            lst_deg: ``(n_time,)`` local sidereal times [deg].
            lat_deg / az_deg / el_deg / selfrot_deg: the fixed site latitude
                and pointing of the drift scan [deg].
            lmax: static band-limit.
            lst_ref_deg: reference LST [deg]; defaults to ``lst_deg[0]``.
            normalize: numpy limTOD ``normalize_beam`` semantics (static).
                Needs ``nside`` for the ones-map quadrature alms.
            nside: HEALPix nside — required iff ``normalize`` or
                ``horizon_mask``.
            horizon_mask: apply :func:`horizon_masked_beam_alm` before the
                reference rotation (default off, matching numpy limTOD).
            apod_deg / mask_iterations: forwarded to the mask.
            uniform_sampling: opt in to the FFT synthesis (static); requires
                ``lst_deg`` to be a uniform grid over a full sidereal turn.
        """
        lst_deg = jnp.asarray(lst_deg)
        if lst_deg.ndim != 1:
            raise ValueError(f"lst_deg must be 1D (n_time,), got {lst_deg.shape}")
        if (normalize or horizon_mask) and nside is None:
            raise ValueError("nside is required when normalize or horizon_mask is set")
        if horizon_mask:
            assert nside is not None  # checked above; narrows the type
            beam_alm = horizon_masked_beam_alm(
                beam_alm,
                az_deg,
                el_deg,
                selfrot_deg,
                nside=nside,
                lmax=lmax,
                apod_deg=apod_deg,
                iterations=mask_iterations,
            )
        if lst_ref_deg is None:
            lst_ref_deg = lst_deg[0]
        beam_ref = beam_alm_at_reference(
            beam_alm, lst_ref_deg, lat_deg, az_deg, el_deg, selfrot_deg, lmax=lmax
        )
        ones_alm = None
        if normalize:
            assert nside is not None  # checked above; narrows the type
            ones_alm = ones_quadrature_alm(nside=nside, lmax=lmax)
        return cls(
            beam_ref_alm=beam_ref,
            dphi=jnp.deg2rad(lst_deg - lst_ref_deg),
            lmax=lmax,
            normalize=normalize,
            ones_alm=ones_alm,
            uniform_sampling=uniform_sampling,
        )

    def mmodes(self, sky_alm: jnp.ndarray) -> jnp.ndarray:
        """m-modes ``Ṽ_m`` (complex, ``(lmax+1,)``) of the drift-scan TOD."""
        return mmodes_from_sky(self.beam_ref_alm, sky_alm, lmax=self.lmax)

    def __call__(self, sky_alm: jnp.ndarray) -> jnp.ndarray:
        """``(n_time,)`` sky TOD for packed quadrature ``sky_alm``."""
        return driftscan_tod(
            self.beam_ref_alm,
            sky_alm,
            self.dphi,
            lmax=self.lmax,
            normalize=self.normalize,
            ones_alm=self.ones_alm,
            uniform=self.uniform_sampling,
        )

    def adjoint(self, tod: jnp.ndarray) -> jnp.ndarray:
        """Exact sky-slot transpose of :meth:`__call__` (packed alms out)."""
        return driftscan_tod_adjoint(
            tod,
            self.beam_ref_alm,
            self.dphi,
            lmax=self.lmax,
            normalize=self.normalize,
            ones_alm=self.ones_alm,
            uniform=self.uniform_sampling,
        )
