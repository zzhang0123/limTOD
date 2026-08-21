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

m-mode analysis is the standard harmonic treatment of drift-scan (transit)
observations; the conventions and equation numbering here follow *M-mode RIME
explicit in beam, fringe and sky modes* (Jan 2024,
https://zh-zhang.com/myNotes/MmodeNote.pdf) — the expressions above are its
eqns (13)–(15) — specialized to that note's "MT interpretation" with the
fringe ≡ 1 (single-dish autocorrelation): the modulated beam IS the primary
beam.

The ``Ṽ_m`` are the **m-modes** — the Fourier coefficients of the
sidereal-day-periodic TOD — and the projection sky → m-modes is a single
per-(l,m) product with the reference-frame beam.

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

POLARISATION. Everything above is spin-independent, and that is not an
accident: a rotation about z contributes ``D^l_{m'm}(α,0,0) = δ_{m'm}·e^{−imα}``
whatever the spin, so the E and B alms of a polarised beam pick up the SAME
drift phase as T. Set the static ``npol`` (1/3/4) and give the alms a leading
Stokes axis ``(..., npol, n_alm)`` of packed ``T``/``E``/``B``/``V`` rows; the
m-modes gain one extra sum,

    ``Ṽ_m = Σ_row Σ_l conj(B_row,lm(ref)) · S̃_row,lm``

and nothing else changes — same phase law, same synthesis, same FFT fast
path, same block-diagonal-in-m structure. See :mod:`limtod_jax.stokes` for
the contract and ``tests/limtod_jax/test_polarisation.py`` for the locks
(the m-mode TOD is checked against BOTH the generic JAX path and numpy
limTOD's own full-Stokes ``generate_TOD_sky``).

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

from limtod_jax.alm import packed_lm_arrays
from limtod_jax.angles import zyz_of_pointing, zyzyz2zyz
from limtod_jax.core import rotate_alm
from limtod_jax.hpx import alm2map, map2alm_iter, ones_quadrature_alm
from limtod_jax.stokes import (
    STOKES_ALM_ROWS,
    STOKES_MAP_ROWS,
    match_npol,
    stokes_i,
    validate_npol,
    validate_pol_alm,
    validate_unpolarised,
)
from limtod_jax.wigner import generate_rotate_dls

# Above how many (lmax+1)*n_time phase-matrix entries the non-uniform
# synthesis and its adjoint stop materializing the matrix and fall back to a
# sequential accumulation. 10**7 entries is 160 MB in complex128 / 80 MB in
# float64 — the point where the matrix starts to rival the map<->alm
# transforms that dominate a realistic peak, while everything below it is
# small enough that paying 28x the runtime to avoid it is a bad trade.
# Static: it is compared against shapes, never traced values.
_PHASE_MATRIX_MAX = 10**7


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
    dl_array: jnp.ndarray | None = None,
    npol: int | None = None,
) -> jnp.ndarray:
    """Beam-local packed alms -> celestial-frame packed alms at ``lst_ref``.

    The single Wigner rotation of the drift-scan chain (differentiable in
    every angle). Equivalent to what the generic path applies at the sample
    with ``lst = lst_ref``; every other sample follows by phases.

    Accepts a Stokes stack ``(npol, n_alm)`` when ``npol`` is set — one
    rotation for the whole stack, sharing the Risbo recursion across rows.

    Args:
        npol: static 1/3/4 asserting the leading Stokes axis; ``None`` for
            the unpolarised case (unchanged).
        dl_array: optional precomputed Wigner-d plane, from
            :func:`dl_plane_for_pointing` with the SAME
            ``(lat, az, el, selfrot)``. Those four fix the polar angle, and
            ``lst_ref`` does not enter it — LST lands in the first-applied
            slot of the zyz composition, so it shifts ``psi`` alone (verified
            bit-exactly in ``tests/limtod_jax/test_driftscan.py``). A caller
            with a fixed pointing can therefore build the plane once and skip
            the Risbo recursion on every evaluation, which is the only way to
            amortize the rotation when the BEAM is the fitted parameter and
            the reference-frame trick is unavailable.
    """
    validate_pol_alm("beam_alm", beam_alm, lmax, validate_npol(npol))
    psi, theta, phi = zyz_of_pointing(lst_ref_deg, lat_deg, az_deg, el_deg, selfrot_deg)
    return rotate_alm(beam_alm, psi, theta, phi, lmax=lmax, dl_array=dl_array)


def dl_plane_for_pointing(
    lat_deg: jnp.ndarray,
    az_deg: jnp.ndarray,
    el_deg: jnp.ndarray,
    selfrot_deg: jnp.ndarray = 0.0,
    *,
    lmax: int,
) -> jnp.ndarray:
    """Wigner-d plane for a fixed drift-scan pointing, for reuse across calls.

    Feed the result to :func:`beam_alm_at_reference` as ``dl_array``. The plane
    is a function of the pointing ONLY — the reference LST is deliberately not
    an argument, because it cannot enter: it composes on the right of the zyz
    rotation and moves ``psi``, never the polar angle the plane is built from.

    Size is ``(lmax+1)·(2lmax+1)²`` reals — 215 MB at lmax=191 in float64, so
    pass float32 angles if the plane is the memory ceiling (the Risbo
    recursion is float32-stable to ~2e-6, and the dtype now follows the angles
    rather than the session default).
    """
    _, theta, _ = zyz_of_pointing(0.0, lat_deg, az_deg, el_deg, selfrot_deg)
    return generate_rotate_dls(lmax + 1, theta)


def mmodes_from_sky(
    beam_ref_alm: jnp.ndarray,
    sky_alm: jnp.ndarray,
    *,
    lmax: int | None = None,
    npol: int | None = None,
) -> jnp.ndarray:
    """m-modes of the drift-scan TOD: ``Ṽ_m = Σ_row Σ_l conj(B_lm(ref))·S̃_lm``.

    Args:
        beam_ref_alm: packed celestial-frame beam alms at the reference LST
            (:func:`beam_alm_at_reference` output).
        sky_alm: packed QUADRATURE sky alms (see :mod:`limtod_jax.core` for
            the exactness contract with numpy limTOD).
        lmax: static band-limit; inferred from the alm length when omitted.
        npol: static 1/3/4 — CONTRACT a leading Stokes axis
            ``(..., npol, n_alm)`` of packed T/E/B/V rows, adding the row sum
            to eqn (14). ``None`` (default) leaves every leading axis a batch
            axis, exactly as before.

    Returns
    -------
        Complex ``(lmax+1,)`` array, entry ``m`` holding ``Ṽ_m`` for
        ``m ≥ 0`` (the note's eqn (14); real fields make
        ``Ṽ_{−m} = conj(Ṽ_m)`` redundant). These are the Fourier coefficients
        of the sidereal-day TOD, eqn (15). Leading batch dims — everything
        outside the Stokes axis — pass through.

        The polarised m-modes are a SINGLE complex series, not one per
        Stokes row: the rows are contracted into each TOD sample, so the
        block-diagonal-in-m map-making structure is unchanged (it is the SKY
        that gains rows, on the adjoint side).
    """
    if beam_ref_alm.shape[-1] != sky_alm.shape[-1]:
        raise ValueError(
            f"alm length mismatch: {beam_ref_alm.shape[-1]} vs {sky_alm.shape[-1]}"
        )
    npol = validate_npol(npol)
    if lmax is None:
        from limtod_jax.alm import lmax_of_nalm

        lmax = lmax_of_nalm(beam_ref_alm.shape[-1])
    validate_pol_alm("beam_ref_alm", beam_ref_alm, lmax, npol)
    validate_pol_alm("sky_alm", sky_alm, lmax, npol)
    match_npol("beam_ref_alm", beam_ref_alm, "sky_alm", sky_alm, npol)
    _, ms = packed_lm_arrays(lmax)
    prod = jnp.conj(beam_ref_alm) * sky_alm
    out = jnp.zeros(prod.shape[:-1] + (lmax + 1,), dtype=prod.dtype)
    mm = out.at[..., ms].add(prod)
    return mm if npol is None else jnp.sum(mm, axis=-2)


def tod_from_mmodes(mmodes: jnp.ndarray, dphi: jnp.ndarray) -> jnp.ndarray:
    """Synthesize the TOD from its m-modes at arbitrary LST offsets.

    ``V(Δ_t) = Ṽ_0 + 2·Re Σ_{m≥1} Ṽ_m·exp(+i·m·Δ_t)`` — the note's eqn
    (13) (phase sign locked numerically against the generic path). ``dphi``
    holds ``lst − lst_ref`` in RADIANS; any sampling, uniform or not.

    Two implementations, chosen by a STATIC size threshold
    (``_PHASE_MATRIX_MAX``), because neither wins everywhere:

    * **Small** — form the ``(lmax+1, n_time)`` phase matrix and contract it.
      One big matmul, which is what the hardware wants. Measured at
      lmax=191 / n_time=512: **0.56 ms against 15.65 ms** for the scan, and
      0.60 ms against 19.03 ms under reverse-mode AD. The matrix costs
      1.5 MB — 1.3 % of that call's 114 MB peak, so the memory the scan
      protects is not worth 28x the time.
    * **Large** — accumulate with ``lax.scan`` over m: memory
      O(n_time + lmax), never materializing the phase matrix. At
      n_time=86400 / lmax=191 that matrix would be 133 MB and would double
      the peak; at lmax=1024 it is 708 MB. The scan step is
      ``jax.checkpoint``-ed so the bound survives reverse-mode AD (the
      backward pass recomputes the cos/sin planes rather than storing them;
      without it, the grad residual IS the phase matrix).

    Both are differentiable in both arguments and produce the same numbers to
    roundoff; only the time/memory trade differs.
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

    if n_m * dphi.shape[0] <= _PHASE_MATRIX_MAX:
        phase = m_values[:, None] * dphi[None, :]  # (n_m, n_time)
        return jnp.einsum(
            "m,mt->t",
            weights * jnp.real(mmodes),
            jnp.cos(phase),
        ) - jnp.einsum("m,mt->t", weights * jnp.imag(mmodes), jnp.sin(phase))

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
    return _zeta(tod, dphi, lmax) / dphi.shape[0]


def _zeta(tod: jnp.ndarray, dphi: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """``ζ_m = Σ_t y_t·exp(−i·m·Δ_t)`` for m in [0, lmax] — adjoint phases.

    Same static size threshold as :func:`tod_from_mmodes`, for the same
    reason: below it, one matmul against the ``(lmax+1, n_time)`` phase
    matrix (0.54 ms at lmax=191 / n_time=512); above it, a sequential
    ``lax.map`` that never materializes the matrix (2.63 ms, but bounded
    memory). This is the transpose direction, so the two must switch on the
    SAME condition — otherwise forward and adjoint would have different
    memory profiles at the same size, which is exactly what a map-making
    iteration cannot afford.
    """
    real_dtype = jnp.result_type(tod.dtype, dphi.dtype)
    m_values = jnp.arange(lmax + 1, dtype=real_dtype)

    if (lmax + 1) * dphi.shape[0] <= _PHASE_MATRIX_MAX:
        phase = m_values[:, None] * dphi[None, :]  # (n_m, n_time)
        return jnp.einsum("t,mt->m", tod, jnp.cos(phase)) - 1j * jnp.einsum(
            "t,mt->m", tod, jnp.sin(phase)
        )

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
    npol: int | None = None,
) -> jnp.ndarray:
    """Drift-scan sky TOD via m-modes — the fast exact special case of
    :func:`limtod_jax.core.generate_tod_sky` for fixed az/el/selfrot.

    Args:
        beam_ref_alm: packed celestial-frame beam alms at the reference LST
            (:func:`beam_alm_at_reference`), ``(n_alm,)`` or
            ``(npol, n_alm)``.
        sky_alm: packed QUADRATURE sky alms (exactness contract of
            :mod:`limtod_jax.core`), same shape.
        dphi: ``(n_time,)`` LST offsets ``deg2rad(lst_deg − lst_ref_deg)``.
        lmax: static band-limit matching the alm lengths.
        normalize: static; divide each sample by the rotated beam's pixel
            sum (numpy limTOD's ``normalize_beam`` semantics) — the
            **Stokes-I** row's sum when polarised, applied to every row, as
            ``pointing_beam_in_eq_sys`` does. Along a drift the denominator
            is constant up to the ones-map's tiny m ≠ 0 quadrature residues;
            it is computed exactly anyway.
        ones_alm: quadrature alms of the ones map; required iff normalize.
            Always a single unpolarised row ``(n_alm,)``.
        npol: static 1/3/4 — the Stokes axis to CONTRACT (see
            :mod:`limtod_jax.stokes`). Costs one extra sum in the m-mode
            projection and nothing at all in the synthesis, since the drift
            phase is spin-independent.
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
    npol = validate_npol(npol)
    validate_pol_alm("beam_ref_alm", beam_ref_alm, lmax, npol)
    validate_pol_alm("sky_alm", sky_alm, lmax, npol)
    match_npol("beam_ref_alm", beam_ref_alm, "sky_alm", sky_alm, npol)
    _validate_dphi(dphi)
    if normalize and ones_alm is None:
        raise ValueError(
            "normalize=True requires ones_alm — the quadrature alms of the "
            "ones map (limtod_jax.hpx.ones_quadrature_alm)"
        )
    if ones_alm is not None:
        validate_unpolarised("ones_alm", ones_alm)
    if uniform:
        _validate_nyquist(dphi.shape[0], lmax)
        check_uniform_grid(dphi)
    num = _synthesize(
        mmodes_from_sky(beam_ref_alm, sky_alm, lmax=lmax, npol=npol), dphi, uniform
    )
    if not normalize:
        return num
    assert ones_alm is not None  # checked above; narrows the type
    den = _synthesize(
        mmodes_from_sky(stokes_i(beam_ref_alm, npol), ones_alm, lmax=lmax),
        dphi,
        uniform,
    )
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
    npol: int | None = None,
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

    Polarised (``npol`` set), ζ is unchanged — it is a property of the TIME
    sampling alone — and the same ζ multiplies every Stokes row, so the
    output is ``(npol, n_alm)``. That is the whole cost of polarised
    map-making here: the normal equations stay block-diagonal in m, with
    ``npol``-sized blocks instead of scalars.
    """
    npol = validate_npol(npol)
    validate_pol_alm("beam_ref_alm", beam_ref_alm, lmax, npol)
    _validate_dphi(dphi)
    if tod.ndim != 1 or tod.shape[0] != dphi.shape[0]:
        raise ValueError(
            f"tod must be 1D of length n_time={dphi.shape[0]} "
            f"(batch with jax.vmap), got shape {tod.shape}"
        )
    if uniform:
        _validate_nyquist(dphi.shape[0], lmax)
        check_uniform_grid(dphi)
    if ones_alm is not None:
        validate_unpolarised("ones_alm", ones_alm)
    if normalize:
        if ones_alm is None:
            raise ValueError(
                "normalize=True requires ones_alm — the quadrature alms of "
                "the ones map (limtod_jax.hpx.ones_quadrature_alm)"
            )
        den = _synthesize(
            mmodes_from_sky(stokes_i(beam_ref_alm, npol), ones_alm, lmax=lmax),
            dphi,
            uniform,
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


def horizon_partition_weights(nside: int) -> np.ndarray:
    """How the beam's SOLID ANGLE divides at the horizon: 1 above, 0 below, 0.5 on.

    Not the same object as :func:`horizon_weights`, and the difference is not
    cosmetic. ``horizon_weights`` is a MASK — what to multiply the beam by
    before re-analysis — and its hard cut is a strict ``el > 0``, so the ring
    of pixels centred exactly ON the horizon (4*nside of them; their elevation
    is exactly zero, not nearly) gets weight 0. As a mask that is a thin edge
    detail. As a PARTITION it is a systematic error: a pixel centred on the
    horizon is half sky and half ground.

    Measured on the quantity that needs a partition -- the above-horizon beam
    fraction ``f_sky`` that splits an antenna temperature into its sky and
    ground shares -- against a projector run on a sky map with the ground
    painted in, at a latitude where the horizon is fixed in celestial
    coordinates. On a ~200 K effect at nside 16: the ring counted as nothing
    costs -8.6 K, as all sky +8.7 K, and half **+0.005 K**. The two one-sided
    errors are symmetric and halve with nside, which is the signature of a
    miscounted ring rather than of anything harmonic.

    There is deliberately no ``apod_deg``: a tapered region does not partition
    a sphere. Apodization belongs to the mask.

    Static numpy: a pure function of ``nside``, never traced.
    """
    el = 90.0 - np.rad2deg(_pixel_thetas(nside))
    return np.where(el > 0.0, 1.0, np.where(el < 0.0, 0.0, 0.5))


def horizon_truncated_beam(
    beam_map: jnp.ndarray,
    *,
    nside: int,
    el_deg: float = 90.0,
    apod_deg: float = 0.0,
    npol: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cut a beam MAP at the horizon, and report the fraction that survives.

    For a drift scan the pointing is fixed, so the horizon is fixed too and the
    truncated beam is a CONSTANT: one elementwise multiply, done once, before
    the single analysis the caller was going to run anyway.

    That is worth having beside :func:`horizon_masked_beam_alm`, which masks the
    ALMS and therefore pays a Wigner rotation, a synthesis, an iterative
    re-analysis and a rotation back on EVERY call — 14.6 ms against 1.79 ms
    unmasked at nside 16 / lmax 47, **8.2x**, of which the re-analysis alone is
    65%. This path costs **1.04x**. The two agree to 2.8e-5 relative; the
    residual is the alm->map->alm round trip the masking path takes BEFORE it
    masks, which this one does not.

    WHY ``el_deg = 90`` IS EXACT AND ANYTHING ELSE IS REFUSED. The mask is a
    pure function of elevation and this chart puts the ZENITH at the pole, while
    a beam-local map puts the BORESIGHT there. At a zenith pointing those poles
    coincide, so the two charts can differ only by a rotation ABOUT that shared
    pole — which a pure-elevation function is invariant under. Azimuth and
    self-rotation are therefore irrelevant and no rotation is needed at all.
    Away from zenith the poles part, the horizon becomes a tilted great circle
    in the beam-local chart, and :func:`horizon_masked_beam_alm` is the tool.

    POLARISATION. The taper is a real per-pixel scalar, so it multiplies
    ``I``, ``Q``, ``U`` and ``V`` alike — no spin algebra is involved and a
    ``(npol, npix)`` map needs nothing but broadcasting. The FRACTION,
    however, is a property of the beam's SOLID ANGLE and must come from the
    Stokes-I row alone: ``f_sky`` is the weight that splits an antenna
    temperature into sky and ground shares, and Q/U integrate to something
    that is not a power at all (a beam with zero net Q would make a per-row
    fraction divide by zero). Pass ``npol`` to select that row.

    Args:
        beam_map: ``(..., npix)`` HEALPix RING beam map(s) in the beam-local
            frame — ``(..., npol, npix)`` Stokes rows ``I,Q,U[,V]`` when
            ``npol`` is set. Traced — gradients flow to the beam.
        nside: HEALPix resolution of ``beam_map``.
        el_deg: boresight elevation [deg]; only 90 is supported (see above).
        apod_deg: cosine-apodization width of the cut [deg of elevation]. The
            MAPS carry the taper; the fraction always uses the hard partition of
            :func:`horizon_partition_weights`.
        npol: static 1/3/4 marking a leading Stokes axis on ``beam_map``;
            ``None`` (default) treats every leading axis as a batch axis.

    Returns:
        ``(truncated_map, sky_fraction)`` — the map with ``beam_map``'s shape,
        the fraction with its leading axes (the Stokes axis dropped when
        ``npol`` is set, since one beam has one sky fraction).

    Raises:
        ValueError: for a non-zenith ``el_deg`` or a bad map length.
    """
    if abs(float(el_deg) - 90.0) > 1e-9:
        raise ValueError(
            f"horizon_truncated_beam supports a zenith pointing (el_deg=90), got "
            f"{el_deg}. Only there do the beam-local and horizontal charts share "
            "a pole, which is what makes a pure-elevation mask applicable without "
            "any rotation. For a tilted pointing use horizon_masked_beam_alm."
        )
    beam_map = jnp.asarray(beam_map)
    npol = validate_npol(npol)
    if beam_map.shape[-1] != 12 * nside**2:
        raise ValueError(
            f"beam_map has {beam_map.shape[-1]} pixels, not 12*nside**2 = "
            f"{12 * nside**2} for nside={nside}."
        )
    if npol is not None and (beam_map.ndim < 2 or beam_map.shape[-2] != npol):
        raise ValueError(
            f"beam_map must have shape (..., npol, npix) = (..., {npol}, "
            f"{12 * nside**2}) for npol={npol} — Stokes rows "
            f"{STOKES_MAP_ROWS[npol]} — got {beam_map.shape}"
        )
    taper = jnp.asarray(horizon_weights(nside, apod_deg))
    partition = jnp.asarray(horizon_partition_weights(nside))
    # f_sky is a solid-angle split: Stokes I only (see the docstring).
    beam_i = beam_map[..., 0, :] if npol is not None else beam_map
    fraction = jnp.sum(beam_i * partition, axis=-1) / jnp.sum(beam_i, axis=-1)
    return beam_map * taper, fraction


def horizon_beam_fraction(
    beam_alm: jnp.ndarray,
    az_deg: jnp.ndarray,
    el_deg: jnp.ndarray,
    selfrot_deg: jnp.ndarray = 0.0,
    *,
    nside: int,
    lmax: int,
    npol: int | None = None,
) -> jnp.ndarray:
    """Above-horizon share of a beam's solid angle, for any fixed pointing.

    ``f_sky = int_above B dOmega / int_4pi B dOmega``, the weight that splits an
    antenna temperature into its sky and ground shares. The alm-side companion
    to :func:`horizon_truncated_beam`, using the same (az, el, selfrot)
    sub-chain as :func:`horizon_masked_beam_alm` so the two describe one beam.

    Computed in PIXEL space, on purpose. The band-limited masked beam that
    :func:`horizon_masked_beam_alm` builds is a Gibbs approximation to a
    discontinuous target, and its own solid-angle integral is off by ~0.7% at
    nside 16 / lmax 47 -- ``map2alm`` of a sharply cut map does not preserve the
    mean. Using that as ``f_sky`` leaves -17 K of a 200 K spill bias; this
    leaves ~0. The two are different objects: one is how the beam's solid angle
    divides, the other is how the visible part weights the sky.

    POLARISATION: ``f_sky`` is a SOLID-ANGLE split, so only the Stokes-I
    (``T``) row enters — pass ``npol`` and the row is selected for you, with
    no spin-2 transform involved. Q/U carry no total power to divide (a beam
    with zero net Q would make a per-row fraction singular), so there is
    deliberately no per-Stokes fraction.

    Args:
        beam_alm: ``(..., n_alm)`` packed beam alms in the BEAM-LOCAL frame,
            or ``(..., npol, n_alm)`` T/E/B/V rows when ``npol`` is set.
        az_deg / el_deg / selfrot_deg: the fixed pointing [deg].
        nside: resolution to evaluate the partition at.
        lmax: band-limit matching ``beam_alm``.
        npol: static 1/3/4 marking the Stokes axis; ``None`` for Stokes I.

    Returns:
        The fraction, with ``beam_alm``'s leading axes (the Stokes axis
        dropped when ``npol`` is set — one beam, one fraction).
    """
    npol = validate_npol(npol)
    validate_pol_alm("beam_alm", beam_alm, lmax, npol)
    beam_alm = stokes_i(beam_alm, npol)
    psi, theta, phi = zyzyz2zyz(
        0.0, 0.0, -jnp.asarray(az_deg), jnp.asarray(el_deg) - 90.0,
        jnp.asarray(selfrot_deg),
    )
    partition = jnp.asarray(horizon_partition_weights(nside))

    def one(alm: jnp.ndarray) -> jnp.ndarray:
        beam_h = alm2map(rotate_alm(alm, psi, theta, phi, lmax=lmax),
                         nside=nside, lmax=lmax)
        # HEALPix pixels are equal-area, so a plain sum IS the integral.
        return jnp.sum(beam_h * partition) / jnp.sum(beam_h)

    flat = beam_alm.reshape(-1, beam_alm.shape[-1])
    return jax.vmap(one)(flat).reshape(beam_alm.shape[:-1])


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
    npol: int | None = None,
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

    POLARISATION. Supported via ``npol``, but note that this is the ONE part
    of the drift-scan path where polarisation is not free: the rotation and
    the dot are spin-independent and therefore row-wise, whereas masking has
    to leave harmonic space, which means a genuine spin-2 synthesis of (Q, U)
    from (E, B) and a spin-2 analysis back. Two consequences worth knowing
    before switching it on, both detailed in :mod:`limtod_jax.hpx`:

    * The spin-2 transforms need ``nside >= 2``, ``lmax + 1 >= 2*nside``, and
      a DENSE precompute kernel of O(nside·lmax²) — cheap up to nside ~32
      (16 MB) but ~1 GB at nside 128 / lmax 255. It is cached and this is a
      one-off beam preparation, but it does cap the usable resolution.
    * At a ZENITH pointing, prefer :func:`horizon_truncated_beam`: also fully
      polarised, exact rather than band-limited, ~8x cheaper, and it needs no
      spin-2 machinery at all (the taper is a real scalar).
    """
    npol = validate_npol(npol)
    validate_pol_alm("beam_alm", beam_alm, lmax, npol)
    psi, theta, phi = zyzyz2zyz(
        0.0,
        0.0,
        -jnp.asarray(az_deg),
        jnp.asarray(el_deg) - 90.0,
        jnp.asarray(selfrot_deg),
    )
    beam_h = rotate_alm(beam_alm, psi, theta, phi, lmax=lmax)
    map_h = alm2map(beam_h, nside=nside, lmax=lmax, npol=npol)
    weights = jnp.asarray(horizon_weights(nside, apod_deg))
    masked_alm = map2alm_iter(
        map_h * weights, nside=nside, lmax=lmax, iterations=iterations, npol=npol
    )
    # Inverse rotation: (psi, theta, phi) -> (-phi, -theta, -psi), locked
    # numerically (roundtrip test in the suite).
    return rotate_alm(masked_alm, -phi, -theta, -psi, lmax=lmax)


class DriftScanMmode(eqx.Module):
    """Drift-scan m-mode operator: sky alms -> TOD, one beam rotation total.

    An :class:`equinox.Module` (frozen pytree) bundling the reference-frame
    beam with the LST sampling, ready for jit/vmap/grad and for use as a
    field of larger models. Build it with
    :meth:`from_pointing` (friendly constructor: beam-local alms +
    pointing) or directly from a precomputed ``beam_ref_alm``.

    Attributes:
        beam_ref_alm: ``(n_alm,)`` — or ``(npol, n_alm)`` when polarised —
            packed celestial-frame beam alms at the reference LST (traced —
            differentiable beam).
        dphi: ``(n_time,)`` LST offsets from the reference, RADIANS (traced).
        lmax: static band-limit matching ``beam_ref_alm``.
        normalize: static; numpy limTOD ``normalize_beam`` semantics.
        ones_alm: quadrature alms of the ones map; required iff ``normalize``.
        uniform_sampling: static; assert that ``dphi`` is a uniform grid over
            a full sidereal turn and use the FFT synthesis
            (see :func:`driftscan_tod`). Validated at construction whenever
            ``dphi`` is concrete.
        npol: static 1/3/4 for a polarised beam (packed T/E/B/V rows), or
            ``None`` for Stokes I. The TOD stays ``(n_time,)`` either way —
            the Stokes rows contract into each sample.
    """

    beam_ref_alm: jax.Array
    dphi: jax.Array
    lmax: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True, default=False)
    ones_alm: jax.Array | None = None
    uniform_sampling: bool = eqx.field(static=True, default=False)
    npol: int | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        validate_npol(self.npol)
        want_ndim = 1 if self.npol is None else 2
        if self.beam_ref_alm.ndim != want_ndim:
            raise ValueError(
                f"beam_ref_alm must be {want_ndim}D "
                f"({'(n_alm,)' if self.npol is None else f'({self.npol}, n_alm)'})"
                f" — the operator is single-beam; batch frequencies by vmapping "
                f"the constructor or the call — got shape "
                f"{self.beam_ref_alm.shape}"
            )
        validate_pol_alm("beam_ref_alm", self.beam_ref_alm, self.lmax, self.npol)
        _validate_dphi(self.dphi)
        if self.normalize and self.ones_alm is None:
            raise ValueError(
                "normalize=True requires ones_alm "
                "(limtod_jax.hpx.ones_quadrature_alm)"
            )
        if self.ones_alm is not None:
            # Checked HERE and not only on __call__: a Stokes stack in this
            # slot broadcasts rather than failing, so the operator would build
            # cleanly and only surface a wrong-shaped TOD much later.
            validate_unpolarised("ones_alm", self.ones_alm)
            validate_pol_alm("ones_alm", self.ones_alm, self.lmax, None)
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
        npol: int | None = None,
    ) -> "DriftScanMmode":
        """Build the operator from beam-local alms and drift-scan pointing.

        Args:
            beam_alm: ``(n_alm,)`` packed beam alms in the beam-local frame
                (as ``hp.map2alm(beam_map)`` computes them in numpy limTOD),
                or ``(npol, n_alm)`` packed T/E/B/V rows for a polarised beam.
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
            npol: 1/3/4, or ``None`` (default) to INFER it — 1-D
                ``beam_alm`` means Stokes I, 2-D means a Stokes stack whose
                row count must be 1, 3 or 4. Inference is safe here (and only
                here) because 2-D ``beam_alm`` was previously a hard error on
                this operator, so no existing shape changes meaning; the
                functional layer never infers. Pass it explicitly if you
                would rather the shape be checked than read.
        """
        lst_deg = jnp.asarray(lst_deg)
        beam_alm = jnp.asarray(beam_alm)
        if lst_deg.ndim != 1:
            raise ValueError(f"lst_deg must be 1D (n_time,), got {lst_deg.shape}")
        if npol is None and beam_alm.ndim == 2:
            npol = beam_alm.shape[0]
            if npol not in STOKES_ALM_ROWS:
                raise ValueError(
                    f"2-D beam_alm is read as a Stokes stack (npol, n_alm), so "
                    f"its first axis must be 1, 3 or 4 rows — got {npol}. A "
                    f"FREQUENCY axis belongs in jax.vmap over this "
                    f"constructor, not in beam_alm, or the rows would be "
                    f"summed into one TOD."
                )
        npol = validate_npol(npol)
        if beam_alm.ndim > 2:
            raise ValueError(
                f"beam_alm must be (n_alm,) or (npol, n_alm); batch anything "
                f"further with jax.vmap — got shape {beam_alm.shape}"
            )
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
                npol=npol,
            )
        if lst_ref_deg is None:
            lst_ref_deg = lst_deg[0]
        beam_ref = beam_alm_at_reference(
            beam_alm, lst_ref_deg, lat_deg, az_deg, el_deg, selfrot_deg,
            lmax=lmax, npol=npol,
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
            npol=npol,
        )

    def mmodes(self, sky_alm: jnp.ndarray) -> jnp.ndarray:
        """m-modes ``Ṽ_m`` (complex, ``(lmax+1,)``) of the drift-scan TOD.

        Raises:
            ValueError: if the operator is ``normalize=True``. The
                normalization divides the TOD by the rotated beam's solid
                angle, and that division is NOT diagonal in ``m`` — so no
                ``(lmax+1,)`` array is the m-mode content of what
                :meth:`__call__` returns, and the projection this method
                computes would disagree with it by that solid angle (a
                factor of ~33 for a 20° beam at nside 16, not a small
                correction). See the error message for the two ways out.
        """
        if self.normalize:
            raise ValueError(
                "mmodes() is undefined for a normalize=True operator: the "
                "m-mode projection is the beam-weighted INTEGRAL, while "
                "__call__ returns it divided by the rotated beam's solid "
                "angle (~33x at nside 16 / lmax 47 / 20 deg FWHM). The "
                "division is by a t-dependent denominator, so it is not "
                "diagonal in m and no (lmax+1,) array is exactly the "
                "m-mode content of the normalized TOD.\n"
                "  - m-modes of the TOD the operator returns (uniform "
                "full-turn grid): mmodes_from_tod_uniform(op(sky_alm), "
                "lmax=op.lmax, phase0=op.dphi[0])\n"
                "  - the un-normalized projection, if that is what you "
                "wanted: mmodes_from_sky(op.beam_ref_alm, sky_alm, "
                "lmax=op.lmax, npol=op.npol)"
            )
        return mmodes_from_sky(
            self.beam_ref_alm, sky_alm, lmax=self.lmax, npol=self.npol
        )

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
            npol=self.npol,
        )

    def adjoint(self, tod: jnp.ndarray) -> jnp.ndarray:
        """Exact sky-slot transpose of :meth:`__call__` (packed alms out,
        ``(npol, n_alm)`` when polarised)."""
        return driftscan_tod_adjoint(
            tod,
            self.beam_ref_alm,
            self.dphi,
            lmax=self.lmax,
            normalize=self.normalize,
            ones_alm=self.ones_alm,
            uniform=self.uniform_sampling,
            npol=self.npol,
        )
