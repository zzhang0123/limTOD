"""The polarisation contract: one leading Stokes axis, contracted at the dot.

limTOD's Stokes convention is shape-based and lives in MAP space: a 1D map is
Stokes I, a 3-row map is ``[I, Q, U]``, a 4-row map is ``[I, Q, U, V]``
(``limTOD.simulator.generate_TOD_sky``). What crosses into harmonic space —
and therefore what every function in :mod:`limtod_jax` sees — is the healpy
ANALYSIS of those rows:

===========  =======================  ==================================
``npol``     map rows                 packed alm rows (what you pass here)
===========  =======================  ==================================
1            ``I``                    ``T``
3            ``I, Q, U``              ``T, E, B``
4            ``I, Q, U, V``           ``T, E, B, V``
===========  =======================  ==================================

i.e. exactly ``hp.map2alm(beam_map)`` for 1/3 rows, and for 4 rows
``vstack(hp.map2alm(map[:3]), hp.map2alm(map[3]))`` — the same split numpy
limTOD performs internally, because V is a spin-0 field that must not ride
along in the spin-2 transform.

WHY THE WHOLE LINEAR CHAIN IS THEN THE SCALAR CHAIN, ROW BY ROW.
Three facts, each locked numerically in
``tests/limtod_jax/test_polarisation.py`` rather than trusted on paper:

1. Spin-weighted alms rotate with the SAME Wigner-D as scalar ones — the spin
   index never enters the m-mixing — so healpy's 3-row ``rotate_alm`` is
   BIT-IDENTICAL to the scalar rotation applied to each of T, E, B. E and B
   do not mix under rotation (they are a parity split, not a rotational one).
2. Spin-2 harmonics are orthonormal, so the pixel dot splits row-wise,
   ``Σ_p (Q_b Q_s + U_b U_s) = Σ_lm [conj(E_b)E_s + conj(B_b)B_s]``; the E-B
   cross terms are purely imaginary under the real-field symmetry and drop.
   The quadrature-alm exactness contract of :mod:`limtod_jax.core` therefore
   carries over unchanged, per row.
3. A rotation about z contributes ``D^l_{m'm}(α,0,0) = δ_{m'm}·e^{−imα}``
   INDEPENDENTLY OF SPIN, so every row picks up the same drift-scan phase and
   the m-mode collapse of :mod:`limtod_jax.driftscan` needs no new algebra —
   only an extra sum over rows.

Consequently the TOD is ``V_t = Σ_row Σ_lm conj(B_row,lm(t))·S̃_row,lm``: the
Stokes axis is CONTRACTED, not batched. This is numpy limTOD's
``np.sum(beam_map * sky_map)`` — a total-detected-power convention with unit
Mueller response and no leakage terms.

OPT-IN, NEVER INFERRED. ``npol`` is a STATIC argument, defaulting to ``None``
= unpolarised, in which case all leading axes remain BATCH axes exactly as
before. It is not inferred from the array shape at the functional layer,
because a Stokes axis (contracted) and a frequency axis (passed through) are
both leading axes and shape alone cannot tell them apart — inferring would
turn a 3-frequency stack into a silently wrong Stokes contraction. The
friendly constructor :meth:`limtod_jax.driftscan.DriftScanMmode.from_pointing`
does infer it, but only because 2-D input was previously a hard error there,
so there is no ambiguity to resolve.
"""

from __future__ import annotations

import jax.numpy as jnp

from limtod_jax.alm import nalm_of_lmax

#: Stokes row counts limTOD accepts, and the alm rows each implies.
STOKES_ALM_ROWS: dict[int, tuple[str, ...]] = {
    1: ("T",),
    3: ("T", "E", "B"),
    4: ("T", "E", "B", "V"),
}

#: Map-space Stokes rows each ``npol`` implies (documentation/error messages).
STOKES_MAP_ROWS: dict[int, tuple[str, ...]] = {
    1: ("I",),
    3: ("I", "Q", "U"),
    4: ("I", "Q", "U", "V"),
}


def validate_npol(npol: int | None) -> int | None:
    """Return ``npol`` unchanged, or raise if it is not 1, 3 or 4.

    The {1, 3, 4} restriction is limTOD's own (``I`` / ``IQU`` / ``IQUV``);
    it is enforced rather than ignored because it is the only structural
    guard against a frequency stack being mistaken for a Stokes stack.
    """
    if npol is None:
        return None
    if not isinstance(npol, (int, jnp.integer)) or isinstance(npol, bool):
        raise TypeError(
            f"npol must be a static Python int (1, 3 or 4) or None, got "
            f"{npol!r} of type {type(npol).__name__}"
        )
    npol = int(npol)
    if npol not in STOKES_ALM_ROWS:
        raise ValueError(
            f"npol must be 1 (I), 3 (I,Q,U) or 4 (I,Q,U,V), got {npol}. "
            "A FREQUENCY axis is not a Stokes axis — batch frequencies with "
            "jax.vmap, never as a leading array axis, or they would be summed "
            "into a single sample."
        )
    return npol


def validate_pol_alm(
    name: str, alm: jnp.ndarray, lmax: int, npol: int | None
) -> None:
    """Check ``alm`` against ``lmax`` and, when polarised, the Stokes axis.

    Unpolarised (``npol is None``): only the trailing packed length is
    constrained; leading axes are the caller's batch axes.
    Polarised: the array must be ``(..., npol, n_alm)``.
    """
    n_alm = nalm_of_lmax(lmax)
    if alm.shape[-1] != n_alm:
        raise ValueError(
            f"{name} packed length {alm.shape[-1]} does not match lmax={lmax} "
            f"(expected {n_alm})"
        )
    if npol is None:
        return
    if alm.ndim < 2 or alm.shape[-2] != npol:
        raise ValueError(
            f"{name} must have shape (..., npol, n_alm) = (..., {npol}, "
            f"{n_alm}) for npol={npol} — packed alm rows "
            f"{STOKES_ALM_ROWS[npol]}, i.e. the healpy analysis of Stokes "
            f"{STOKES_MAP_ROWS[npol]} — got {alm.shape}"
        )


def validate_stokes_axis(name: str, arr: jnp.ndarray, npol: int | None) -> None:
    """Check the Stokes axis alone, for callers with no ``lmax`` to check.

    Checks against ``npol`` ITSELF, not merely that two arrays agree with each
    other: a pair of 5-row frequency stacks agrees with itself perfectly and
    would otherwise be summed into one plausible, finite, wrong number — the
    exact failure the {1, 3, 4} restriction exists to prevent.
    """
    if npol is None:
        return
    if arr.ndim < 2 or arr.shape[-2] != npol:
        got = "1-D" if arr.ndim < 2 else f"{arr.shape[-2]} rows"
        raise ValueError(
            f"{name} must have a leading Stokes axis of {npol} rows "
            f"{STOKES_ALM_ROWS[npol]} for npol={npol}, got {got} "
            f"(shape {arr.shape}). A FREQUENCY axis is not a Stokes axis — "
            f"batch frequencies with jax.vmap, or they would be summed into a "
            f"single sample."
        )


def match_npol(name_a: str, a: jnp.ndarray, name_b: str, b: jnp.ndarray,
               npol: int | None) -> None:
    """Raise unless both arrays carry the ``npol``-row Stokes axis."""
    if npol is None:
        return
    validate_stokes_axis(name_a, a, npol)
    validate_stokes_axis(name_b, b, npol)


def validate_unpolarised(name: str, arr: jnp.ndarray) -> None:
    """Raise unless ``arr`` is a single packed row.

    For arguments that are ALWAYS unpolarised whatever ``npol`` is — notably
    ``ones_alm``, since the ``normalize`` denominator is the Stokes-I beam's
    pixel sum. Passing a Stokes stack there is not an error JAX would catch:
    it broadcasts, and the TOD silently acquires an extra axis.
    """
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (n_alm,) — it is always a single unpolarised "
            f"row, because the normalize denominator is the STOKES-I beam's "
            f"pixel sum applied to every row (numpy limTOD's "
            f"pointing_beam_in_eq_sys does the same). Got shape {arr.shape}; "
            f"a Stokes stack here would broadcast and silently add an axis to "
            f"the TOD."
        )


def stokes_i(alm: jnp.ndarray, npol: int | None) -> jnp.ndarray:
    """The Stokes-I (``T``) row of a possibly-polarised alm array.

    Used by the ``normalize`` denominator, which is the rotated **Stokes-I**
    beam's pixel sum for every row — numpy limTOD's
    ``pointing_beam_in_eq_sys`` divides Q/U/V by ``np.sum(beam_pointed[0])``,
    not by their own sums (and must, or a beam with zero net Q would blow up).
    """
    return alm if npol is None else alm[..., 0, :]
