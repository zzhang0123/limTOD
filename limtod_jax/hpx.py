"""HEALPix (RING) map <-> alm inside JAX, via s2fft — healpy never enters.

Every operation here is linear and jit/vmap/grad-safe:

* :func:`alm2map` — exact synthesis of a bandlimited function at pixel
  centers (matches ``hp.alm2map`` to roundoff).
* :func:`map2alm_quad` — QUADRATURE analysis
  ``S̃_lm = Σ_p m(p)·Y*_lm(p) = (npix/4π)·hp.map2alm(m, iter=0)``.
  This is deliberately NOT an approximation of the true alms: paired with
  :func:`limtod_jax.alm.alm_dot` it turns HEALPix pixel dots into exact
  harmonic identities (see :mod:`limtod_jax.core`), which is what the
  TOD forward model and its adjoint are built on.
* :func:`map2alm_iter` — healpy-style iterative analysis (true alms).
* :func:`ones_quadrature_alm` — the exact pixel-sum functional.
* :func:`eb_to_qu` / :func:`qu_to_eb_quad` — the spin-2 pair, used by the
  polarised paths; see the warning below before touching them.

``method="jax"`` equivalents only — s2fft's ``jax_healpy`` backend routes
through healpy and is banned from this package.

POLARISATION. Pass ``npol`` (1/3/4) to work on Stokes stacks: maps
``(npol, npix)`` in limTOD's ``I,Q,U[,V]`` layout, alms ``(npol, n_alm)`` in
the packed ``T,E,B[,V]`` layout healpy's own 3-row transforms use. ``T`` and
``V`` stay spin-0; ``(Q, U)`` <-> ``(E, B)`` is spin-2 and uses a DIFFERENT
s2fft backend from everything else here — see the warning below, it is not a
stylistic choice.

.. warning::

   **s2fft's on-the-fly (Price-McEwen) recursion is WRONG at spin != 0 on the
   HEALPix grid**, so ``s2fft.inverse_jax`` / ``forward_jax`` — which this
   module uses at spin 0 — must NOT be used at spin 2. The recursion
   renormalises with ``log|d^l_{mn}|`` and ``1/|d^l_{mn}|`` and accumulates
   with ``nansum``; wherever a Wigner-d is EXACTLY zero at a ring colatitude
   this gives ``log 0 = -inf`` and ``0*inf = nan``, and the whole l term is
   silently dropped. HEALPix rings sit at rational ``cos(theta)``
   (``1 - t²/3nside²`` in the caps, ``4/3 - 2t/3nside`` in the belt), so those
   exact zeros really occur — at ``cos(theta) = ±2/l``, killing l = 3, 6, 8,
   16, 32, 48 ... depending on nside. MW/GL/DH sampling never triggers it, and
   neither does spin 0, which is why the rest of this module is fine.

   Measured cost of getting this wrong, on a random band-limited field:
   **1.5-5 % RMS-relative, 4.5-20 % in max-norm** (worst at low nside),
   concentrated entirely on the affected multipoles — and a silent
   1e-3..1e-2 break of the pixel-dot == alm-dot exactness contract. The
   spin-2 path here therefore goes through the PRECOMPUTE transforms with
   ``recursion="risbo"``, which reproduces healpy to ~1e-14;
   ``recursion="auto"`` routes back to the broken code and is equally wrong.
   Cross-checked against healpy and against an independent brute-force
   Wigner-d evaluator.

   Price of the fix: the precompute kernel is a dense
   ``(4·nside-1, L, 2L-1)`` complex128 array, i.e. O(nside·lmax²) memory —
   0.5 MB at nside 8 / lmax 23, 16 MB at nside 32 / lmax 63, but ~1 GB at
   nside 128 / lmax 255 and ~8.6 GB at nside 256 / lmax 511. It is cached per
   ``(L, spin, nside, forward)`` (as NUMPY — see :func:`_spin_kernel` for why
   that detail is load-bearing), and polarised masking is a one-off beam
   preparation, so this is usually paid once. It does put a practical ceiling
   on the resolution at which a polarised beam can be masked inside JAX, and
   the cache is deliberately small: there is no exposed eviction, so a few
   high-resolution kernels would pin memory for the life of the process.

PRECISION: s2fft's healpix transforms need float64 — in a float32 session
(``jax_enable_x64`` off) they carry O(10%) errors even at L ~ 12 (s2fft warns
about exactly this at import). Enable x64 wherever these functions feed real
analysis. The Wigner rotation path (:mod:`limtod_jax.wigner`, Risbo
recursion) does NOT share this problem — it stays at roundoff accuracy in
float32.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import s2fft

from limtod_jax.alm import packed_from_2d, packed_to_2d
from limtod_jax.stokes import validate_npol, validate_stokes_axis


def _check_spin2_grid(nside: int, lmax: int) -> None:
    """s2fft's HEALPix spin transforms need ``nside >= 2`` and ``L >= 2·nside``.

    Both limits are hard and both fail OBSCURELY inside s2fft (a bare
    ``AssertionError``, or ``"Need at least one array to stack"``), so they are
    stated here instead. Only the spin-2 path is affected; spin 0 is not.
    """
    if nside < 2:
        raise ValueError(
            f"the polarised (spin-2) HEALPix transforms need nside >= 2, got "
            f"{nside} (s2fft fails with an opaque stacking error below that)."
        )
    if lmax + 1 < 2 * nside:
        raise ValueError(
            f"the polarised (spin-2) HEALPix transforms need lmax+1 >= 2*nside, "
            f"got lmax={lmax}, nside={nside} (needs lmax >= {2 * nside - 1}). "
            f"Raise lmax or lower nside; s2fft asserts this internally without "
            f"a message."
        )


@lru_cache(maxsize=4)
def _spin_kernel(L: int, spin: int, nside: int, forward: bool) -> np.ndarray:
    """Cached s2fft precompute kernel — ``recursion="risbo"``, never "auto".

    ``"auto"``/``"price-mcewen"`` on the builder route back to the on-the-fly
    recursion this module exists to avoid (see the module warning); ``"risbo"``
    is the one that reproduces healpy. The forward kernel already folds in the
    4π/npix HEALPix quadrature weight. The key is complete: ``reality``,
    ``sampling`` and ``recursion`` are literals here, so a cache hit can never
    return a kernel built for a different configuration.

    RETURNS NUMPY, DELIBERATELY, and the ``_jax`` builder is not used. That
    builder is made of ``jnp`` ops, so the FIRST call inside a ``jit`` trace
    returns a tracer — which ``lru_cache`` would then store process-globally,
    making every later call outside that trace die with
    ``UnexpectedTracerError``, permanently and order-dependently (warm the
    cache eagerly and it works; call it under ``jit`` first and the session is
    poisoned). The kernel is a pure function of static ints, so a numpy array
    is the honest representation: it is concrete by construction, and
    ``jnp.asarray`` at the use site stages it as the compile-time constant it
    always was.

    Memory is the reason for the small ``maxsize``: each kernel is
    ``(4·nside−1, L, 2L−1)`` complex128 — 0.5 MB at nside 8 / lmax 23 but
    ~1 GB at nside 128 / lmax 255, and a polarised analysis needs three of
    them (spin +2 inverse, spin ±2 forward). There is no exposed eviction, so
    holding many at high resolution would pin memory for the life of the
    process.
    """
    from s2fft.precompute_transforms.construct import spin_spherical_kernel

    return np.asarray(
        spin_spherical_kernel(
            L, spin=spin, reality=False, sampling="healpix", nside=nside,
            forward=forward, recursion="risbo",
        )
    )


def _kernel(L: int, spin: int, nside: int, forward: bool) -> jnp.ndarray:
    """:func:`_spin_kernel` as a jax array (a constant under ``jit``)."""
    return jnp.asarray(_spin_kernel(L, spin, nside, forward))


def eb_to_qu(
    e_alm: jnp.ndarray, b_alm: jnp.ndarray, *, nside: int, lmax: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Packed ``(E, B)`` alms -> ``(Q, U)`` HEALPix RING maps.

    healpy's convention, locked numerically (every sign alternative loses by
    >= 1.3 relative):  ``Q ± iU = −Σ_lm (E_lm ± i·B_lm)·(±2)Y_lm``. Since Q
    and U are real, ``P_- = conj(P_+)`` exactly (verified to 2e-15), so ONE
    spin-2 synthesis suffices: ``Q = Re P_+``, ``U = Im P_+``.

    Note the dense expansion happens on E and B SEPARATELY: each is a
    real-field alm array and obeys ``a_{l,-m} = (−1)^m conj(a_lm)``, but their
    combination ``−(E + iB)`` — the spin-2 coefficient — does not. Combining
    first and symmetrizing after is a subtle, plausible-looking error worth
    ~90 % (it was made and caught during development).
    """
    _check_spin2_grid(nside, lmax)
    from s2fft.precompute_transforms.spherical import inverse_transform_jax

    L = lmax + 1
    a_p2 = -(packed_to_2d(e_alm, lmax) + 1j * packed_to_2d(b_alm, lmax))
    p = inverse_transform_jax(
        a_p2, _kernel(L, 2, nside, False), L, "healpix", False, 2, nside
    )
    return jnp.real(p), jnp.imag(p)


def qu_to_eb_quad(
    q_map: jnp.ndarray, u_map: jnp.ndarray, *, nside: int, lmax: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(Q, U)`` maps -> packed QUADRATURE ``(E, B)`` alms.

    The spin-2 counterpart of :func:`map2alm_quad`, with the SAME ``npix/4π``
    factor — which is what keeps the pixel-dot == alm-dot exactness contract
    intact at spin 2 (``Σ_p (Q_b Q_s + U_b U_s) = ⟨E_b, Ẽ_s⟩ + ⟨B_b, B̃_s⟩``).
    Equals ``(npix/4π)·hp.map2alm([0,Q,U], lmax, iter=0)[1:]``.

    Both spin branches are computed here (unlike the synthesis): the analysis
    of an arbitrary, non-band-limited map has no reality shortcut to exploit,
    and it is a one-off cost.
    """
    _check_spin2_grid(nside, lmax)
    from s2fft.precompute_transforms.spherical import forward_transform_jax

    L = lmax + 1
    scale = (12 * nside * nside) / (4.0 * np.pi)
    p = q_map + 1j * u_map
    g_p2 = forward_transform_jax(
        p, _kernel(L, 2, nside, True), L, "healpix", False, 2, nside
    )
    g_m2 = forward_transform_jax(
        jnp.conj(p), _kernel(L, -2, nside, True), L, "healpix", False, -2, nside
    )
    e2d = -(g_p2 + g_m2) / 2.0
    b2d = 1j * (g_p2 - g_m2) / 2.0
    return (
        scale * packed_from_2d(e2d, lmax),
        scale * packed_from_2d(b2d, lmax),
    )


def _scalar_alm2map(alm: jnp.ndarray, nside: int, lmax: int) -> jnp.ndarray:
    flm = packed_to_2d(alm, lmax)
    # s2fft returns a complex array with ~0 imaginary part even for
    # reality=True; the synthesis of a real field is real by construction.
    return jnp.real(
        s2fft.inverse_jax(
            flm, lmax + 1, spin=0, nside=nside, sampling="healpix", reality=True
        )
    )


def _scalar_map2alm_quad(m: jnp.ndarray, nside: int, lmax: int) -> jnp.ndarray:
    flm = s2fft.forward_jax(
        m, lmax + 1, spin=0, nside=nside, sampling="healpix", reality=True
    )
    npix = 12 * nside * nside
    return (npix / (4.0 * np.pi)) * packed_from_2d(flm, lmax)


def alm2map(
    alm: jnp.ndarray, *, nside: int, lmax: int, npol: int | None = None
) -> jnp.ndarray:
    """Packed real-field alms -> HEALPix RING map ``(12·nside²,)``.

    With ``npol`` set, takes ``(npol, n_alm)`` packed ``T,E,B[,V]`` rows and
    returns ``(npol, npix)`` Stokes ``I,Q,U[,V]`` maps — matching
    ``hp.alm2map`` on a 3-row alm array, with V synthesized as the spin-0
    field it is rather than riding along in the spin-2 transform.
    """
    npol = validate_npol(npol)
    if npol is None:
        return _scalar_alm2map(alm, nside, lmax)
    validate_stokes_axis("alm", alm, npol)
    rows = [_scalar_alm2map(alm[0], nside, lmax)]
    if npol >= 3:
        rows.extend(eb_to_qu(alm[1], alm[2], nside=nside, lmax=lmax))
    if npol == 4:
        rows.append(_scalar_alm2map(alm[3], nside, lmax))
    return jnp.stack(rows)


def map2alm_quad(
    m: jnp.ndarray, *, nside: int, lmax: int, npol: int | None = None
) -> jnp.ndarray:
    """HEALPix RING map -> packed quadrature alms ``Σ_p m(p)·Y*_lm(p)``.

    With ``npol`` set, takes ``(npol, npix)`` Stokes ``I,Q,U[,V]`` maps and
    returns ``(npol, n_alm)`` packed ``T,E,B[,V]`` quadrature alms.
    """
    npol = validate_npol(npol)
    if npol is None:
        return _scalar_map2alm_quad(m, nside, lmax)
    validate_stokes_axis("m", m, npol)
    rows = [_scalar_map2alm_quad(m[0], nside, lmax)]
    if npol >= 3:
        rows.extend(qu_to_eb_quad(m[1], m[2], nside=nside, lmax=lmax))
    if npol == 4:
        rows.append(_scalar_map2alm_quad(m[3], nside, lmax))
    return jnp.stack(rows)


def map2alm_iter(
    m: jnp.ndarray, *, nside: int, lmax: int, iterations: int = 3,
    npol: int | None = None,
) -> jnp.ndarray:
    """healpy-style iterative analysis: matches ``hp.map2alm(m, lmax, iter=k)``.

    Jacobi refinement of the quadrature estimate: starting from
    ``a = (4π/npix)·Σ_p m(p)·Y*_lm(p)``, repeat ``iterations`` times

        ``a ← a + (4π/npix)·analysis(m − synthesis(a))``

    which is exactly healpy's ``iter`` scheme (oracle-locked in
    ``tests/limtod_jax/test_driftscan.py``). Unlike :func:`map2alm_quad`
    this RETURNS TRUE alms of the (non-bandlimited) map's best bandlimited
    representation — use it when a map-space operation (e.g. a horizon
    mask) must be carried back to harmonic space the same way numpy
    limTOD's internal ``hp.map2alm(..., iter=3)`` would.

    ``iterations`` is static; the whole function is linear in ``m`` and
    jit/vmap/grad-safe.

    With ``npol`` set the SAME loop runs on the Stokes stack, reproducing
    ``hp.map2alm([I,Q,U], lmax, iter=k)``. T decouples from (E, B) — the
    refinement never mixes them — so this is the scalar scheme with a wider
    residual, not a new algorithm.
    """
    npol = validate_npol(npol)
    npix = 12 * nside * nside
    scale = (4.0 * np.pi) / npix
    alm = scale * map2alm_quad(m, nside=nside, lmax=lmax, npol=npol)
    for _ in range(iterations):
        residual = m - alm2map(alm, nside=nside, lmax=lmax, npol=npol)
        alm = alm + scale * map2alm_quad(
            residual, nside=nside, lmax=lmax, npol=npol
        )
    return alm


def ones_quadrature_alm(*, nside: int, lmax: int) -> jnp.ndarray:
    """Quadrature alms of the ones map — the exact pixel-sum functional.

    ``alm_dot(x, ones_quadrature_alm(...)) == Σ_p alm2map(x)[p]`` for any
    bandlimited ``x``; this is the ``normalize`` denominator in
    :func:`limtod_jax.core.generate_tod_sky` (numpy limTOD's
    ``normalize_beam`` divides by the rotated beam's pixel sum).
    """
    return map2alm_quad(jnp.ones(12 * nside * nside), nside=nside, lmax=lmax)
