"""Rotation-convention lock: the frozen mapping vs healpy, numerically.

The port contract forbids trusting Euler sign/order conventions on paper.
Ground truth is limTOD's exact call pattern
(``_rotate_healpix_map(alm, psi, theta, phi)`` ->
``hp.rotate_alm(alm, phi, theta, psi)`` — its phi in healpy's first slot).

The original lock probe ran 8 candidate mappings (angle order/sign x
m-reversal) over random alms and angle triples including theta ~ 0 and
theta ~ pi; the UNIQUE winner at rel err < 1e-11 was the identity mapping
``(alpha, beta, gamma) = (psi, theta, phi)`` (worst rel err 1.8e-15, all
others O(1)). This file keeps both directions as regressions: the frozen
mapping still matches healpy, and the rejected candidates still do NOT
(guarding against a silent convention change in future s2fft releases).
"""

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from limtod_jax import wigner
from limtod_jax.alm import packed_from_2d, packed_to_2d

# Generic, near-theta=0, near-theta=pi, and angles outside [0, pi].
ANGLE_TRIPLES = [
    (0.7, 1.2, 0.3),
    (-2.1, 2.9, 4.0),
    (0.3, 0.02, -0.5),
    (1.0, np.pi - 0.01, 2.0),
]


def _healpy_reference(alm, psi, theta, phi):
    """The rotation limTOD applies for its (psi, theta, phi)."""
    ref = alm.copy()
    hp.rotate_alm(ref, phi, theta, psi)
    return ref


def _rotate_via_wigner(alm, lmax, alpha, beta, gamma, flip_m=False):
    flm = packed_to_2d(jnp.asarray(alm), lmax)
    if flip_m:
        flm = flm[..., ::-1]
    out = wigner.rotate_flm_2d(
        flm, lmax + 1, jnp.asarray(alpha), jnp.asarray(beta), jnp.asarray(gamma)
    )
    if flip_m:
        out = out[..., ::-1]
    return np.asarray(packed_from_2d(out, lmax))


@pytest.mark.parametrize("lmax,nside", [(7, 8), (23, 8), (47, 16)])
@pytest.mark.parametrize("angles", ANGLE_TRIPLES)
def test_frozen_mapping_matches_healpy(lmax, nside, angles, rng):
    psi, theta, phi = angles
    alm = hp.map2alm(rng.standard_normal(hp.nside2npix(nside)), lmax=lmax)
    ref = _healpy_reference(alm, psi, theta, phi)
    a, b, g = wigner.angles_to_alpha_beta_gamma(
        jnp.asarray(psi), jnp.asarray(theta), jnp.asarray(phi)
    )
    got = _rotate_via_wigner(alm, lmax, a, b, g)
    scale = np.max(np.abs(alm))
    assert np.max(np.abs(got - ref)) / scale < 1e-11
    assert wigner.HEALPY_CONVENTION == "identity"


@pytest.mark.parametrize(
    "candidate",
    [
        ("swap", False),
        ("neg", False),
        ("swap-neg", False),
        ("identity", True),  # identity + m-flip is NOT the convention either
    ],
)
def test_rejected_candidates_still_mismatch(candidate, rng):
    """If one of these ever starts matching, the kernel's convention moved."""
    kind, flip = candidate
    lmax = 7
    alm = hp.map2alm(rng.standard_normal(hp.nside2npix(8)), lmax=lmax)
    scale = np.max(np.abs(alm))
    psi, theta, phi = 0.7, 1.2, 0.3
    ref = _healpy_reference(alm, psi, theta, phi)
    mapping = {
        "identity": (psi, theta, phi),
        "swap": (phi, theta, psi),
        "neg": (-psi, theta, -phi),
        "swap-neg": (-phi, theta, -psi),
    }[kind]
    got = _rotate_via_wigner(alm, lmax, *mapping, flip_m=flip)
    assert np.max(np.abs(got - ref)) / scale > 1e-3


def test_rotation_composes(rng):
    """Two-step z-then-full rotation equals the single call (as healpy does)."""
    lmax = 23
    alm = jnp.asarray(hp.map2alm(rng.standard_normal(hp.nside2npix(8)), lmax=lmax))
    psi, theta, phi = 0.7, 1.2, 0.3
    flm = packed_to_2d(alm, lmax)
    one = wigner.rotate_flm_2d(
        flm, lmax + 1, jnp.asarray(psi), jnp.asarray(theta), jnp.asarray(phi)
    )
    # gamma applied first (innermost): split it off as its own z-rotation.
    step1 = wigner.rotate_flm_2d(
        flm, lmax + 1, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(phi)
    )
    step2 = wigner.rotate_flm_2d(
        step1, lmax + 1, jnp.asarray(psi), jnp.asarray(theta), jnp.asarray(0.0)
    )
    np.testing.assert_allclose(np.asarray(one), np.asarray(step2), atol=1e-13)


def test_zero_rotation_is_identity(rng):
    lmax = 23
    alm = jnp.asarray(hp.map2alm(rng.standard_normal(hp.nside2npix(8)), lmax=lmax))
    flm = packed_to_2d(alm, lmax)
    out = wigner.rotate_flm_2d(
        flm, lmax + 1, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0)
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(flm), atol=1e-14)


def test_precomputed_dl_array_path(rng):
    """rotate with dl_array precomputed == streaming path."""
    lmax = 15
    L = lmax + 1
    alm = jnp.asarray(hp.map2alm(rng.standard_normal(hp.nside2npix(8)), lmax=lmax))
    flm = packed_to_2d(alm, lmax)
    a, b, g = (jnp.asarray(x) for x in (0.7, 1.2, 0.3))
    stream = wigner.rotate_flm_2d(flm, L, a, b, g)
    dls = wigner.generate_rotate_dls(L, b)
    pre = wigner.rotate_flm_2d(flm, L, a, b, g, dl_array=dls)
    np.testing.assert_allclose(np.asarray(pre), np.asarray(stream), atol=1e-14)
