"""Acceptance test 1 — oracle equivalence against numpy limTOD.

``generate_tod_sky`` must match ``limTOD.simulator.generate_TOD_sky`` to
rel err < 1e-6 in float64, on a pointing grid that INCLUDES the extreme
corners (zenith el=90, low elevation, lat in {53.24, 0, -90},
LST in {0, 179.9, 359.9}) — failure modes concentrate at boundaries.

Exactness recipe (why 1e-6 is beatable — expected ~1e-12): feed the native
chain the SAME beam alms the oracle computes internally
(``hp.map2alm(beam_map)``, iter=3) and QUADRATURE sky alms
``(npix/4π)·map2alm(sky, iter=0)``. Since the rotated beam is exactly
bandlimited, the pixel dot ``Σ_p B_rot·s`` equals the weighted harmonic
dot exactly; the only differences left are the Wigner kernels. The oracle
runs with ``truncate_frac_thres=0.0`` (the native port is the linear
chain; truncation is a nonlinear cleanup outside the port's scope).
"""

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from limtod_jax.angles import zyz_of_pointing
from limtod_jax.core import generate_tod_sky, rotate_alm

LATS = [53.24, 0.0, -90.0]
LSTS = [0.0, 179.9, 359.9]
AZELS = [(0.0, 90.0), (123.4, 90.0), (0.0, 5.0), (-42.3, 41.0)]

pytestmark = pytest.mark.filterwarnings("ignore:Gimbal lock detected")


def _corner_pointings():
    """3 LST x 4 (az, el) = 12 pointings, selfrot alternating 0/30 deg."""
    lst, az, el = [], [], []
    for t in LSTS:
        for a, e in AZELS:
            lst.append(t)
            az.append(a)
            el.append(e)
    sr = [0.0 if i % 2 == 0 else 30.0 for i in range(len(lst))]
    return (np.asarray(lst), np.asarray(az), np.asarray(el), np.asarray(sr))


def _zyz_stack(lst, lat, az, el, sr):
    psi, theta, phi = zyz_of_pointing(
        jnp.asarray(lst), lat, jnp.asarray(az), jnp.asarray(el), jnp.asarray(sr)
    )
    return jnp.stack([psi, theta, phi], axis=-1)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("lat", LATS)
def test_oracle_equivalence_nside8_corner_grid(
    lat, normalize, rng, quad_alm, beam_alm_iter3, oracle_tod
):
    nside, lmax = 8, 23
    npix = hp.nside2npix(nside)
    beam_map = rng.random(npix)
    sky_map = rng.random(npix)
    lst, az, el, sr = _corner_pointings()

    direct = oracle_tod(beam_map, sky_map, lst, lat, az, el, sr, normalize_beam=normalize)

    native = generate_tod_sky(
        jnp.asarray(beam_alm_iter3(beam_map, lmax)),
        jnp.asarray(quad_alm(sky_map, lmax)),
        _zyz_stack(lst, lat, az, el, sr),
        lmax=lmax,
        normalize=normalize,
        ones_alm=jnp.asarray(quad_alm(np.ones(npix), lmax)),
    )

    assert native.shape == direct.shape == (12,)
    assert native.dtype == jnp.float64
    rel = np.max(np.abs(np.asarray(native) - direct)) / np.max(np.abs(direct))
    assert rel < 1e-6, f"rel err {rel:.3e}"


def test_oracle_equivalence_nside16_spot(rng, quad_alm, beam_alm_iter3, oracle_tod):
    """Spot check at nside=16 (lmax=47) incl. the lat=-90 zenith gimbal corner."""
    nside, lmax, lat = 16, 47, -90.0
    npix = hp.nside2npix(nside)
    beam_map = rng.random(npix)
    sky_map = rng.random(npix)
    lst = np.asarray([0.0, 179.9, 359.9, 100.0, 200.0, 300.0])
    az = np.asarray([0.0, 123.4, -42.3, 0.0, 80.0, 200.0])
    el = np.asarray([90.0, 90.0, 5.0, 41.0, 60.0, 10.0])
    sr = np.asarray([0.0, 30.0, 0.0, -15.0, 0.0, 7.0])

    direct = oracle_tod(beam_map, sky_map, lst, lat, az, el, sr)
    native = generate_tod_sky(
        jnp.asarray(beam_alm_iter3(beam_map, lmax)),
        jnp.asarray(quad_alm(sky_map, lmax)),
        _zyz_stack(lst, lat, az, el, sr),
        lmax=lmax,
    )
    rel = np.max(np.abs(np.asarray(native) - direct)) / np.max(np.abs(direct))
    assert rel < 1e-6, f"rel err {rel:.3e}"


def test_default_truncation_realism_case(rng, quad_alm, beam_alm_iter3, oracle_tod):
    """Gaussian-beam realism: attribute the ONLY default-args discrepancy.

    Three-way comparison for a wide Gaussian beam (FWHM 60 deg) at nside=8:

    1. native == oracle(truncate=0) to float64 roundoff (measured 8e-16):
       the linear chain is exact for realistic beams too.
    2. oracle(default truncate=1e-10) deviates from the linear chain at the
       ~6.5e-4 level HERE — that is numpy limTOD's own nonlinear cleanup:
       at nside=8, map2alm(iter=3) leaves synthesis ringing of ~1e-3 of the
       beam peak on the far side (measured ±1.2e-5 vs peak 1.4e-2), and
       ``_truncate_map`` zeroes those ~124 (of 768) below-threshold pixels,
       negatives included. The gap shrinks with nside; it is a property of
       the oracle's truncation, not of the port.
    """
    sim = pytest.importorskip("limTOD.simulator")
    nside, lmax, lat = 8, 23, 53.24
    npix = hp.nside2npix(nside)
    theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
    sigma = np.radians(60.0) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    beam_map = np.exp(-0.5 * (theta_pix / sigma) ** 2)
    beam_map /= np.sum(beam_map)
    sky_map = rng.random(npix)

    lst = np.asarray([0.0, 100.0, 200.0, 300.0])
    az = np.asarray([0.0, 45.0, -60.0, 123.4])
    el = np.asarray([41.0, 60.0, 30.0, 85.0])
    sr = np.zeros(4)

    linear = oracle_tod(beam_map, sky_map, lst, lat, az, el, sr)
    default = sim.generate_TOD_sky(beam_map, sky_map, lst, lat, az, el, sr)
    native = generate_tod_sky(
        jnp.asarray(beam_alm_iter3(beam_map, lmax)),
        jnp.asarray(quad_alm(sky_map, lmax)),
        _zyz_stack(lst, lat, az, el, sr),
        lmax=lmax,
    )
    scale = np.max(np.abs(linear))
    rel_native = np.max(np.abs(np.asarray(native) - linear)) / scale
    rel_trunc = np.max(np.abs(np.asarray(default) - linear)) / scale
    assert rel_native < 1e-12, f"port broke the linear chain: {rel_native:.3e}"
    assert rel_trunc < 1e-2, f"truncation effect blew up: {rel_trunc:.3e}"


def test_rotate_alm_public_matches_healpy(rng):
    """Public rotate_alm(alm, psi, theta, phi) == _rotate_healpix_map's alm op."""
    lmax = 23
    alm = hp.map2alm(rng.standard_normal(hp.nside2npix(8)), lmax=lmax)
    for psi, theta, phi in [(0.7, 1.2, 0.3), (-2.1, 2.9, 4.0), (0.5, 0.0, 1.0)]:
        ref = alm.copy()
        hp.rotate_alm(ref, phi, theta, psi)  # limTOD's argument order
        got = rotate_alm(
            jnp.asarray(alm),
            jnp.asarray(psi),
            jnp.asarray(theta),
            jnp.asarray(phi),
            lmax=lmax,
        )
        scale = np.max(np.abs(alm))
        assert np.max(np.abs(np.asarray(got) - ref)) / scale < 1e-11
