"""Test that splitting rotate_alm into two calls is equivalent to one call."""

import numpy as np
import healpy as hp


def test_rotate_alm_split_vs_single():
    """Confirm that applying Z(phi)Y(theta) then Z(psi) separately
    gives the same result as a single rotate_alm(phi, theta, psi)."""
    nside = 16
    lmax = 3 * nside - 1
    np.random.seed(42)

    # Random sky map -> alm
    sky = np.random.randn(hp.nside2npix(nside))
    alm = hp.map2alm(sky, lmax=lmax)

    phi_rad, theta_rad, psi_rad = 0.3, 1.2, 0.7

    # Two-step rotation
    alm_two_step = alm.copy()
    hp.rotate_alm(alm_two_step, phi_rad, theta_rad, 0.0)
    hp.rotate_alm(alm_two_step, 0.0, 0.0, psi_rad)

    # Single-call rotation
    alm_single = alm.copy()
    hp.rotate_alm(alm_single, phi_rad, theta_rad, psi_rad)

    np.testing.assert_allclose(alm_two_step, alm_single, atol=1e-12,
                               err_msg="Two-step and single-call rotations differ")


if __name__ == "__main__":
    test_rotate_alm_split_vs_single()
    print("PASSED: two-step and single-call rotations are identical.")
