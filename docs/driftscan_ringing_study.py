"""Horizon-mask ringing study for the drift-scan m-mode path.

Question (docs/driftscan.md): the physically correct drift-scan beam is the
horizon-MASKED beam, but the m-mode formalism represents it at a finite
band-limit — a sharp cut is not bandlimited, so Gibbs ringing appears. How
large is the resulting TOD error at practical lmax, and how much does
cosine apodization of the cut buy?

Protocol (pure healpy — this is a study script, not package code):

* Horizontal-frame beam map at ``NSIDE_HI = 256``: Gaussian beam rotated to
  (az, el), times the elevation mask (hard cut or cosine apodized).
* Bandlimited sky, ``C_l ∝ (l+10)^-2.7`` up to ``LMAX_SKY`` — its harmonic
  rotation to the horizontal frame at each LST is EXACT, so the reference
  pixel dot ``Σ_p B(p)·S_t(p)`` carries no sky-side band-limit error.
* Reference TOD: pixel dot with the sharply masked beam map itself.
* Test TOD at working band-limit L: pixel dot with the beam RECONSTRUCTED
  from ``map2alm(masked_map, lmax=L, iter=3)`` — by the quadrature-alm
  exactness identity this equals the m-mode/harmonic TOD at lmax=L.
* Metrics per (beam, el, apod, L):
    - relTOD: RMS(TOD_L − TOD_ref) / RMS(TOD_ref)
    - leak:  RMS of the reconstructed beam BELOW the horizon / beam peak
      (pure ringing: the target is identically zero there)
* "mask" column: RMS(TOD_unmasked − TOD_ref)/RMS(TOD_ref) — how much the
  mask matters at all for that geometry (upper bound on the error of
  ignoring it, the numpy-limTOD default).

Run:  python docs/driftscan_ringing_study.py            (~3 min)
"""

import numpy as np
import healpy as hp

from limTOD.simulator import zyz_of_pointing

NSIDE_HI = 256
LMAX_REF = 3 * NSIDE_HI  # sharp-mask reference representation
LMAX_SKY = 150
N_LST = 24
LAT = -30.7
RNG = np.random.default_rng(42)

BEAMS = {"narrow(2deg)": 2.0, "wide(25deg)": 25.0}
ELEVATIONS = [10.0, 41.0, 75.0]
APODS = [0.0, 1.0, 2.0, 5.0, 10.0]
WORKING_LMAX = [96, 192, 384]


def gaussian_beam_alm(fwhm_deg, lmax):
    sigma = np.deg2rad(fwhm_deg) / np.sqrt(8.0 * np.log(2.0))
    theta, _ = hp.pix2ang(NSIDE_HI, np.arange(hp.nside2npix(NSIDE_HI)))
    return hp.map2alm(np.exp(-(theta**2) / (2 * sigma**2)), lmax=lmax, iter=3)


def horizontal_beam_map(beam_alm, az, el):
    """Beam rotated into the horizontal frame (peak at chart (90-el, 180-az))."""
    from limTOD.simulator import zyzyz2zyz

    psi, theta, phi = zyzyz2zyz(0.0, 0.0, -az, el - 90.0, 0.0)
    a = beam_alm.copy()
    hp.rotate_alm(a, phi, theta, psi)  # limTOD slot convention (locked)
    return hp.alm2map(a, NSIDE_HI)


def elevation_weights(apod_deg):
    theta, _ = hp.pix2ang(NSIDE_HI, np.arange(hp.nside2npix(NSIDE_HI)))
    el = 90.0 - np.rad2deg(theta)
    if apod_deg == 0.0:
        return (el > 0.0).astype(float)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip(el, 0.0, apod_deg) / apod_deg))
    return np.where(el <= 0.0, 0.0, np.where(el >= apod_deg, 1.0, ramp))


def sky_alm():
    ell = np.arange(LMAX_SKY + 1)
    cl = 1.0 / (ell + 10.0) ** 2.7
    return hp.synalm(cl, lmax=LMAX_SKY, new=True)


def horizontal_skies(s_alm, az, el):
    """Sky synthesized in the horizontal frame at each LST (exact rotation).

    The celestial->horizontal transform is the INVERSE of limTOD's
    beam-local->celestial chain evaluated at the parked configuration
    (az=0, el=90, selfrot=0), where beam-local == horizontal (pinned in
    tests/test_beam_orientation.py). az/el of the DRIFT POINTING do not
    enter the sky rotation - only LST and latitude do.
    """
    skies = np.empty((N_LST, hp.nside2npix(NSIDE_HI)))
    for i, lst in enumerate(np.linspace(0.0, 360.0, N_LST, endpoint=False)):
        psi, theta, phi = zyz_of_pointing(lst, LAT, 0.0, 90.0, 0.0)
        a = s_alm.copy()
        # inverse of rotate(psi,theta,phi): healpy slots swapped + negated
        hp.rotate_alm(a, -psi, -theta, -phi)
        skies[i] = hp.alm2map(a, NSIDE_HI)
    return skies


def main():
    s_alm = sky_alm()
    print(f"# nside_hi={NSIDE_HI} lmax_ref={LMAX_REF} lmax_sky={LMAX_SKY} "
          f"n_lst={N_LST} lat={LAT}")
    print(f"{'beam':>13} {'el':>5} {'apod':>5} {'mask':>9} "
          + " ".join(f"relTOD@L{L:<4d} leak@L{L:<4d}" for L in WORKING_LMAX))

    for bname, fwhm in BEAMS.items():
        b_alm = gaussian_beam_alm(fwhm, LMAX_REF)
        for el in ELEVATIONS:
            bh = horizontal_beam_map(b_alm, az=41.5, el=el)
            skies = horizontal_skies(s_alm, az=41.5, el=el)
            below = elevation_weights(0.0) == 0.0
            peak = bh.max()
            tod_unmasked = skies @ bh
            for apod in APODS:
                masked = bh * elevation_weights(apod)
                tod_ref = skies @ masked
                rms_ref = np.sqrt(np.mean(tod_ref**2))
                mask_size = np.sqrt(np.mean((tod_unmasked - tod_ref) ** 2)) / rms_ref
                cols = []
                for L in WORKING_LMAX:
                    recon = hp.alm2map(hp.map2alm(masked, lmax=L, iter=3), NSIDE_HI)
                    tod_l = skies @ recon
                    rel = np.sqrt(np.mean((tod_l - tod_ref) ** 2)) / rms_ref
                    leak = np.sqrt(np.mean(recon[below] ** 2)) / peak
                    cols.append(f"{rel:11.2e} {leak:10.2e}")
                print(f"{bname:>13} {el:5.1f} {apod:5.1f} {mask_size:9.2e} "
                      + " ".join(cols))


if __name__ == "__main__":
    main()
