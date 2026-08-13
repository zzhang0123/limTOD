"""Sky-side conventions: what a TRIS temperature actually contains.

The archive headers say only "Sky Brightness Temperature (K)".  They do not
state Rayleigh-Jeans versus thermodynamic, and they do not say whether the CMB
monopole is included.  Both were therefore settled **from the data**, by
requiring that the two rings give a physically sensible Galactic spectral
index (they share an RA grid, so this is a clean per-sample test):

===========================  ==========================================
Treatment                    Galactic spectral index across 600.5/817.8
===========================  ==========================================
CMB monopole left in         min -2.39, median -2.10, max -1.81
CMB monopole removed in RJ   min -3.12, median -2.91, max -2.63
===========================  ==========================================

Synchrotron at these frequencies is about -2.5 to -3.1, and it flattens toward
the Galactic plane.  Only the second row is physical, and it does flatten the
right way (-2.98 at the coldest RA, -2.76 at the hottest).  So:

**TRIS temperatures are antenna (Rayleigh-Jeans) temperatures and they include
the CMB monopole.**

This is not bookkeeping pedantry.  The RJ monopole is 2.7111 K at 600.5 MHz
and 2.7059 K at 817.8 MHz, while the thermodynamic value is 2.72548 K -- a
0.014-0.020 K difference, larger than the 0.010 K median statistical error on
a ring sample.  Use :func:`cmb_monopole_rj_k`, not 2.725, and add it to any
Galactic-only sky model before comparing with TRIS.
"""

import numpy as np

from ._validate import _RealScalarInput, _validate_positive_scalar

#: Planck 2018 CMB thermodynamic monopole temperature, in kelvin.
CMB_T0_K = 2.72548

_PLANCK_H = 6.62607015e-34
_BOLTZMANN_K = 1.380649e-23


def cmb_monopole_rj_k(
    frequency_mhz: _RealScalarInput, *, t_cmb_k: _RealScalarInput = CMB_T0_K
) -> float:
    """Return the CMB monopole as a Rayleigh-Jeans temperature, in kelvin.

    ``T_RJ = T0 * x / (exp(x) - 1)`` with ``x = h nu / (k T0)``.  At TRIS's
    effective frequencies this gives 2.7111 K (600.5 MHz) and 2.7059 K
    (817.8 MHz), both meaningfully below the thermodynamic 2.72548 K at the
    precision of the published statistical errors.
    """
    frequency = _validate_positive_scalar(frequency_mhz, "frequency_mhz")
    temperature = _validate_positive_scalar(t_cmb_k, "t_cmb_k")
    x = _PLANCK_H * frequency * 1.0e6 / (_BOLTZMANN_K * temperature)
    return float(temperature * x / np.expm1(x))


def to_tris_temperature_convention(
    galactic_map_k: np.ndarray,
    frequency_mhz: _RealScalarInput,
    *,
    t_cmb_k: _RealScalarInput = CMB_T0_K,
) -> np.ndarray:
    """Put a Galactic-only RJ sky map into the TRIS temperature convention.

    Sky models in this field (Haslam-derived maps, GSM/pyGDSM, ...) are
    antenna temperatures of Galactic emission with the CMB monopole removed.
    TRIS published temperatures include it.  This adds it back at the supplied
    frequency, which is the one line that makes a forward model comparable
    with the archive at the kelvin level.

    Pass the **effective** frequency (``ring.effective_frequency_mhz``), not
    the nominal file label.
    """
    sky = np.asarray(galactic_map_k, dtype=float)
    if sky.ndim != 1 or sky.size == 0:
        raise ValueError("galactic_map_k must be a non-empty one-dimensional map")
    if not np.all(np.isfinite(sky)):
        raise ValueError("galactic_map_k must contain only finite values")
    return sky + cmb_monopole_rj_k(frequency_mhz, t_cmb_k=t_cmb_k)


def galactic_spectral_index(
    temperature_low_k: np.ndarray,
    frequency_low_mhz: _RealScalarInput,
    temperature_high_k: np.ndarray,
    frequency_high_mhz: _RealScalarInput,
    *,
    t_cmb_k: _RealScalarInput = CMB_T0_K,
) -> np.ndarray:
    """Galactic spectral index between two TRIS rings, CMB monopole removed.

    Both inputs must be TRIS-convention temperatures sampled at the *same*
    sky positions; the published 600- and 820-MHz rings share an RA grid, so
    they can be passed directly.  The CMB monopole is subtracted in RJ units
    at each frequency before the ratio is taken -- skip that and the index
    comes out around -2.1, which is not synchrotron.
    """
    low = np.asarray(temperature_low_k, dtype=float)
    high = np.asarray(temperature_high_k, dtype=float)
    if low.shape != high.shape:
        raise ValueError("the two temperature arrays must have the same shape")
    nu_low = _validate_positive_scalar(frequency_low_mhz, "frequency_low_mhz")
    nu_high = _validate_positive_scalar(frequency_high_mhz, "frequency_high_mhz")
    if nu_low == nu_high:
        raise ValueError("the two frequencies must differ")
    galactic_low = low - cmb_monopole_rj_k(nu_low, t_cmb_k=t_cmb_k)
    galactic_high = high - cmb_monopole_rj_k(nu_high, t_cmb_k=t_cmb_k)
    if np.any(galactic_low <= 0.0) or np.any(galactic_high <= 0.0):
        raise ValueError(
            "CMB-subtracted temperatures must be positive to define a spectral index"
        )
    return np.log(galactic_low / galactic_high) / np.log(nu_low / nu_high)
