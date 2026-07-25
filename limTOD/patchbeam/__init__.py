"""
limTOD.patchbeam: MeerKLASS-optimal TOD path with native (l, m) beam interpolation.

Merged from the standalone Simeer package (same author/license); the
original repository remains available, and this subpackage is the
maintained home going forward. Purely additive: nothing in classic limTOD
usage changes.

Sky TOD vs Full TOD
-------------------

The patch-beam path computes the **Sky TOD**: the noiseless beam-weighted sky signal,
``T_sky(nu, t) = (1/Omega_b(nu)) * integral B(l, m, nu) T(l, m, nu) dOmega``.
It does NOT add gain, 1/f noise, or white noise; the **Full TOD**
assembly (multiplying by gain and injecting noise) is delegated to
:class:`limTOD.TODSim.generate_TOD`, which :class:`PatchBeamTODSim`
inherits unchanged.

Public API
----------

Sky-TOD generation:

*   :func:`limTOD.patchbeam.sky_integrator.integrate_tod` -- Sky TOD for a
    list of pointings; returns ``(n_freq, n_time)`` ndarray.
*   :func:`limTOD.patchbeam.sky_integrator.integrate_sample` -- Sky TOD for
    one pointing; returns ``(n_freq,)`` ndarray.

Full-TOD generation:

*   :class:`PatchBeamTODSim` -- drop-in replacement for
    :class:`limTOD.TODSim` whose sky-TOD step uses the (l, m) disc
    path. ``PatchBeamTODSim.generate_TOD(...)`` returns
    ``(overall_TOD, sky_TOD, gain_noise)``.

Building blocks (useful for custom pipelines / tests):

*   :class:`MeerKLASSBeam` -- load and query the holographic beam.
*   :func:`limTOD.patchbeam.projection.direction_cosines`
*   :func:`limTOD.patchbeam.interpolation.precompute_bilinear_weights`
*   :func:`limTOD.patchbeam.disc.select_disc`

Parallelism: ``n_jobs != 1`` needs ``joblib`` (pip install joblib, or the
``[full]`` extra); the default serial path has no extra dependency.
"""

from .beam import MeerKLASSBeam, synthetic_gaussian_beam
from .sky_integrator import (
    integrate_sample,
    integrate_tod,
    materialise_sky_cube,
    materialize_sky_cube,
)

__all__ = [
    "PatchBeamTODSim",
    "MeerKLASSBeam",
    "synthetic_gaussian_beam",
    "integrate_sample",
    "integrate_tod",
    "materialize_sky_cube",
    # ``materialise_sky_cube`` is the deprecated British spelling kept for
    # back-compat; will be removed in v0.2.
    "materialise_sky_cube",
]

# Version is that of the limTOD distribution this subpackage ships in.
from limTOD import __version__  # noqa: E402,F401

__author__ = "Zheng Zhang"
__email__ = "zheng.zhang@manchester.ac.uk"
__license__ = "MIT"


def __getattr__(name: str) -> object:
    """Lazy import of :class:`PatchBeamTODSim` (defers the simulator module,
    and with it astropy, until the class is actually referenced)."""
    if name == "PatchBeamTODSim":
        from .simulator import PatchBeamTODSim as _S

        return _S
    raise AttributeError(name)


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
