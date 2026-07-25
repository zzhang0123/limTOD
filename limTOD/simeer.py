"""Deprecated alias for :mod:`limTOD.patchbeam`.

The patch-beam subpackage shipped as ``limTOD.simeer`` only in release
1.4.0 (same-day); from 1.5.0 it lives at :mod:`limTOD.patchbeam` with
``PatchBeamTODSim`` replacing ``SimeerTODSim``. This shim keeps the old
imports working with a :class:`DeprecationWarning`; it will be removed
in 2.0.
"""

import sys
import warnings

import limTOD.patchbeam as _patchbeam
from limTOD.patchbeam import (  # noqa: F401  (re-exported public API)
    MeerKLASSBeam,
    PatchBeamTODSim,
    integrate_sample,
    integrate_tod,
    materialize_sky_cube,
    synthetic_gaussian_beam,
)
from limTOD.patchbeam import (
    beam,
    disc,
    interpolation,
    projection,
    simulator,
    sky_integrator,
    stokes,
)

warnings.warn(
    "limTOD.simeer was renamed to limTOD.patchbeam in 1.5.0 "
    "(SimeerTODSim -> PatchBeamTODSim); this alias will be removed in 2.0.",
    DeprecationWarning,
    stacklevel=2,
)

SimeerTODSim = PatchBeamTODSim

__all__ = [
    "MeerKLASSBeam",
    "PatchBeamTODSim",
    "SimeerTODSim",
    "integrate_sample",
    "integrate_tod",
    "materialize_sky_cube",
    "synthetic_gaussian_beam",
]

# 1.4.0 also exposed the submodules (limTOD.simeer.beam etc.); alias them
# so those import paths keep resolving.
for _name, _mod in [
    ("beam", beam), ("disc", disc), ("interpolation", interpolation),
    ("projection", projection), ("simulator", simulator),
    ("sky_integrator", sky_integrator), ("stokes", stokes),
]:
    sys.modules[f"{__name__}.{_name}"] = _mod


def __getattr__(name: str) -> object:
    # Anything else (e.g. the deprecated materialise_sky_cube alias)
    # delegates to the real package.
    return getattr(_patchbeam, name)
