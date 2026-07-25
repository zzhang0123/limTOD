"""The limTOD.simeer deprecation shim (name shipped only in 1.4.0)."""

import importlib
import sys
import warnings

import pytest


def _fresh_import():
    for mod in [m for m in sys.modules if m.startswith("limTOD.simeer")]:
        del sys.modules[mod]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("limTOD.simeer")
    return module, caught


def test_import_warns_deprecation():
    _, caught = _fresh_import()
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "limTOD.patchbeam" in str(w.message)
        for w in caught
    )


def test_old_names_alias_new_ones():
    shim, _ = _fresh_import()
    import limTOD.patchbeam as pb

    assert shim.SimeerTODSim is pb.PatchBeamTODSim
    assert shim.PatchBeamTODSim is pb.PatchBeamTODSim
    assert shim.MeerKLASSBeam is pb.MeerKLASSBeam
    assert shim.integrate_tod is pb.integrate_tod
    assert shim.synthetic_gaussian_beam is pb.synthetic_gaussian_beam


def test_submodule_paths_still_import():
    _fresh_import()
    import limTOD.patchbeam.beam as pb_beam
    from limTOD.simeer import beam as shim_beam

    assert shim_beam is pb_beam
    assert importlib.import_module("limTOD.simeer.sky_integrator") is (
        importlib.import_module("limTOD.patchbeam.sky_integrator")
    )


def test_getattr_delegates_unknown_names():
    shim, _ = _fresh_import()
    import limTOD.patchbeam as pb

    # The 1.4.0-era deprecated spelling routed through patchbeam's own
    # __getattr__.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert shim.materialise_sky_cube is not None
    with pytest.raises(AttributeError):
        shim.definitely_not_a_real_name
