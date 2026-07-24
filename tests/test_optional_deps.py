"""mpi4py and pygdsm are optional: limTOD must import and run without them.

Since the test environment usually HAS both installed, absence is simulated
in a subprocess whose meta-path blocks their import (raising ImportError as
a real missing package would). Pinned behaviors (CHANGELOG 1.3.0):

* ``import limTOD`` (simulator, sky_model, HPW_filter) succeeds;
* ``limTOD.mpiutil`` degrades to serial mode (rank=0, size=1, world=None)
  and its helpers behave serially;
* the core TOD chain runs end to end;
* ``GDSM_sky_model`` raises an ImportError pointing at ``limTOD[gdsm]``.
"""

import subprocess
import sys

_BLOCKER_SCRIPT = r"""
import sys
import importlib.abc


class Blocker(importlib.abc.MetaPathFinder):
    blocked = {"mpi4py", "pygdsm"}

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError(f"{fullname} blocked to simulate a missing optional dep")
        return None


sys.meta_path.insert(0, Blocker())

import numpy as np

import limTOD
import limTOD.mpiutil as mpiutil
from limTOD.simulator import generate_TOD_sky
from limTOD.sky_model import GDSM_sky_model

assert "mpi4py" not in sys.modules and "pygdsm" not in sys.modules

# --- serial fallback state ---
assert mpiutil.rank == 0 and mpiutil.size == 1, (mpiutil.rank, mpiutil.size)
assert mpiutil.world is None and mpiutil.rank0 is True

# --- mpiutil helpers behave serially ---
assert mpiutil.partition_list_mpi([1, 2, 3]) == [1, 2, 3]
mpiutil.barrier()
assert mpiutil.parallel_map_gather(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

# --- core TOD chain end to end (healpy is a base dependency) ---
import healpy as hp

nside = 4
npix = hp.nside2npix(nside)
rng = np.random.default_rng(0)
tod = generate_TOD_sky(
    rng.random(npix), rng.random(npix),
    np.array([0.0, 30.0]), 53.2, np.zeros(2), np.full(2, 90.0), np.zeros(2),
)
assert tod.shape == (2,) and np.all(np.isfinite(tod))

# --- TODSim path: the code that actually consumes mpiutil in serial mode
#     (partition_list_mpi + the `size == 1` branches in simulate_sky_TOD) ---
def _flat_map(*, freq, nside):
    return np.full(12 * nside**2, 1.0)

sim = limTOD.TODSim(
    beam_func=_flat_map, sky_func=_flat_map, beam_nside=nside, sky_nside=nside
)
sky_tod = sim.simulate_sky_TOD(
    freq_list=[100.0], time_list=[0.0, 2.0],
    azimuth_deg_list=[0.0, 0.0], elevation_deg=90.0,
)
assert sky_tod.shape == (1, 2) and np.all(np.isfinite(sky_tod))

# --- GDSM raises a helpful error ---
try:
    GDSM_sky_model(freq=100.0, nside=nside)
except ImportError as exc:
    assert "limTOD[gdsm]" in str(exc), str(exc)
else:
    raise AssertionError("GDSM_sky_model should raise ImportError without pygdsm")

print("OPTIONAL-DEPS OK")
"""


def test_limtod_runs_without_mpi4py_and_pygdsm():
    out = subprocess.run(
        [sys.executable, "-c", _BLOCKER_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OPTIONAL-DEPS OK" in out.stdout


def test_mpi_path_unchanged_when_mpi4py_present():
    """With mpi4py installed (this process), mpiutil exposes a real communicator."""
    import importlib.util

    if importlib.util.find_spec("mpi4py") is None:
        import pytest

        pytest.skip("mpi4py not installed in this environment")
    import limTOD.mpiutil as mpiutil

    assert mpiutil.world is not None
    assert mpiutil.size >= 1 and mpiutil.rank == 0
    assert mpiutil.partition_list_mpi([1, 2, 3]) == [1, 2, 3]  # serial run
