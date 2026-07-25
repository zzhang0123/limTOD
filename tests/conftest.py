"""Suite-wide fixtures: keep the tests hermetic on offline machines.

astropy's apparent-sidereal-time computation (used by
``limTOD.simulator.generate_LSTs_deg``) consults IERS Earth-orientation
tables and by default may try to refresh them from the network. The bundled
tables fully cover the epochs the tests use, so disable auto-download to
avoid network latency/warnings on firewalled CI runners.
"""

try:
    from astropy.utils import iers

    iers.conf.auto_download = False
except ImportError:  # pragma: no cover — astropy is a base dependency
    pass
