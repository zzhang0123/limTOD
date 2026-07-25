"""limTOD.uvbeam: pyuvdata UVBeam adapters.

The az-convention lock follows the house methodology: the mapping
``az_uvbeam = pi/2 - phi_healpix`` was selected by a numerical probe
(strongly displaced circular beam, three-way comparison against the
simeer (l, m) disc path) with the winner at ~0.5% and every other
candidate mapping 66-90% off; this suite keeps both directions pinned.
"""

import subprocess
import sys

import healpy as hp
import numpy as np
import pytest

pyuvdata = pytest.importorskip("pyuvdata")
from pyuvdata import UVBeam  # noqa: E402
from pyuvdata.analytic_beam import GaussianBeam  # noqa: E402

from limTOD.simeer import SimeerTODSim  # noqa: E402
from limTOD.simeer.beam import MeerKLASSBeam  # noqa: E402
from limTOD.simulator import TODSim  # noqa: E402
from limTOD.uvbeam import (  # noqa: E402
    healpix_phi_to_uvbeam_az,
    uvbeam_beam_func,
    uvbeam_to_healpix_maps,
    uvbeam_to_patch_beam,
)

FREQS_HZ = np.array([950e6, 1050e6])


def make_power_uvbeam(b_of_lm, *, za_max_deg=12.0, n_za=121, n_az=361, freqs=FREQS_HZ):
    """Build an XX+YY power UVBeam from an analytic B(l, m) power pattern.

    (l, m) are direction cosines with l = East, m = North under pyuvdata's
    azimuth convention (E = 0, N = pi/2).
    """
    az = np.linspace(0.0, 2 * np.pi, n_az)[:-1]
    za = np.linspace(0.0, np.deg2rad(za_max_deg), n_za)
    AZ, ZA = np.meshgrid(az, za, indexing="xy")
    ell = np.sin(ZA) * np.cos(AZ)
    emm = np.sin(ZA) * np.sin(AZ)
    data = b_of_lm(ell, emm)[None, None, None, :, :]
    data = np.repeat(data, len(freqs), axis=2)
    data = np.repeat(data, 2, axis=1)  # identical XX and YY
    return UVBeam.new(
        telescope_name="test",
        data_normalization="physical",
        freq_array=np.asarray(freqs, dtype=np.float64),
        feed_name="test-feed",
        feed_version="0",
        model_name="analytic",
        model_version="0",
        polarization_array=np.array([-5, -6]),
        feed_array=np.array(["x", "y"]),
        feed_angle=np.array([np.pi / 2, 0.0]),
        pixel_coordinate_system="az_za",
        axis1_array=az,
        axis2_array=za,
        data_array=data.astype(np.float64),
    )


def gaussian_efield_uvbeam(sigma_rad=0.03, freqs=FREQS_HZ, za_max_deg=20.0):
    gb = GaussianBeam(sigma=sigma_rad)  # achromatic efield Gaussian
    az = np.linspace(0.0, 2 * np.pi, 181)[:-1]
    za = np.linspace(0.0, np.deg2rad(za_max_deg), 101)
    return gb.to_uvbeam(
        freq_array=np.asarray(freqs, dtype=np.float64),
        axis1_array=az,
        axis2_array=za,
        beam_type="efield",
    )


@pytest.fixture(scope="module")
def circ_power_beam():
    sig = np.deg2rad(2.0)
    return make_power_uvbeam(lambda l, m: np.exp(-(l**2 + m**2) / sig**2))


class TestHealpixMaps:
    def test_shapes_and_dtype(self, circ_power_beam):
        m_i = uvbeam_to_healpix_maps(circ_power_beam, freq_MHz=1000.0, nside=32)
        assert m_i.shape == (hp.nside2npix(32),) and m_i.dtype == np.float64

        efield = gaussian_efield_uvbeam()
        m_iquv = uvbeam_to_healpix_maps(efield, freq_MHz=1000.0, nside=32, stokes="IQUV")
        assert m_iquv.shape == (4, hp.nside2npix(32))

    def test_circular_gaussian_matches_analytic(self, circ_power_beam):
        nside = 64
        sig = np.deg2rad(2.0)
        out = uvbeam_to_healpix_maps(circ_power_beam, freq_MHz=1000.0, nside=nside)
        theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        inside = theta <= np.deg2rad(12.0)
        expected = np.exp(-np.sin(theta[inside]) ** 2 / sig**2)
        np.testing.assert_allclose(out[inside], expected, atol=2e-3)
        # beyond the UVBeam za coverage: filled with zero
        assert np.all(out[~inside] == 0.0)

    def test_efield_pstokes_matches_power_copol_average(self):
        """pyuvdata's pstokes-I from an efield Gaussian equals the co-pol
        power average of the matching power beam (peak-normalized)."""
        sig = 0.03
        efield = gaussian_efield_uvbeam(sigma_rad=sig)
        # matching POWER pattern: |E|^2 has sigma_p = sigma_e / sqrt(2)... in
        # direction-cosine form B = exp(-(r/sig)^2) with r = sin(za) is what
        # GaussianBeam(sigma) produces for the power product; compare the two
        # adapter paths against each other rather than re-deriving:
        m_e = uvbeam_to_healpix_maps(
            efield, freq_MHz=1000.0, nside=32, stokes="I", peak_normalize=True
        )
        power = efield.copy()
        power.efield_to_power(calc_cross_pols=False)
        m_p = uvbeam_to_healpix_maps(
            power, freq_MHz=1000.0, nside=32, stokes="I", peak_normalize=True
        )
        np.testing.assert_allclose(m_e, m_p, atol=1e-8)

    def test_peak_normalize(self, circ_power_beam):
        """Scaling the beam by an arbitrary factor must be undone exactly."""
        sig = np.deg2rad(2.0)
        scaled = make_power_uvbeam(
            lambda l, m: 7.3 * np.exp(-(l**2 + m**2) / sig**2)
        )
        out_scaled = uvbeam_to_healpix_maps(
            scaled, freq_MHz=1000.0, nside=32, peak_normalize=True
        )
        out_unit = uvbeam_to_healpix_maps(
            circ_power_beam, freq_MHz=1000.0, nside=32, peak_normalize=False
        )
        np.testing.assert_allclose(out_scaled, out_unit, atol=1e-10)
        assert out_scaled.max() <= 1.0 + 1e-9

    def test_frequency_interpolation(self):
        """A chromatic beam interpolates between its native channels."""
        sigs = {950.0: np.deg2rad(2.4), 1050.0: np.deg2rad(1.6)}

        def chromatic(freqs=FREQS_HZ):
            az = np.linspace(0.0, 2 * np.pi, 181)[:-1]
            za = np.linspace(0.0, np.deg2rad(12.0), 101)
            AZ, ZA = np.meshgrid(az, za, indexing="xy")
            r2 = np.sin(ZA) ** 2
            cubes = [np.exp(-r2 / sigs[f / 1e6] ** 2) for f in freqs]
            data = np.stack(cubes)[None, None, :, :, :]
            data = np.repeat(data, 2, axis=1)
            return UVBeam.new(
                telescope_name="test",
                data_normalization="physical",
                freq_array=np.asarray(freqs, dtype=np.float64),
                feed_name="test-feed",
                feed_version="0",
                model_name="chromatic",
                model_version="0",
                polarization_array=np.array([-5, -6]),
                feed_array=np.array(["x", "y"]),
                feed_angle=np.array([np.pi / 2, 0.0]),
                pixel_coordinate_system="az_za",
                axis1_array=az,
                axis2_array=za,
                data_array=data.astype(np.float64),
            )

        uvb = chromatic()
        nside = 32
        m_lo = uvbeam_to_healpix_maps(uvb, freq_MHz=950.0, nside=nside)
        m_mid = uvbeam_to_healpix_maps(uvb, freq_MHz=1000.0, nside=nside)
        m_hi = uvbeam_to_healpix_maps(uvb, freq_MHz=1050.0, nside=nside)
        # Beam narrows with frequency: off-axis response strictly between.
        theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        probe = (theta > np.deg2rad(2.0)) & (theta < np.deg2rad(6.0))
        assert np.all(m_mid[probe] <= m_lo[probe] + 1e-12)
        assert np.all(m_mid[probe] >= m_hi[probe] - 1e-12)


class TestIQUVRowOrder:
    """Rows of stokes='IQUV' output must be [pI, pQ, pU, pV] in that order —
    pinned against a power beam whose four pseudo-Stokes products are
    numerically distinguishable constants times a Gaussian."""

    SCALES = {"pI": 1.0, "pQ": 0.5, "pU": 0.25, "pV": 0.125}

    def _pstokes_beam(self):
        sig = np.deg2rad(2.0)
        az = np.linspace(0.0, 2 * np.pi, 181)[:-1]
        za = np.linspace(0.0, np.deg2rad(10.0), 81)
        AZ, ZA = np.meshgrid(az, za, indexing="xy")
        base = np.exp(-np.sin(ZA) ** 2 / sig**2)
        # (Naxes_vec=1, Npol=4, Nfreq=1, Nza, Naz)
        data = np.stack([c * base for c in self.SCALES.values()])[None, :, None, :, :]
        return UVBeam.new(
            telescope_name="test",
            data_normalization="physical",
            freq_array=np.array([1e9]),
            feed_name="f", feed_version="0",
            model_name="pstokes", model_version="0",
            polarization_array=np.array([1, 2, 3, 4]),  # pI, pQ, pU, pV
            feed_array=np.array(["x", "y"]),
            feed_angle=np.array([np.pi / 2, 0.0]),
            pixel_coordinate_system="az_za",
            axis1_array=az, axis2_array=za,
            data_array=data,
        )

    def test_rows_in_iquv_order(self):
        uvb = self._pstokes_beam()
        maps = uvbeam_to_healpix_maps(uvb, freq_MHz=1000.0, nside=16, stokes="IQUV")
        pole_pix = 0  # near boresight, base ~ 1
        vals = maps[:, pole_pix]
        ratios = vals / vals[0]
        np.testing.assert_allclose(ratios, [1.0, 0.5, 0.25, 0.125], rtol=1e-6)


class TestOrientationLock:
    """The az-convention lock: displaced circular beam, three-way agreement.

    Resolution matters: the sigma=0.9 deg blob needs nside=128 and fine
    (az, za)/(l, m) grids for the two paths to agree at the sub-percent
    level (the numbers below reproduce the original lock probe: winner
    0.5%, every other candidate mapping 66-90% off). At nside=64 the
    pixelization error alone is tens of percent and proves nothing.
    """

    SL = SM = np.deg2rad(0.9)
    L0, M0 = np.deg2rad(3.0), np.deg2rad(1.5)
    NSIDE = 128
    FREQ_HZ = np.array([1000e6])

    @classmethod
    def _b(cls, l, m):
        return np.exp(-((l - cls.L0) ** 2 / cls.SL**2 + (m - cls.M0) ** 2 / cls.SM**2))

    @pytest.fixture(scope="class")
    def scan(self):
        nside = self.NSIDE
        rng_cl = np.exp(-np.arange(3 * nside) * 0.02)
        np.random.seed(0)
        sky_map = 20.0 + hp.synfast(rng_cl, nside)

        def sky_func(*, freq, nside):
            return sky_map

        n_t = 6
        return {
            "sky_func": sky_func,
            "lat": -30.713,
            "az": np.linspace(-50.0, 60.0, n_t),
            "el": np.full(n_t, 55.0),
            "t": np.arange(n_t) * 30.0,
        }

    @pytest.fixture(scope="class")
    def tod_simeer(self, scan):
        margin = np.linspace(-6.0, 6.0, 481)
        MM, LL = np.meshgrid(np.deg2rad(margin), np.deg2rad(margin), indexing="ij")
        cube = self._b(LL, MM)[None, :, :].astype(np.float32)
        patch = MeerKLASSBeam.from_arrays(
            freq_MHz=self.FREQ_HZ / 1e6, margin_deg=margin, power={"HH": cube}
        )
        sim = SimeerTODSim(
            beam=patch, sky_func=scan["sky_func"], sky_nside=self.NSIDE,
            disc_radius_deg=9.0, polarization="HH",
            ant_latitude_deg=scan["lat"],
        )
        return np.asarray(
            sim.simulate_sky_TOD([1000.0], scan["t"], scan["az"], scan["el"])
        )

    def _tod_healpix(self, scan, phi_to_az):
        import limTOD.uvbeam as luv

        uvb = make_power_uvbeam(self._b, n_za=241, n_az=721, freqs=self.FREQ_HZ)
        orig = luv.healpix_phi_to_uvbeam_az
        luv.healpix_phi_to_uvbeam_az = phi_to_az
        try:
            sim = TODSim(
                ant_latitude_deg=scan["lat"],
                beam_func=uvbeam_beam_func(uvb),
                sky_func=scan["sky_func"],
                beam_nside=self.NSIDE,
                sky_nside=self.NSIDE,
            )
            return np.asarray(
                sim.simulate_sky_TOD(
                    [1000.0], scan["t"], scan["az"], scan["el"], normalize_beam=True
                )
            )
        finally:
            luv.healpix_phi_to_uvbeam_az = orig

    def test_frozen_mapping_agrees_with_simeer_path(self, scan, tod_simeer):
        tod_l = self._tod_healpix(scan, healpix_phi_to_uvbeam_az)
        rel = np.max(np.abs(tod_l - tod_simeer) / np.abs(tod_simeer))
        assert rel < 0.02, f"frozen mapping disagrees with the disc path: {rel:.3f}"

    def test_mirror_mapping_still_rejected(self, scan, tod_simeer):
        """If the mirrored mapping ever starts matching, the beam-map
        handedness moved somewhere in the chain."""
        mirror = lambda p: (np.asarray(p) - 0.5 * np.pi) % (2 * np.pi)  # noqa: E731
        tod_l = self._tod_healpix(scan, mirror)
        rel = np.max(np.abs(tod_l - tod_simeer) / np.abs(tod_simeer))
        assert rel > 0.2, f"mirror mapping unexpectedly close: {rel:.3f}"

    def test_frozen_mapping_values(self):
        np.testing.assert_allclose(float(healpix_phi_to_uvbeam_az(0.0)), np.pi / 2)
        np.testing.assert_allclose(float(healpix_phi_to_uvbeam_az(np.pi / 2)), 0.0)


class TestPatchBridge:
    def test_patch_beam_matches_analytic_cube(self):
        sig = np.deg2rad(1.5)
        uvb = make_power_uvbeam(lambda l, m: np.exp(-(l**2 + m**2) / sig**2))
        margin = np.linspace(-5.0, 5.0, 101)
        patch = uvbeam_to_patch_beam(uvb, margin_deg=margin, polarization="HH")
        assert isinstance(patch, MeerKLASSBeam)
        np.testing.assert_allclose(np.asarray(patch.freq_MHz), FREQS_HZ / 1e6)

        MM, LL = np.meshgrid(np.deg2rad(margin), np.deg2rad(margin), indexing="ij")
        expected = np.exp(-(LL**2 + MM**2) / sig**2)
        got = patch.power_cube("HH")[0]
        np.testing.assert_allclose(got, expected, atol=2e-3)

    def test_patch_beam_from_efield(self):
        efield = gaussian_efield_uvbeam()
        margin = np.linspace(-5.0, 5.0, 51)
        patch = uvbeam_to_patch_beam(efield, margin_deg=margin, polarization="VV")
        cube = patch.power_cube("VV")
        assert np.all(np.isfinite(cube)) and cube.max() > 0

    def test_patch_beam_plugs_into_simeer(self, ):
        sig = np.deg2rad(2.0)
        uvb = make_power_uvbeam(lambda l, m: np.exp(-(l**2 + m**2) / sig**2))
        patch = uvbeam_to_patch_beam(uvb, margin_deg=np.linspace(-6.0, 6.0, 121))
        sky = np.full(hp.nside2npix(32), 5.0)
        sim = SimeerTODSim(
            beam=patch, sky_func=lambda *, freq, nside: sky, sky_nside=32,
            disc_radius_deg=8.0,
        )
        tod = sim.simulate_sky_TOD([950.0, 1050.0], [0.0, 60.0], [0.0, 10.0], 55.0)
        assert tod.shape == (2, 2)
        # flat sky through a normalized beam average returns the sky value
        np.testing.assert_allclose(tod, 5.0, rtol=5e-2)


class TestErrorPaths:
    def test_single_pol_power_rejected_for_stokes_i(self):
        az = np.linspace(0.0, 2 * np.pi, 91)[:-1]
        za = np.linspace(0.0, 0.2, 41)
        data = np.ones((1, 1, 1, 41, 90))
        uvb = UVBeam.new(
            telescope_name="test", data_normalization="physical",
            freq_array=np.array([1e9]), feed_name="f", feed_version="0",
            model_name="m", model_version="0",
            polarization_array=np.array([-5]),
            feed_array=np.array(["x"]),
            feed_angle=np.array([np.pi / 2]),
            pixel_coordinate_system="az_za",
            axis1_array=az, axis2_array=za, data_array=data,
        )
        with pytest.raises(ValueError, match="cannot provide stokes"):
            uvbeam_to_healpix_maps(uvb, freq_MHz=1000.0, nside=16)

    def test_power_beam_rejected_for_iquv(self, circ_power_beam):
        with pytest.raises(ValueError, match="cannot provide stokes"):
            uvbeam_to_healpix_maps(
                circ_power_beam, freq_MHz=1000.0, nside=16, stokes="IQUV"
            )

    def test_bad_stokes_string(self, circ_power_beam):
        with pytest.raises(ValueError, match="stokes must be"):
            uvbeam_to_healpix_maps(circ_power_beam, freq_MHz=1000.0, nside=16, stokes="QU")

    def test_healpix_coordinate_beam_rejected(self):
        class _FakeHealpixBeam:
            pixel_coordinate_system = "healpix"
            beam_type = "power"

        with pytest.raises(NotImplementedError, match="az_za"):
            uvbeam_to_healpix_maps(_FakeHealpixBeam(), freq_MHz=1000.0, nside=16)

    def test_bad_patch_polarization(self, circ_power_beam):
        with pytest.raises(ValueError, match="polarization must be"):
            uvbeam_to_patch_beam(
                circ_power_beam, margin_deg=np.linspace(-3, 3, 11), polarization="RR"
            )

    def test_missing_pyuvdata_error_message(self):
        code = (
            "import sys, importlib.abc\n"
            "class B(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        if name.split('.')[0] == 'pyuvdata':\n"
            "            raise ImportError('blocked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, B())\n"
            "from limTOD.uvbeam import uvbeam_beam_func\n"
            "try:\n"
            "    uvbeam_beam_func(object())\n"
            "except ImportError as e:\n"
            "    assert 'limTOD[uvbeam]' in str(e), str(e)\n"
            "    print('MESSAGE OK')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        assert out.returncode == 0, out.stderr[-1500:]
        assert "MESSAGE OK" in out.stdout
