"""CST far-field exports -> HEALPix beam maps.

The synthetic exports built here have answers known in closed form, so every
assertion is a statement about the reader rather than about the test's own
arithmetic. Two properties of the synthetic pattern do real work and are
explained where it is defined: the azimuthal modulation vanishes at the pole
(so "where is the peak?" is meaningful) and it is ODD in phi (so reversing the
handedness is detectable at all).

Moved here from rheplicant, where the reader used to live: how a measured beam
becomes a beam map is limTOD's subject, exactly as the horizon partition is.
"""

from __future__ import annotations

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
pytest.importorskip("scipy")
import healpy as hp  # noqa: E402

from limTOD.cstbeam import (  # noqa: E402
    cst_beam_func,
    cst_beam_maps,
    cst_frequency_table,
    read_cst_farfield,
)

THETA_STEP, PHI_STEP = 2.0, 5.0


def synthetic_directivity(theta_deg, phi_deg, *, sigma_deg, az_depth):
    """A Gaussian main lobe with genuine azimuthal structure, as directivity.

    Two properties the tests rely on:

    * the azimuthal modulation carries a ``sin(theta)`` factor, so it vanishes
      at the pole. A far-field pattern must be single-valued at ``theta = 0``;
      a modulation that survives there makes "where is the peak?" meaningless.
    * it is ``sin(phi)``, ODD about ``phi = 0``, so reversing the azimuth
      handedness is detectable. ``cos(phi)`` would be its own mirror image and
      the ``phi_sense`` test would pass on a no-op.

    Normalized so ``int D dOmega == 4*pi`` on its own quadrature.
    """
    theta = np.deg2rad(theta_deg)[:, None]
    phi = np.deg2rad(phi_deg)[None, :]
    pattern = np.exp(-0.5 * (theta / np.deg2rad(sigma_deg)) ** 2) * (
        1.0 + az_depth * np.sin(theta) * np.sin(phi)
    )
    weight = np.sin(theta) * np.deg2rad(THETA_STEP) * np.deg2rad(PHI_STEP)
    return 4.0 * np.pi * pattern / float((pattern * weight).sum())


def write_cst(path, *, sigma_deg=15.0, az_depth=0.5):
    theta_deg = np.arange(0.0, 180.0 + THETA_STEP, THETA_STEP)
    phi_deg = np.arange(0.0, 360.0, PHI_STEP)
    directivity = synthetic_directivity(
        theta_deg, phi_deg, sigma_deg=sigma_deg, az_depth=az_depth
    )
    rows = []
    for j, phi in enumerate(phi_deg):          # theta runs fastest, as CST writes
        for i, theta in enumerate(theta_deg):
            dbi = 10.0 * np.log10(directivity[i, j])
            rows.append(f"{theta:10.3f} {phi:10.3f} {dbi:22.14e} 0 0 0 0 0")
    path.write_text(
        "Theta [deg.]  Phi [deg.]  Abs(Dir.)[dBi]  Abs(Theta)[dBi]  "
        "Phase(Theta)[deg.]  Abs(Phi)[dBi]  Phase(Phi)[deg.]  Ax.Ratio[dB]\n"
        + "-" * 100 + "\n" + "\n".join(rows) + "\n"
    )
    return theta_deg, phi_deg, directivity


class TestReader:
    def test_it_recovers_the_grid_and_the_linear_power(self, tmp_path):
        path = tmp_path / "Horn70.txt"
        theta_deg, phi_deg, directivity = write_cst(path)
        got_theta, got_phi, got_dir = read_cst_farfield(path)
        np.testing.assert_allclose(got_theta, theta_deg)
        np.testing.assert_allclose(got_phi, phi_deg)
        np.testing.assert_allclose(got_dir, directivity, rtol=1e-10)

    def test_the_reshape_is_theta_fastest_not_phi_fastest(self, tmp_path):
        """CST writes theta fastest within each phi block. Reshaping the other
        way gives a correctly-shaped array with the samples transposed — a
        beam, just not this one."""
        path = tmp_path / "Horn70.txt"
        _, _, directivity = write_cst(path, az_depth=0.9)
        _, _, got = read_cst_farfield(path)
        table = np.loadtxt(path, skiprows=2)
        wrong = 10.0 ** (table[:, 2].reshape(directivity.shape) / 10.0)
        assert not np.allclose(got, wrong)
        np.testing.assert_allclose(got, directivity, rtol=1e-10)

    def test_an_incomplete_grid_is_refused(self, tmp_path):
        path = tmp_path / "Horn70.txt"
        write_cst(path)
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-5]) + "\n")
        with pytest.raises(ValueError, match="do not fill"):
            read_cst_farfield(path)

    def test_a_table_without_the_directivity_column_is_refused(self, tmp_path):
        path = tmp_path / "Horn70.txt"
        path.write_text("header\n----\n0.0 0.0\n2.0 0.0\n")
        with pytest.raises(ValueError, match="at least 3 columns"):
            read_cst_farfield(path)


class TestFrequencyTable:
    def test_frequencies_come_from_the_trailing_number_in_megahertz(self, tmp_path):
        write_cst(tmp_path / "HornDry70.5.txt")
        write_cst(tmp_path / "HornDry71.txt")
        (tmp_path / "notes.txt").write_text("no frequency here\n")
        table = cst_frequency_table(tmp_path)
        assert sorted(table) == [70.5, 71.0]

    def test_an_empty_directory_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="No CST exports"):
            cst_frequency_table(tmp_path)


class TestHealpixSampling:
    NSIDE = 16

    def test_a_directivity_still_integrates_to_four_pi(self, tmp_path):
        """Resampling a directivity must not change what it normalizes to."""
        write_cst(tmp_path / "Horn70.txt")
        maps = cst_beam_maps(tmp_path, [70.0], nside=self.NSIDE)
        integral = maps[0].sum() * 4.0 * np.pi / hp.nside2npix(self.NSIDE)
        assert abs(integral / (4.0 * np.pi) - 1.0) < 0.01

    def test_the_boresight_lands_on_the_pole(self, tmp_path):
        write_cst(tmp_path / "Horn70.txt")
        maps = cst_beam_maps(tmp_path, [70.0], nside=self.NSIDE)
        theta, _ = hp.pix2ang(self.NSIDE, int(np.argmax(maps[0])))
        assert np.rad2deg(theta) < 5.0

    def test_phi_sense_flips_the_azimuthal_structure(self, tmp_path):
        """The handedness is not derivable from the file, so it is a knob. This
        pins that the knob does the RIGHT thing — a reflection about phi = 0,
        not a relabelling."""
        write_cst(tmp_path / "Horn70.txt", az_depth=0.9)
        ccw = cst_beam_maps(tmp_path, [70.0], nside=self.NSIDE, phi_sense="ccw")[0]
        cw = cst_beam_maps(tmp_path, [70.0], nside=self.NSIDE, phi_sense="cw")[0]
        assert not np.allclose(ccw, cw)

        theta, phi = hp.pix2ang(self.NSIDE, np.arange(hp.nside2npix(self.NSIDE)))
        mirrored = hp.ang2pix(self.NSIDE, theta, (2.0 * np.pi - phi) % (2.0 * np.pi))
        band = theta > np.deg2rad(10.0)   # the pole's pixels are their own mirror
        np.testing.assert_allclose(ccw[mirrored][band], cw[band], rtol=2e-2)

    def test_phi0_rotates_the_pattern(self, tmp_path):
        write_cst(tmp_path / "Horn70.txt", az_depth=0.9)
        base = cst_beam_maps(tmp_path, [70.0], nside=self.NSIDE)[0]
        turned = cst_beam_maps(tmp_path, [70.0], nside=self.NSIDE, phi0_deg=180.0)[0]
        assert not np.allclose(base, turned)
        assert abs(base.sum() - turned.sum()) / base.sum() < 1e-3

    def test_an_unknown_phi_sense_is_refused(self, tmp_path):
        write_cst(tmp_path / "Horn70.txt")
        with pytest.raises(ValueError, match="phi_sense"):
            cst_beam_maps(tmp_path, [70.0], nside=8, phi_sense="widdershins")


class TestFrequencyInterpolation:
    def test_a_frequency_on_the_grid_reproduces_its_own_file(self, tmp_path):
        write_cst(tmp_path / "Horn60.txt", sigma_deg=20.0)
        write_cst(tmp_path / "Horn80.txt", sigma_deg=12.0)
        both = cst_beam_maps(tmp_path, [60.0, 80.0], nside=16)
        only60 = cst_beam_maps(tmp_path, [60.0], nside=16)[0]
        np.testing.assert_allclose(both[0], only60, rtol=1e-12)
        assert not np.allclose(both[0], both[1]), "the two files must differ"

    def test_a_midpoint_is_the_average_of_its_neighbours(self, tmp_path):
        write_cst(tmp_path / "Horn60.txt", sigma_deg=20.0)
        write_cst(tmp_path / "Horn80.txt", sigma_deg=12.0)
        ends = cst_beam_maps(tmp_path, [60.0, 80.0], nside=16)
        mid = cst_beam_maps(tmp_path, [70.0], nside=16)[0]
        np.testing.assert_allclose(mid, 0.5 * (ends[0] + ends[1]), rtol=1e-10)

    def test_extrapolation_beyond_the_simulated_band_is_refused(self, tmp_path):
        write_cst(tmp_path / "Horn60.txt")
        write_cst(tmp_path / "Horn80.txt")
        with pytest.raises(ValueError, match="covers only"):
            cst_beam_maps(tmp_path, [55.0], nside=8)
        with pytest.raises(ValueError, match="covers only"):
            cst_beam_maps(tmp_path, [85.0], nside=8)


class TestBeamFunc:
    """The TODSim contract: ``beam_func(freq=..., nside=...) -> (npix,)``."""

    def test_it_matches_cst_beam_maps(self, tmp_path):
        write_cst(tmp_path / "Horn60.txt", sigma_deg=20.0)
        write_cst(tmp_path / "Horn80.txt", sigma_deg=12.0)
        beam_func = cst_beam_func(tmp_path)
        np.testing.assert_allclose(
            beam_func(freq=72.5, nside=16),
            cst_beam_maps(tmp_path, [72.5], nside=16)[0],
            rtol=1e-12,
        )

    def test_it_is_chromatic(self, tmp_path):
        write_cst(tmp_path / "Horn60.txt", sigma_deg=20.0)
        write_cst(tmp_path / "Horn80.txt", sigma_deg=12.0)
        beam_func = cst_beam_func(tmp_path)
        assert not np.allclose(
            beam_func(freq=60.0, nside=16), beam_func(freq=80.0, nside=16)
        )

    def test_configuration_errors_surface_at_construction(self, tmp_path):
        """Not at the first channel, halfway through a simulation."""
        write_cst(tmp_path / "Horn70.txt")
        with pytest.raises(ValueError, match="phi_sense"):
            cst_beam_func(tmp_path, phi_sense="widdershins")
        with pytest.raises(ValueError, match="No CST exports"):
            cst_beam_func(tmp_path / "nothing-here")

    def test_files_are_parsed_once_across_calls(self, tmp_path, monkeypatch):
        """A sweep over channels must not re-parse a 65k-row export per
        channel; the cache is the reason this wrapper exists at all."""
        write_cst(tmp_path / "Horn60.txt", sigma_deg=20.0)
        write_cst(tmp_path / "Horn80.txt", sigma_deg=12.0)

        from limTOD import cstbeam

        calls = []
        original = cstbeam.read_cst_farfield

        def counted(path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(cstbeam, "read_cst_farfield", counted)
        beam_func = cst_beam_func(tmp_path)
        for freq in np.linspace(60.0, 80.0, 25):
            beam_func(freq=float(freq), nside=8)
        assert len(calls) == 2, f"parsed {len(calls)} times for 2 files"
