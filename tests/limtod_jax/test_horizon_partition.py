"""The horizon as a PARTITION of the beam's solid angle, and the cheap cut.

``horizon_weights`` is a mask — what to multiply a beam by. ``f_sky`` is a
partition — how the beam's solid angle divides. They are different objects and
the difference is measurable, so this suite pins it two ways:

1. against a directly computable reference — a sky map with the ground painted
   in, observed at a latitude where the local horizon coincides with the
   celestial equator and stops moving with LST, so the answer is not a matter
   of opinion;
2. against ``horizon_masked_beam_alm``, the rotation-based path, which must
   describe the same instrument.

Every lock is numerical. The conventions here (which chart has which pole,
where the horizon ring falls) are exactly the kind that read fine on paper and
cost kelvin in practice.
"""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from limtod_jax.core import rotate_alm
from limtod_jax.driftscan import (
    driftscan_tod,
    horizon_beam_fraction,
    horizon_masked_beam_alm,
    horizon_partition_weights,
    horizon_truncated_beam,
    horizon_weights,
)
from limtod_jax.hpx import alm2map, map2alm_iter, map2alm_quad, ones_quadrature_alm

T_SKY, T_GROUND = 3000.0, 290.0


def gaussian_beam(nside, sigma_deg=35.0, floor=0.02):
    """A main lobe plus a sidelobe floor, so the below-horizon share is real."""
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return jnp.asarray(np.exp(-0.5 * (theta / np.deg2rad(sigma_deg)) ** 2) + floor)


class TestPartitionWeights:
    @pytest.mark.parametrize("nside", [4, 8, 16, 32])
    def test_the_horizon_ring_is_counted_half(self, nside):
        """``horizon_weights``' hard cut is a strict ``el > 0``, so the ring of
        pixels centred exactly on the horizon is dropped. Those pixels are half
        sky and half ground, and there are 4*nside of them."""
        partition = horizon_partition_weights(nside)
        theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        on_horizon = np.isclose(theta, np.pi / 2)

        assert on_horizon.sum() == 4 * nside
        np.testing.assert_array_equal(np.unique(partition), [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(partition[on_horizon], 0.5)
        np.testing.assert_array_equal(
            partition[theta < np.pi / 2 - 1e-12], 1.0
        )
        np.testing.assert_array_equal(
            partition[theta > np.pi / 2 + 1e-12], 0.0
        )

    @pytest.mark.parametrize("nside", [4, 8, 16])
    def test_it_partitions_the_sphere_exactly(self, nside):
        """The property that makes it a partition and the mask not one: the
        weights and their complement sum to every pixel, so an isotropic beam
        splits exactly in half at any resolution."""
        partition = horizon_partition_weights(nside)
        npix = hp.nside2npix(nside)
        assert abs(partition.sum() / npix - 0.5) < 1e-12
        assert abs(horizon_weights(nside, 0.0).sum() / npix - 0.5) > 1e-3

    def test_it_takes_no_apodization(self):
        """A tapered region does not partition a sphere; apodization belongs to
        the mask. The signature says so by having no knob."""
        import inspect

        assert "apod_deg" not in inspect.signature(horizon_partition_weights).parameters


class TestTruncatedBeam:
    NSIDE, LMAX = 16, 47

    def test_it_zeroes_the_beam_below_the_horizon(self):
        truncated, _ = horizon_truncated_beam(gaussian_beam(self.NSIDE), nside=self.NSIDE)
        theta, _ = hp.pix2ang(self.NSIDE, np.arange(hp.nside2npix(self.NSIDE)))
        assert jnp.all(truncated[theta > np.pi / 2] == 0.0)
        assert jnp.any(truncated[theta < np.pi / 2] > 0.0)

    def test_an_isotropic_beam_splits_in_half(self):
        _, fraction = horizon_truncated_beam(
            jnp.ones(hp.nside2npix(self.NSIDE)), nside=self.NSIDE
        )
        assert abs(float(fraction) - 0.5) < 1e-12

    def test_apodization_tapers_the_map_but_not_the_fraction(self):
        beam = gaussian_beam(self.NSIDE)
        sharp_map, sharp_f = horizon_truncated_beam(beam, nside=self.NSIDE, apod_deg=0.0)
        soft_map, soft_f = horizon_truncated_beam(beam, nside=self.NSIDE, apod_deg=5.0)
        assert jnp.allclose(sharp_f, soft_f)
        assert not jnp.allclose(sharp_map, soft_map)

    def test_a_tilted_pointing_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="zenith"):
            horizon_truncated_beam(gaussian_beam(self.NSIDE), nside=self.NSIDE, el_deg=45.0)

    def test_a_bad_map_length_is_refused(self):
        with pytest.raises(ValueError, match="12"):
            horizon_truncated_beam(jnp.ones(100), nside=self.NSIDE)

    def test_it_batches_and_differentiates(self):
        beam = gaussian_beam(self.NSIDE)
        stacked = jnp.stack([beam, 2.0 * beam])
        maps, fractions = horizon_truncated_beam(stacked, nside=self.NSIDE)
        assert maps.shape == stacked.shape and fractions.shape == (2,)
        # scale-invariant, so the gradient is the interesting check, not the value
        grad = jax.jit(jax.grad(
            lambda b: horizon_truncated_beam(b, nside=self.NSIDE)[1]
        ))(beam)
        assert jnp.all(jnp.isfinite(grad)) and jnp.any(grad != 0.0)

    def test_it_agrees_with_the_alm_masking_path(self):
        """Two routes to one instrument: truncate the map once, or mask the
        alms. The residual is the alm->map->alm round trip the masking path
        takes BEFORE it masks, which this one does not."""
        beam = gaussian_beam(self.NSIDE)
        alm = map2alm_iter(beam, nside=self.NSIDE, lmax=self.LMAX)

        by_alm = horizon_masked_beam_alm(
            alm, 0.0, 90.0, nside=self.NSIDE, lmax=self.LMAX
        )
        truncated, _ = horizon_truncated_beam(beam, nside=self.NSIDE)
        by_map = map2alm_iter(truncated, nside=self.NSIDE, lmax=self.LMAX)

        ones = ones_quadrature_alm(nside=self.NSIDE, lmax=self.LMAX)
        from limtod_jax.alm import alm_dot

        rel = abs(
            float(alm_dot(by_alm, ones, lmax=self.LMAX))
            / float(alm_dot(by_map, ones, lmax=self.LMAX))
            - 1.0
        )
        assert rel < 1e-3

    def test_the_two_fraction_routes_agree(self):
        beam = gaussian_beam(self.NSIDE)
        _, from_map = horizon_truncated_beam(beam, nside=self.NSIDE)
        from_alm = horizon_beam_fraction(
            map2alm_iter(beam, nside=self.NSIDE, lmax=self.LMAX),
            0.0, 90.0, nside=self.NSIDE, lmax=self.LMAX,
        )
        assert abs(float(from_map) - float(from_alm)) < 1e-4


class TestBeamFraction:
    NSIDE, LMAX = 16, 47

    def test_azimuth_and_self_rotation_do_not_move_it_at_zenith(self):
        """The invariance that makes the map path exact at zenith: this chart's
        pole is the zenith and a beam-local map's pole is the boresight, so at
        el = 90 they coincide and the two charts differ only by a rotation ABOUT
        that shared pole -- which a pure-elevation partition cannot see.

        Exact for the continuous function; here it is 3e-6, because this route
        rotates the ALMS and re-synthesizes, and a HEALPix grid does not sample
        azimuth identically ring by ring. That residual is the discretization,
        not a convention error -- and it is one more small reason to prefer
        ``horizon_truncated_beam`` at zenith, which rotates nothing and is
        invariant to machine precision."""
        alm = map2alm_iter(gaussian_beam(self.NSIDE), nside=self.NSIDE, lmax=self.LMAX)
        base = horizon_beam_fraction(alm, 0.0, 90.0, nside=self.NSIDE, lmax=self.LMAX)
        for az, selfrot in ((37.0, 0.0), (0.0, 121.0), (210.0, -45.0)):
            spun = horizon_beam_fraction(
                alm, az, 90.0, selfrot, nside=self.NSIDE, lmax=self.LMAX
            )
            assert abs(float(spun) - float(base)) < 1e-5

    def test_the_map_route_is_invariant_to_machine_precision(self):
        """The same statement for the rotation-free path: azimuth and
        self-rotation are not even arguments there, so the invariance is
        structural rather than numerical."""
        import inspect

        params = inspect.signature(horizon_truncated_beam).parameters
        assert "az_deg" not in params and "selfrot_deg" not in params

    def test_tilting_the_pointing_lowers_it(self):
        """A beam tipped towards the horizon must lose solid angle to the
        ground. Cheap, and it would catch a rotation applied backwards."""
        alm = map2alm_iter(gaussian_beam(self.NSIDE), nside=self.NSIDE, lmax=self.LMAX)
        zenith = float(horizon_beam_fraction(alm, 0.0, 90.0, nside=self.NSIDE, lmax=self.LMAX))
        tilted = float(horizon_beam_fraction(alm, 0.0, 40.0, nside=self.NSIDE, lmax=self.LMAX))
        assert tilted < zenith - 0.05

    def test_it_is_not_the_masked_beams_own_integral(self):
        """The distinction the docstring rests on. ``map2alm`` of a sharply cut
        map does not preserve the mean, so the band-limited masked beam's
        solid-angle integral is a DIFFERENT number -- and using it as f_sky is
        what leaves kelvin on the table (see TestClosure)."""
        from limtod_jax.alm import alm_dot

        alm = map2alm_iter(gaussian_beam(self.NSIDE), nside=self.NSIDE, lmax=self.LMAX)
        masked = horizon_masked_beam_alm(alm, 0.0, 90.0, nside=self.NSIDE, lmax=self.LMAX)
        ones = ones_quadrature_alm(nside=self.NSIDE, lmax=self.LMAX)
        harmonic = float(alm_dot(masked, ones, lmax=self.LMAX)
                         / alm_dot(alm, ones, lmax=self.LMAX))
        partition = float(
            horizon_beam_fraction(alm, 0.0, 90.0, nside=self.NSIDE, lmax=self.LMAX)
        )
        assert abs(harmonic - partition) > 1e-3


class TestClosure:
    """The reference that is not a matter of opinion.

    At latitude 90 with a zenith pointing the local horizon coincides with the
    celestial equator and does not move with LST, so a celestial sky map can
    hold the ground and the projector can simply be asked for the answer.
    """

    @staticmethod
    def _tod(beam_map, sky_map, *, nside, lmax, n_time=12):
        from limtod_jax.driftscan import driftscan_tod  # noqa: F401  (re-export lock)

        lst = jnp.deg2rad(360.0 * jnp.arange(n_time) / n_time)
        beam_alm = map2alm_iter(beam_map, nside=nside, lmax=lmax)
        # lat = 90, az = 0, el = 90: the beam-local frame IS the celestial one
        # up to a rotation about the pole, which the drift phases supply.
        sky_alm = map2alm_quad(sky_map, nside=nside, lmax=lmax)
        return driftscan_tod(beam_alm, sky_alm, lst, lmax=lmax, normalize=True,
                             ones_alm=ones_quadrature_alm(nside=nside, lmax=lmax))

    @pytest.mark.parametrize("nside", [8, 16])
    def test_the_split_reproduces_a_painted_ground_sky(self, nside):
        lmax = 3 * nside - 1
        beam = gaussian_beam(nside)
        theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

        # The horizon ring is half sky and half ground here too; a reference
        # that gave it entirely to one side would be testing a convention.
        painted = np.where(theta < np.pi / 2 - 1e-9, T_SKY, T_GROUND)
        painted[np.isclose(theta, np.pi / 2)] = 0.5 * (T_SKY + T_GROUND)
        uniform = jnp.full(hp.nside2npix(nside), T_SKY)

        exact = self._tod(beam, jnp.asarray(painted), nside=nside, lmax=lmax)
        truncated, f_sky = horizon_truncated_beam(beam, nside=nside)
        visible = self._tod(truncated, uniform, nside=nside, lmax=lmax)
        modelled = f_sky * visible + (1.0 - f_sky) * T_GROUND

        assert float(jnp.max(jnp.abs(modelled - exact))) < 0.2

    def test_ignoring_the_split_is_a_two_hundred_kelvin_error(self):
        nside, lmax = 8, 23
        beam = gaussian_beam(nside)
        theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        painted = np.where(theta < np.pi / 2 - 1e-9, T_SKY, T_GROUND)
        painted[np.isclose(theta, np.pi / 2)] = 0.5 * (T_SKY + T_GROUND)

        exact = self._tod(beam, jnp.asarray(painted), nside=nside, lmax=lmax)
        naive = self._tod(beam, jnp.full(hp.nside2npix(nside), T_SKY),
                          nside=nside, lmax=lmax)
        assert float(jnp.mean(naive - exact)) > 150.0

    @pytest.mark.parametrize("ring_weight,sign", [(0.0, -1), (1.0, +1)])
    def test_the_one_sided_ring_conventions_are_symmetrically_wrong(
        self, ring_weight, sign
    ):
        """Counting the horizon ring as nothing or as all sky costs the same in
        opposite directions -- a miscounted ring, not a harmonic effect."""
        nside, lmax = 16, 47
        beam = gaussian_beam(nside)
        theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        on = np.isclose(theta, np.pi / 2)
        painted = np.where(theta < np.pi / 2 - 1e-9, T_SKY, T_GROUND)
        painted[on] = 0.5 * (T_SKY + T_GROUND)

        exact = float(jnp.mean(self._tod(beam, jnp.asarray(painted),
                                         nside=nside, lmax=lmax)))
        truncated, _ = horizon_truncated_beam(beam, nside=nside)
        visible = float(jnp.mean(self._tod(truncated, jnp.full(hp.nside2npix(nside), T_SKY),
                                           nside=nside, lmax=lmax)))
        skewed = np.where(on, ring_weight, horizon_weights(nside, 0.0))
        f = float((np.asarray(beam) * skewed).sum() / np.asarray(beam).sum())
        bias = f * visible + (1.0 - f) * T_GROUND - exact
        assert sign * bias > 1.0
