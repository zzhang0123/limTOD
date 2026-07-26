# limtod_jax

The pure-JAX port. All public entry points are re-exported at the
package level (`import limtod_jax as ltj`); they are documented here by
defining module. Requires the `[jax]` extra.

## Core sky→TOD chain

```{eval-rst}
.. automodule:: limtod_jax.core
   :members:
```

## Spherical harmonics and HEALPix

```{eval-rst}
.. automodule:: limtod_jax.alm
   :members:

.. automodule:: limtod_jax.hpx
   :members:
```

## Rotations

```{eval-rst}
.. automodule:: limtod_jax.angles
   :members:

.. automodule:: limtod_jax.wigner
   :members:
```

## Map-space projection

```{eval-rst}
.. automodule:: limtod_jax.projection
   :members:
```

## Drift-scan m-modes

```{eval-rst}
.. automodule:: limtod_jax.driftscan
   :members:
```
