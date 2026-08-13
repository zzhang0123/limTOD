# Task 4 report — compact TRIS documentation

## Files changed

- `docs/tris.md` — canonical compact report: project boundary, archive provenance,
  uncertainty and beam conventions, limTOD translation, identifiability,
  offline workflow, and no-go claims.
- `docs/api/tris.md` — thin generated reference for every public
  `limTOD.tris` API.
- `docs/index.md` and `docs/api/index.md` — TRIS guide and API navigation.
- `README.md` — compact TRIS capability statement and documentation-table link.
- `CHANGELOG.md` — Unreleased documentation entry.

## Documentation build

Command run exactly:

```bash
PYTHONPATH=. sphinx-build -W --keep-going docs docs/_build
```

Result: Sphinx rendered `docs/tris.md` and `docs/api/tris.md` with no
TRIS-document warnings, but the repository-wide invocation exited non-zero
with 29 pre-existing warnings treated as errors. They are outside this task's
owned files: missing optional `s2fft` causes existing `limtod_jax` autodoc
imports to fail; offline intersphinx inventory fetches fail; and existing
duplicate-label, non-consecutive-header, and orphan-document warnings occur in
`cstbeam` and `docs/superpowers/`. No build issue was introduced or fixed
outside the task scope.

## Checks performed

- Read the task brief, the binding design/source ledger, local archive headers,
  checksums, and `limTOD/tris.py` signatures.
- Used the documented local files in a direct offline smoke test: ring and
  beam-cut readers, Gaussian beam builder, zenith geometry, Fourier design,
  and the 600-MHz rank-gated fit. It passed.
- Ran `git diff --check`; it passed.
- Checked both new pages are present in their corresponding toctrees.
- Confirmed the report records the source product counts, retrieval date,
  checksums, frequency metadata, uncertainty roles, approximation status, and
  unsupported claims.

## Self-review

The guide is deliberately the sole detailed convention report; the README,
changelog, and API page link to it rather than duplicate its scientific
claims. It explicitly separates archive/paper facts, limTOD physical
interpretation, the Gaussian approximation, and unknown or unsupported
quantities. The example calls the exact implemented APIs, performs no network
access, uses the physical-latitude default and `selfrot=-7` geometry once,
recommends beam normalization for brightness convolution, and makes its
zero-floor and symmetric 600-MHz common-mode choices explicit. It never
claims a measured 2D beam, 3° resolution, a 2.5-GHz ring, or a unique sky map.

## Concerns

The requested strict full-document build cannot be green in this environment
without resolving unrelated optional-JAX and pre-existing documentation
warnings. The external LAMBDA and arXiv links are included, but the build
environment has no DNS access for intersphinx inventories.
