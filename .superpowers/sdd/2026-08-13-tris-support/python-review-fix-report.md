# Python review fix report

Date: 2026-08-13

Branch: `codex/tris-support`

Baseline: `0fd227c`

## Scope

The review fix:

- adds one strict real-scalar coercion boundary that rejects Python/NumPy
  booleans, complex and string values, non-scalar arrays, and values that
  cannot be represented as finite built-in floats;
- accepts Python/NumPy real numeric scalars and zero-dimensional real arrays;
- stores detached built-in floats for public scalar metadata in frozen TRIS
  dataclasses, including symmetric/asymmetric uncertainties, frequencies,
  bandwidths, declination/latitude, and rank-diagnostic real scalars;
- adds Python-3.8-compatible annotations to every function, method, property,
  and `__post_init__` in `limTOD/tris.py`, with private aliases and the
  keyword-only beam callable `Protocol`;
- adds a heteroskedastic, nonconstant, full-rank correlated-GLS oracle test;
- removes the extra blank line at EOF from the two internal design documents.

## RED evidence

### Missing type contracts

Command:

```text
mypy --ignore-missing-imports --disallow-untyped-defs limTOD/tris.py
```

Output:

```text
limTOD/tris.py:24: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:36: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:46: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:51: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:64: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:87: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:127: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:161: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:175: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:180: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:185: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:196: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:204: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:225: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:236: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:245: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:266: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:272: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:296: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:314: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:358: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:395: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:445: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:470: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:503: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:512: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:536: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:554: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:582: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:623: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:646: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:651: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:656: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:661: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:666: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:671: error: Function is missing a return type annotation  [no-untyped-def]
limTOD/tris.py:676: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:705: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:718: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:731: error: Function is missing a type annotation  [no-untyped-def]
limTOD/tris.py:830: error: Function is missing a type annotation  [no-untyped-def]
Found 41 errors in 1 file (checked 1 source file)
```

Exit code: 1.

### Scalar ownership and validation

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py -k 'frozen_scalar_metadata or real_scalar_metadata or point_set_rejects_boolean or correlated_gls'
```

Output summary:

```text
FFFFFFFFFFFF.                                                            [100%]
12 failed, 1 passed, 42 deselected, 1 warning in 1.00s
```

The exact failures showed:

```text
ValueError: declination_label_deg must be finite
Failed: DID NOT RAISE <class 'ValueError'>
TypeError: must be real number, not complex
TypeError: must be real number, not str
TypeError: must be real number, not list
```

The failures covered mutable zero-dimensional metadata, Python/NumPy boolean
acceptance, complex/string inputs, list inputs, one-dimensional size-one
arrays, and boolean values in each point-set physical scalar field. Exit code:
1. The GLS test passed against the unmutated implementation, so its sensitivity
was established separately below.

### Correlated-GLS oracle mutation check

To prove that the new oracle catches the intended defect, the production
common-mode covariance addition was temporarily replaced by a no-op. The
production line was restored immediately after this run; the committed
implementation remains Cholesky plus SVD.

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py -k correlated_gls_matches_direct_normal_equation_oracle
```

Output:

```text
F                                                                        [100%]
E       AssertionError:
E       Not equal to tolerance rtol=2e-12, atol=2e-12
E
E       Mismatched elements: 1 / 9 (11.1%)
E       Max absolute difference among violations: 0.0529
E       Max relative difference among violations: 0.87740425
E        ACTUAL: array([[ 0.007391, -0.00089 , -0.000698],
E              [-0.00089 ,  0.014916,  0.004004],
E              [-0.000698,  0.004004,  0.00486 ]])
E        DESIRED: array([[ 0.060291, -0.00089 , -0.000698],
E              [-0.00089 ,  0.014916,  0.004004],
E              [-0.000698,  0.004004,  0.00486 ]])
1 failed, 54 deselected in 0.84s
```

Exit code: 1.

### Overflow normalization edge

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py -k 'real_scalar_metadata_rejects_boolean_and_non_scalar_values'
```

Output:

```text
......F                                                                  [100%]
E       OverflowError: int too large to convert to float
1 failed, 6 passed, 49 deselected in 0.86s
```

Exit code: 1. The coercion boundary now translates this conversion failure to
the same field-specific `ValueError` contract as other invalid scalars.

## GREEN evidence

### Overflow regression

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py -k 'real_scalar_metadata_rejects_boolean_and_non_scalar_values'
```

Output:

```text
.......                                                                  [100%]
7 passed, 49 deselected in 0.82s
```

Exit code: 0.

### Requested focused TRIS suite

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py
```

Output:

```text
........................................................                 [100%]
56 passed in 1.31s
```

Exit code: 0.

### Type check

Command:

```text
mypy --ignore-missing-imports --disallow-untyped-defs limTOD/tris.py
```

Output:

```text
Success: no issues found in 1 source file
```

Exit code: 0.

### Format check

Command:

```text
black --check --fast limTOD/tris.py tests/test_tris.py
```

Output:

```text
All done! ✨ 🍰 ✨
2 files would be left unchanged.
```

Exit code: 0.

### Whitespace check before commit

The equivalent baseline-to-working-tree check was used before the commit so
that the uncommitted EOF corrections were included.

Command:

```text
git diff --check 0fd227c
```

Output: empty. Exit code: 0.

Post-commit command:

```text
git diff --check 0fd227c..HEAD
```

Output: empty. Exit code: 0.

### Full suite against final code

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q
```

Output:

```text
........................................................................ [ 26%]
........................................................................ [ 52%]
........................................................................ [ 78%]
...........................................................              [100%]
=============================== warnings summary ===============================
tests/test_beam_orientation.py: 16 warnings
  /private/tmp/limtod-tris-support/limTOD/simulator.py:397: UserWarning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.
    psi_rad, theta_rad, phi_rad = zyz_of_pointing(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
275 passed, 2 skipped, 16 warnings in 77.76s (0:01:17)
```

Exit code: 0.

## Files

- `limTOD/tris.py`
- `tests/test_tris.py`
- `docs/superpowers/plans/2026-08-13-tris-support.md`
- `docs/superpowers/specs/2026-08-13-tris-support-design.md`
- `.superpowers/sdd/2026-08-13-tris-support/python-review-fix-report.md`

## Self-review and concerns

- No production direct-inverse GLS path was added; `np.linalg.inv` appears only
  in the independent oracle test.
- No public symbol or dependency was added; all new aliases and the beam
  callable protocol are private.
- The only full-suite warnings are the 16 existing gimbal-lock warnings from
  `tests/test_beam_orientation.py`.
- Verification ran under Python 3.12. The added annotations deliberately avoid
  PEP 604 unions and Python-3.9 built-in collection generics and use
  `typing.Protocol`, which is available in Python 3.8; a Python 3.8 interpreter
  was not invoked in this environment.
