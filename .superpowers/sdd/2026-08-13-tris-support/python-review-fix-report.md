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

## Round 2: accurate input contracts and private typing helpers

Review base: `3785a63`

### Scope

- Split the former dimension-ambiguous `_NumericArrayLike` into private real
  scalar, vector, and matrix input aliases. The matrix alias explicitly accepts
  nested sequences.
- Applied the real-scalar input alias to public dataclass constructor fields,
  beam/geometry scalars, the returned beam callable, and optional fit scalars.
- Added a tracked positive typing fixture covering nested-list design matrices,
  NumPy real scalars, and zero-dimensional arrays.
- Imported the six newly needed helper names only through private module aliases,
  while retaining the pre-existing `Optional`, `Tuple`, and `Union` star-import
  surface for compatibility.

### RED: positive input fixture

Command:

```text
mypy --ignore-missing-imports --disallow-untyped-defs limTOD/tris.py tests/typecheck_tris_inputs.py
```

Output summary:

```text
tests/typecheck_tris_inputs.py:21: error: Argument "negative_k" to "AsymmetricUncertainty" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:25: error: Argument "effective_frequency_mhz" to "TRISRing" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:26: error: Argument "bandwidth_mhz" to "TRISRing" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:32: error: Argument "declination_label_deg" to "TRISRing" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:35: error: Argument "nominal_frequency_mhz" to "TRISPointSet" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:37: error: Argument "bandwidth_mhz" to "TRISPointSet" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:42: error: Argument "zero_level_uncertainty_k" to "TRISPointSet" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:49: error: Argument "tolerance" to "TRISRankDiagnostic" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:51: error: Argument "condition_number" to "TRISRankDiagnostic" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:56: error: List item 0 has incompatible type "list[float]"; expected "float"  [list-item]
tests/typecheck_tris_inputs.py:56: error: List item 1 has incompatible type "list[float]"; expected "float"  [list-item]
tests/typecheck_tris_inputs.py:56: error: List item 2 has incompatible type "list[float]"; expected "float"  [list-item]
tests/typecheck_tris_inputs.py:57: error: Argument "uncertainty_floor_k" to "fit_tris_linear_model" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float | None"  [arg-type]
tests/typecheck_tris_inputs.py:59: error: Argument "rank_rtol" to "fit_tris_linear_model" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float | None"  [arg-type]
tests/typecheck_tris_inputs.py:62: error: Argument "fwhm_e_deg" to "approximate_tris_gaussian_beam_map" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:65: error: Argument "fwhm_e_deg" to "tris_beam_func" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:67: error: Argument "freq" to "__call__" of "_BeamFunc" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
tests/typecheck_tris_inputs.py:70: error: Argument "latitude_deg" to "tris_zenith_geometry" has incompatible type "ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
Found 18 errors in 1 file (checked 2 source files)
```

Exit code: 1.

### RED: star-import surface

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py -k star_import_does_not_expose_private_typing_helpers
```

Output summary:

```text
F                                                                        [100%]
E       AssertionError: assert False
E        +  where False = <built-in method isdisjoint ...>({'AsymmetricUncertainty': ..., 'Dict': typing.Dict, 'List': typing.List, 'Optional': typing.Optional, ...})
1 failed, 56 deselected in 0.90s
```

Exit code: 1. The test exposed `Real`, `PathLike`, `Dict`, `List`, `Protocol`,
and `Sequence`. A follow-up characterization also proved that the pre-existing
`Optional`, `Tuple`, and `Union` names had to be retained for compatibility.

### GREEN: focused TRIS suite

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py
```

Output:

```text
.........................................................                [100%]
57 passed in 1.33s
```

Exit code: 0.

### GREEN: strict type check and positive fixture

Command:

```text
mypy --ignore-missing-imports --disallow-untyped-defs limTOD/tris.py tests/typecheck_tris_inputs.py
```

Output:

```text
Success: no issues found in 2 source files
```

Exit code: 0.

### GREEN: format check

Command:

```text
black --check --fast limTOD/tris.py tests/test_tris.py tests/typecheck_tris_inputs.py
```

Output:

```text
All done! ✨ 🍰 ✨
3 files would be left unchanged.
```

Exit code: 0.

### GREEN: Python 3.8 grammar parse

Command:

```text
python -c 'import ast
from pathlib import Path
for name in ("limTOD/tris.py", "tests/typecheck_tris_inputs.py"):
    path = Path(name)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("Python 3.8 grammar parse passed for 2 files")'
```

Output:

```text
Python 3.8 grammar parse passed for 2 files
```

Exit code: 0.

### GREEN: whitespace checks

Commands:

```text
git diff --check 3785a63
git diff --check 0fd227c..HEAD
```

Output: empty for both commands. Exit code: 0 for both commands. The exact
baseline-to-new-HEAD command is rerun after the round-2 commit.

Post-commit command:

```text
git diff --check 0fd227c..HEAD
```

Output: empty. Exit code: 0.

### GREEN: full suite

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q
```

Output:

```text
........................................................................ [ 26%]
........................................................................ [ 52%]
........................................................................ [ 78%]
............................................................             [100%]
=============================== warnings summary ===============================
tests/test_beam_orientation.py: 16 warnings
  /private/tmp/limtod-tris-support/limTOD/simulator.py:397: UserWarning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.
    psi_rad, theta_rad, phi_rad = zyz_of_pointing(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
276 passed, 2 skipped, 16 warnings in 77.45s (0:01:17)
```

Exit code: 0.

### Round-2 files and concerns

- `limTOD/tris.py`
- `tests/test_tris.py`
- `tests/typecheck_tris_inputs.py`
- `.superpowers/sdd/2026-08-13-tris-support/python-review-fix-report.md`

The full suite continues to emit only the 16 existing gimbal-lock warnings.
Python 3.8 grammar was checked explicitly; runtime verification remained on
Python 3.12 because a Python 3.8 interpreter was not invoked.

## Round 3: normalized stored-attribute types

Review base: `a1f020d`

### Scope

- Extended the positive typing fixture so every normalized real scalar from
  `AsymmetricUncertainty`, `TRISRing`, `TRISPointSet`, `TRISZenithGeometry`,
  and `TRISRankDiagnostic` is accepted by a function requiring `float`.
- Extended the fixture so normalized arrays from all array-bearing TRIS public
  models are accepted by a function requiring `np.ndarray`; list-backed
  principal-plane inputs are included.
- Separated constructor input contracts from stored-field contracts using
  frozen `dataclass(init=False)` models with explicit Python-3.8-compatible
  constructors. Constructors retain the existing broad inputs and exact public
  parameter names/order/defaults; stored fields are honestly annotated as
  built-in `float`, built-in `int`, or `np.ndarray` after `__post_init__`.
- Kept dataclass-generated repr, equality, and frozen behavior, one public
  storage field per value, and all existing runtime normalization/validation.

### RED: stored attributes and broad array constructors

Command:

```text
mypy --ignore-missing-imports --disallow-untyped-defs limTOD/tris.py tests/typecheck_tris_inputs.py
```

Output:

```text
tests/typecheck_tris_inputs.py:54: error: Argument "angle_deg" to "TRISPrincipalPlaneCuts" has incompatible type "list[float]"; expected "ndarray[tuple[Any, ...], dtype[Any]]"  [arg-type]
tests/typecheck_tris_inputs.py:55: error: Argument "h_plane_db" to "TRISPrincipalPlaneCuts" has incompatible type "list[float]"; expected "ndarray[tuple[Any, ...], dtype[Any]]"  [arg-type]
tests/typecheck_tris_inputs.py:56: error: Argument "e_plane_db" to "TRISPrincipalPlaneCuts" has incompatible type "list[float]"; expected "ndarray[tuple[Any, ...], dtype[Any]]"  [arg-type]
tests/typecheck_tris_inputs.py:102: error: Argument 1 to "_requires_float" has incompatible type "int | float | integer[Any] | floating[Any] | ndarray[tuple[Any, ...], dtype[Any]]"; expected "float"  [arg-type]
Found 4 errors in 1 file (checked 2 source files)
```

Exit code: 1. The scalar tuple deliberately combines normalized attributes from
all five affected public models, so mypy reports the shared broad union once.

### GREEN: strict type check

Command:

```text
mypy --ignore-missing-imports --disallow-untyped-defs limTOD/tris.py tests/typecheck_tris_inputs.py
```

Output:

```text
Success: no issues found in 2 source files
```

Exit code: 0.

### GREEN: execute positive fixture

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python tests/typecheck_tris_inputs.py
```

Output:

```text
[Zhengs-Mac-Studio][[25744,0],0][btl_tcp_component.c:1021:mca_btl_tcp_component_create_listen] bind() failed: Operation not permitted (1)
```

Exit code: 0. This is a non-fatal local OpenMPI socket warning emitted during
imports; all fixture construction, fitting, beam, and geometry calls completed.

### GREEN: focused TRIS suite

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py
```

Output:

```text
.........................................................                [100%]
57 passed in 1.34s
```

Exit code: 0.

### GREEN: format check

Command:

```text
black --check --fast limTOD/tris.py tests/test_tris.py tests/typecheck_tris_inputs.py
```

Output:

```text
All done! ✨ 🍰 ✨
3 files would be left unchanged.
```

Exit code: 0.

### GREEN: Python 3.8 grammar parse

Command:

```text
python -c 'import ast
from pathlib import Path
for name in ("limTOD/tris.py", "tests/typecheck_tris_inputs.py"):
    path = Path(name)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("Python 3.8 grammar parse passed for 2 files")'
```

Output:

```text
Python 3.8 grammar parse passed for 2 files
```

Exit code: 0.

### GREEN: whitespace checks

Commands:

```text
git diff --check a1f020d
git diff --check 0fd227c..HEAD
```

Output: empty for both commands. Exit code: 0 for both commands. The exact
baseline-to-new-HEAD command is rerun after the round-3 commit.

Post-commit command:

```text
git diff --check 0fd227c..HEAD
```

Output: empty. Exit code: 0.

### GREEN: full suite

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q
```

Output:

```text
........................................................................ [ 26%]
........................................................................ [ 52%]
........................................................................ [ 78%]
............................................................             [100%]
=============================== warnings summary ===============================
tests/test_beam_orientation.py: 16 warnings
  /private/tmp/limtod-tris-support/limTOD/simulator.py:397: UserWarning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.
    psi_rad, theta_rad, phi_rad = zyz_of_pointing(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
276 passed, 2 skipped, 16 warnings in 78.04s (0:01:18)
```

Exit code: 0.

### Round-3 files and concerns

- `limTOD/tris.py`
- `tests/typecheck_tris_inputs.py`
- `.superpowers/sdd/2026-08-13-tris-support/python-review-fix-report.md`

The explicit constructors add boilerplate but avoid duplicate state and are the
only way used here to give mypy broad constructor inputs plus narrow normalized
stored attributes on Python 3.8. The full suite still emits the 16 existing
gimbal-lock warnings. Executing the fixture emits the non-fatal local OpenMPI
socket warning quoted above. Runtime verification used Python 3.12; both edited
Python files pass Python 3.8 grammar parsing.
