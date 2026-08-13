# Final review fix report

Date: 2026-08-13

Branch: `codex/tris-support`

Fix base: `b946ae5`

## Scope

- Enforced a machine-safe lower bound on caller-supplied `rank_rtol`, retained
  valid larger values in the diagnostic, and documented the behavior in both
  the function docstring and compact guide.
- Made `TRISPointSet.zero_level_uncertainty_k` use a scalar-only finite,
  non-negative validator while retaining asymmetric uncertainty support only
  for `TRISRing`.
- Used extended-precision log-response shifting before peak/sum normalization
  of Gaussian beams; `normalization="none"` retains its direct-exponential
  semantics.
- Preserved header source line numbers and added path/line context to malformed
  ring frequency and zero-level numeric diagnostics.
- Defined the exact 16-name binding-spec `limTOD.tris.__all__` without adding
  any package-root export.

## RED evidence

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py -k 'star_import or header_numeric or asymmetric_zero or narrow_gaussian or rank_tolerance'
```

Output summary:

```text
FFF.FFF.                                                                 [100%]
FAILED tests/test_tris.py::test_star_import_does_not_expose_private_typing_helpers
FAILED tests/test_tris.py::test_ring_header_numeric_errors_identify_source_and_line[...-1-frequency]
FAILED tests/test_tris.py::test_ring_header_numeric_errors_identify_source_and_line[...-2-zero-level uncertainty]
FAILED tests/test_tris.py::test_point_set_rejects_asymmetric_zero_level_uncertainty
FAILED tests/test_tris.py::test_narrow_gaussian_beam_normalizations_remain_finite
FAILED tests/test_tris.py::test_linear_fit_rejects_rank_tolerance_below_machine_safe_floor
6 failed, 2 passed, 55 deselected, 2 warnings in 0.90s
```

The essential failure evidence was:

```text
Extra items in the left set: 'Union', 're', 'Path', 'Optional', 'math'...
ValueError: ...bad-header.txt: ring file is missing its frequency header
ValueError: ...bad-header.txt: invalid zero-level uncertainty header: 'nopeK'
Failed: DID NOT RAISE <class 'ValueError'>  # asymmetric point-set value
ACTUAL: array([nan, nan, ..., nan])          # narrow normalized beam
Failed: DID NOT RAISE <class 'ValueError'>  # rank_rtol=1e-30
```

The narrow-beam RED also emitted the expected two divide-by-zero normalization
warnings from the old implementation. Exit code: 1.

## GREEN evidence

### Focused final-review regressions

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py -k 'star_import or header_numeric or asymmetric_zero or narrow_gaussian or rank_tolerance'
```

Output:

```text
........                                                                 [100%]
8 passed, 55 deselected in 0.82s
```

Exit code: 0.

### Complete focused TRIS suite

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q tests/test_tris.py
```

Output:

```text
...............................................................          [100%]
63 passed in 1.38s
```

Exit code: 0.

### Positive typing fixture and strict mypy

Commands:

```text
mypy --ignore-missing-imports --disallow-untyped-defs limTOD/tris.py tests/typecheck_tris_inputs.py
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python tests/typecheck_tris_inputs.py
```

Output:

```text
Success: no issues found in 2 source files
[Zhengs-Mac-Studio][[52102,0],0][btl_tcp_component.c:1021:mca_btl_tcp_component_create_listen] bind() failed: Operation not permitted (1)
```

Both commands exited 0. The second line is the existing non-fatal local OpenMPI
socket warning emitted during imports.

### Negative point-set typing probe

Command:

```text
mypy --ignore-missing-imports -c 'import numpy as np; from limTOD.tris import AsymmetricUncertainty, TRISPointSet; TRISPointSet(2500.0, 2427.8, 3.0, ("0h00m",), np.array([0.0]), np.array([2.3]), None, AsymmetricUncertainty(0.43, 0.30))'
```

Output:

```text
<string>:1: error: Argument 8 to "TRISPointSet" has incompatible type "AsymmetricUncertainty"; expected "int | float | integer[Any] | floating[Any] | ndarray[tuple[Any, ...], dtype[Any]]"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

Exit code: 1, as required for this intentionally invalid static-typing probe.

### Black

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

### Python 3.8 grammar

Command:

```text
python -c 'import ast
from pathlib import Path
for name in ("limTOD/tris.py", "tests/test_tris.py", "tests/typecheck_tris_inputs.py"):
    path = Path(name)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("Python 3.8 grammar parse passed for 3 files")'
```

Output:

```text
Python 3.8 grammar parse passed for 3 files
```

Exit code: 0.

### Whitespace

Pre-commit commands:

```text
git diff --check b946ae5
git diff --check 0fd227c
git diff --check 0fd227c..HEAD
```

Output: empty for all commands. Exit code: 0 for all commands. The first two
include the final working-tree delta; the exact baseline-to-new-HEAD command is
rerun after commit.

### Full pytest suite

Command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m pytest -q
```

Output:

```text
........................................................................ [ 25%]
........................................................................ [ 51%]
........................................................................ [ 76%]
..................................................................       [100%]
=============================== warnings summary ===============================
tests/test_beam_orientation.py: 16 warnings
  /private/tmp/limtod-tris-support/limTOD/simulator.py:397: UserWarning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.
    psi_rad, theta_rad, phi_rad = zyz_of_pointing(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
282 passed, 2 skipped, 16 warnings in 77.69s (0:01:17)
```

Exit code: 0.

### Fresh Sphinx builds

Full command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m sphinx -E -W --keep-going -b html docs /private/tmp/limtod-tris-sphinx-final
```

Relevant output:

```text
reading sources... [ 33%] api/tris
reading sources... [ 95%] tris
writing output... [ 33%] api/tris
writing output... [ 95%] tris
build finished with problems, 22 warnings (with warnings treated as errors).
```

Page-targeted command:

```text
env PYTHONPATH=. MPLCONFIGDIR=/private/tmp/limtod-mpl-cache LIMTOD_FORCE_SERIAL=1 python -m sphinx -E -W --keep-going -b html docs /private/tmp/limtod-tris-sphinx-target docs/tris.md docs/api/tris.md
```

Relevant output:

```text
building [html]: 2 source files given on command line
reading sources... [ 33%] api/tris
reading sources... [ 95%] tris
writing output... [ 50%] api/tris
writing output... [100%] tris
build finished with problems, 23 warnings (with warnings treated as errors).
```

Both commands exited 1 because warnings-as-errors includes unrelated existing
environment/baseline diagnostics. The complete categories were:

- missing optional `s2fft`, causing `limtod_jax` autosummary/autodoc warnings;
- blocked DNS for the Python, NumPy, SciPy, and Astropy intersphinx inventories;
- the pre-existing duplicate `conventions` ID/label in CST/UV-beam docs;
- in the page-targeted build only, an expected incomplete-search-index warning.

Neither build emitted a diagnostic for `docs/tris.md`, `docs/api/tris.md`, or
`limTOD.tris`; both TRIS pages were read and written successfully.

## Files

- `limTOD/tris.py`
- `tests/test_tris.py`
- `docs/tris.md`
- `.superpowers/sdd/2026-08-13-tris-support/final-review-fix-report.md`

## Self-review and concerns

- The safe rank floor is computed from the whitened design independently of
  user input. Below-floor input is rejected, larger input is retained exactly,
  and duplicated columns still fail the default rank gate.
- `TRISPointSet` cannot retain a non-float zero level: its public annotation is
  scalar-only, mypy rejects asymmetric input, and runtime coercion rejects it.
- Normalized Gaussian computation shifts an extended-precision log response;
  the unnormalized path remains the prior direct exponential.
- Header line tuples are private and both point-set header detection and all
  row diagnostics remain covered by the focused suite.
- `__all__` contains exactly the 16 requested binding-spec APIs; no root export
  changed.
- Remaining concerns are only the categorized Sphinx baseline/environment
  warnings, 16 existing pytest gimbal-lock warnings, the non-fatal OpenMPI
  fixture warning, and runtime verification using Python 3.12 rather than a
  Python 3.8 interpreter.
