#!/usr/bin/env bash
# Analysis A: single-elevation survey-strategy comparison at low noise.
#
# Generates _baseline TODs for both meerklass and steer_and_stare, builds
# their map-making operators at ns=16 and ns=64, then writes the comparison
# figures via compare_maps.py --analysis A.
#
# Wall-time on 8 cores: ~12 min for TODs + ~10 min ops + ~30 s figs.
set -euo pipefail
cd "$(dirname "$0")"

WHITE_VAR="${WHITE_VAR:-1e-7}"
GAIN_F0="${GAIN_F0:-1.335e-7}"
NRANKS="${NRANKS:-8}"

CONDA_RUN=(conda run --no-capture-output -n TOD)
MPI_PY=(env OMP_NUM_THREADS=1 mpirun -n "$NRANKS" "${CONDA_RUN[@]}" python)

echo "=== [A] step 1/3 — TODs (low-noise, single elevation) ==="
"${MPI_PY[@]}" sim_meerklass_tod.py \
    --elevations 55 --n-repeats 13 \
    --gain-f0 "$GAIN_F0" --white-var "$WHITE_VAR" \
    --out-suffix _baseline

"${MPI_PY[@]}" sim_steer_and_stare_tod.py \
    --gain-f0 "$GAIN_F0" --white-var "$WHITE_VAR" \
    --out-suffix _baseline

echo "=== [A] step 2/3 — Map-making operators at ns=64 ==="
for NS in 64; do
  "${MPI_PY[@]}" build_mapmaker_ops.py meerklass \
      --nside-target "$NS" --beam-nside-map 64 \
      --threshold 0.01 --truncate-frac 0.01 \
      --tod-suffix _baseline --out-suffix _baseline
  "${MPI_PY[@]}" build_mapmaker_ops.py steer_and_stare \
      --nside-target "$NS" --beam-nside-map 64 \
      --threshold 0.01 --truncate-frac 0.01 \
      --tod-suffix _baseline --out-suffix _baseline
done

echo "=== [A] step 3/3 — Figures ==="
"${CONDA_RUN[@]}" python compare_maps.py --analysis A --white-var "$WHITE_VAR"
