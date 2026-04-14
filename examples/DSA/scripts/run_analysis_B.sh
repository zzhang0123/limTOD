#!/usr/bin/env bash
# Analysis B: MeerKLASS elevation cascade for sub-beam recovery, low noise.
#
# Generates _cascade meerklass TODs (5 elevations × 3 repeats × 2 scans =
# 10 TODs), builds map-making operators at ns=16 and ns=64, and writes the
# figures via compare_maps.py --analysis B. The headline comparison is
# baseline ns=64 (Analysis A) vs cascade ns=64 (this analysis).
#
# Wall-time on 8 cores: ~7 min TODs + ~10 min ops + ~20 s figs.
set -euo pipefail
cd "$(dirname "$0")"

WHITE_VAR="${WHITE_VAR:-1e-7}"
GAIN_F0="${GAIN_F0:-1.335e-5}"
NRANKS="${NRANKS:-8}"

CONDA_RUN=(conda run --no-capture-output -n TOD)
MPI_PY=(env OMP_NUM_THREADS=1 mpirun -n "$NRANKS" "${CONDA_RUN[@]}" python)

echo "=== [B] step 1/3 — TODs (low-noise, 5-elevation cascade) ==="
"${MPI_PY[@]}" sim_meerklass_tod.py \
    --elevations 53 54 55 56 57 --n-repeats 3 \
    --gain-f0 "$GAIN_F0" --white-var "$WHITE_VAR" \
    --out-suffix _cascade

echo "=== [B] step 2/3 — Map-making operators at ns=64 ==="
for NS in 64; do
  "${MPI_PY[@]}" build_mapmaker_ops.py meerklass \
      --nside-target "$NS" --beam-nside-map 64 \
      --threshold 0.01 --truncate-frac 0.01 \
      --tod-suffix _cascade --out-suffix _cascade
done

echo "=== [B] step 3/3 — Figures ==="
"${CONDA_RUN[@]}" python compare_maps.py --analysis B --white-var "$WHITE_VAR"
