#!/bin/bash
set -euo pipefail
ROOT=/mnt/x/MachineLearning/Baiology/SDM/KAN-Maxent
OUT=$ROOT/outputs/methodological_closure_full_v1
LOG=$OUT/run_bcd_parallel.log
METRICS=$OUT/metrics.csv
cd "$ROOT"

echo "[$(date -Is)] watchdog start" | tee -a "$LOG"

find_runner_pids() {
  # match only the real runner, not this bash script
  ps -eo pid=,args= | awk '/python -u benchmarks\/methodological_closure\.py/ && !/awk/ {print $1}'
}

while true; do
  A_N=0
  if [ -f "$METRICS" ]; then
    A_N=$(python - <<'PY'
import pandas as pd
from pathlib import Path
p=Path("outputs/methodological_closure_full_v1/metrics.csv")
df=pd.read_csv(p)
a=df[(df.stage=="A")&(df.model=="additive_kan_ipp")]
print(a.species_id.nunique())
PY
)
  fi
  STAGES=$(python - <<'PY' 2>/dev/null || echo A
import pandas as pd
from pathlib import Path
p=Path("outputs/methodological_closure_full_v1/metrics.csv")
df=pd.read_csv(p)
print(",".join(sorted(df.stage.unique())))
PY
)
  echo "[$(date -Is)] Stage A species=$A_N stages=$STAGES" | tee -a "$LOG"

  if [ "$A_N" -ge 225 ] || echo "$STAGES" | grep -qE 'B|C|D'; then
    break
  fi
  PIDS=$(find_runner_pids || true)
  if [ -z "${PIDS:-}" ]; then
    echo "[$(date -Is)] no runner; will start BCD" | tee -a "$LOG"
    break
  fi
  sleep 30
done

PIDS=$(find_runner_pids || true)
if [ -n "${PIDS:-}" ]; then
  echo "[$(date -Is)] stopping serial runner(s): $PIDS" | tee -a "$LOG"
  # shellcheck disable=SC2086
  kill $PIDS 2>/dev/null || true
  sleep 5
  PIDS2=$(find_runner_pids || true)
  if [ -n "${PIDS2:-}" ]; then
    # shellcheck disable=SC2086
    kill -9 $PIDS2 2>/dev/null || true
    sleep 2
  fi
fi

NCPU=$(nproc)
WORKERS=$(( NCPU * 80 / 100 ))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
echo "[$(date -Is)] launching B,C,D,E with --workers $WORKERS (reserve ~20% of $NCPU)" | tee -a "$LOG"

nohup nice -n 10 python -u benchmarks/methodological_closure.py \
  --regions AWT,CAN,NSW,NZ,SA,SWI \
  --stages B,C,D,E \
  --seeds-deep 0,1,2 \
  --workers "$WORKERS" \
  --blas-threads 1 \
  --resume \
  --outdir outputs/methodological_closure_full_v1 \
  >> "$LOG" 2>&1 &
echo "[$(date -Is)] BCD PID=$!" | tee -a "$LOG"
echo $! > "$OUT/bcd_parallel.pid"
