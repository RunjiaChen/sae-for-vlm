#!/bin/bash
# Submits the whole Fig 3 pipeline with dependencies:
#   smoke ─┬─> steer(unit) ─┬
#          └─> steer(white) ├─> finalize
#   extract ──> dissect ────┘
# Run on xlogin1 AFTER the setup job (venv + downloads) has finished.
set -euo pipefail
cd "$(dirname "$0")"

SMOKE=$(sbatch --parsable smoke.sh)
EXTRACT=$(sbatch --parsable extract.sh)
DISSECT=$(sbatch --parsable --dependency=afterok:$EXTRACT dissect.sh)
SU=$(sbatch --parsable --dependency=afterok:$SMOKE --export=ALL,VARIANT=unit steer.sh)
SW=$(sbatch --parsable --dependency=afterok:$SMOKE --export=ALL,VARIANT=white steer.sh)
FIN=$(sbatch --parsable --dependency=afterok:$DISSECT:$SU:$SW finalize.sh)

echo "smoke:       $SMOKE"
echo "extract:     $EXTRACT"
echo "dissect:     $DISSECT (afterok extract)"
echo "steer unit:  $SU (afterok smoke)"
echo "steer white: $SW (afterok smoke)"
echo "finalize:    $FIN (afterok dissect+steers)"
echo "logs: /mnt/scratch/r/runjia/fig3-workspace/logs/"
