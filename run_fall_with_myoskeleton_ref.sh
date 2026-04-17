#!/usr/bin/env bash
set -euo pipefail

MYO_XML="${MYOSUITE_MODEL_XML:-}"
if [[ -z "$MYO_XML" ]]; then
  MYO_XML=$(python3 - <<'PY'
from pathlib import Path
try:
    import myosuite  # type: ignore
    pkg_dir = Path(myosuite.__file__).resolve().parent
    cand = pkg_dir / 'simhive' / 'myo_model' / 'myoskeleton' / 'myoskeleton.xml'
    if cand.exists():
        print(cand)
except Exception:
    pass
PY
)
fi

if [[ -z "$MYO_XML" || ! -f "$MYO_XML" ]]; then
  echo "Could not find myoskeleton.xml. Install it first with:"
  echo "  bash /mnt/data/install_myoskeleton.sh"
  exit 1
fi

export MYOSUITE_MODEL_XML="$MYO_XML"
echo "Using MYOSUITE_MODEL_XML=$MYOSUITE_MODEL_XML"
exec python3 /mnt/hdd16T/ToqeerHomeBackup/mujoco_project/backward_fall_walking.py
