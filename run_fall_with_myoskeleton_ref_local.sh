#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

export MYOSUITE_MODEL_XML="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/.venv/lib/python3.12/site-packages/myosuite/simhive/myo_model/myoskeleton/myoskeleton.xml"

echo "Using MYOSUITE_MODEL_XML=$MYOSUITE_MODEL_XML"
exec python3 /mnt/hdd16T/ToqeerHomeBackup/mujoco_project/backward_fall_walking.py
