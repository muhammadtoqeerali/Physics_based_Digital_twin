# -*- coding: utf-8 -*-
"""
scenario_21.py
==============
Scenario 21 — Backward fall when trying to sit down.

Biomechanics
------------
  Subject leans too far posteriorly during the lowering phase or misjudges
  chair location.  Posterior CoM displacement exceeds the ankle dorsiflexion
  range; the subject tips backward at ~70°–85° trunk-from-vertical
  (Alexander 1994; Grabiner et al. 2008).

  In elderly individuals reduced ankle proprioception and slower corticospinal
  conduction increase the probability of this scenario (Lord et al. 2001).

  Phase sequence: STAND → PERTURB (sit-down crouch) → REACT (backward push) → FALL
  lie_orient: 'up'  (supine — classic buttocks-first landing)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fall_scenario_library import get_scenario
from fall_core import run_simulation


def run(subject_params: dict) -> dict:
    config = get_scenario(21)
    return run_simulation(config, subject_params)


if __name__ == "__main__":
    run({"age": 72, "height": 1.62, "sex": "female", "weight": None})
