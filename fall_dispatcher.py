# -*- coding: utf-8 -*-
"""
fall_dispatcher.py
==================
Master entry point for the Multi-Scenario Fall Simulation System.

Architecture
------------
  fall_dispatcher.py          <- YOU ARE HERE (menu + subject input + launch)
  fall_scenario_library.py    <- ScenarioConfig registry & biomechanical helpers
  fall_core.py                <- Simulation engine (all physics classes shared)
  scenarios/
      scenario_20.py  ..  scenario_42.py   (one file per fall type)

Usage
-----
  python fall_dispatcher.py
"""

import sys, os
import importlib

# ------------------------------------------------------------------------------
# FALL SCENARIO CATALOGUE
# ------------------------------------------------------------------------------
SCENARIO_CATALOGUE = {
    # -- Sitting Transitions --------------------------------------------------
    20: ("Forward fall when trying to sit down",         "Sitting transitions"),
    21: ("Backward fall when trying to sit down",        "Sitting transitions"),
    22: ("Lateral fall when trying to sit down",         "Sitting transitions"),
    23: ("Forward fall when trying to get up",           "Sitting transitions"),
    24: ("Lateral fall when trying to get up",           "Sitting transitions"),
    # -- Fainting -------------------------------------------------------------
    25: ("Forward fall while sitting, caused by fainting",          "Fainting"),
    26: ("Lateral fall while sitting, caused by fainting",          "Fainting"),
    27: ("Backward fall while sitting, caused by fainting",         "Fainting"),
    28: ("Vertical/forward fall while walking caused by fainting",  "Fainting"),
    29: ("Fall while walking, hands used to dampen",                "Fainting"),
    # -- Moving Falls ---------------------------------------------------------
    30: ("Forward fall while walking caused by a trip",             "Moving falls"),
    31: ("Forward fall while jogging caused by a trip",             "Moving falls"),
    32: ("Forward fall while walking caused by a slip",             "Moving falls"),
    33: ("Lateral fall while walking caused by a slip",             "Moving falls"),
    34: ("Backward fall while walking caused by a slip",            "Moving falls"),
    # -- Elevation / Moving Backward ------------------------------------------
    37: ("Backward fall while slowly moving back",                  "Elevation"),
    38: ("Backward fall while quickly moving back",                 "Elevation"),
    39: ("Forward fall from height",                                "Elevation"),
    40: ("Backward fall from height",                               "Elevation"),
    41: ("Backward fall while climbing up the ladder",              "Elevation"),
    42: ("Backward fall while climbing down the ladder",            "Elevation"),
}

# Module path for scenario files  (place `scenarios/` next to this file)
SCENARIO_MODULE_PREFIX = "scenarios.scenario_"


# ------------------------------------------------------------------------------
# DISPLAY HELPERS
# ------------------------------------------------------------------------------
def _print_banner():
    print()
    print("=" * 78)
    print("   MULTI-SCENARIO HUMAN FALL SIMULATION SYSTEM")
    print("   Based on: Ferrari et al. DETC2025-169046 | Meta Motivo + MuJoCo")
    print("=" * 78)


def _print_catalogue():
    categories = {}
    for sid, (desc, cat) in SCENARIO_CATALOGUE.items():
        categories.setdefault(cat, []).append((sid, desc))

    for cat, items in categories.items():
        print(f"\n  -- {cat} " + "-" * (54 - len(cat)))
        for sid, desc in items:
            marker = "  [IMPLEMENTED]" if sid == 34 else ""
            print(f"    {sid:3d}  {desc}{marker}")
    print()


def _prompt_int(label, default, lo=None, hi=None):
    while True:
        raw = input(f"    {label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
            if lo is not None and v < lo:
                print(f"      x  Must be >= {lo}. Try again.")
                continue
            if hi is not None and v > hi:
                print(f"      x  Must be <= {hi}. Try again.")
                continue
            return v
        except ValueError:
            print("      x  Please enter an integer.")


def _prompt_float(label, default, unit="", lo=None, hi=None):
    while True:
        raw = input(f"    {label} [{default}{' ' + unit if unit else ''}]: ").strip()
        if raw == "":
            return default
        try:
            v = float(raw)
            if lo is not None and v < lo:
                print(f"      x  Must be >= {lo}. Try again.")
                continue
            if hi is not None and v > hi:
                print(f"      x  Must be <= {hi}. Try again.")
                continue
            return v
        except ValueError:
            print("      x  Please enter a number.")


def _prompt_optional_float(label, unit="", lo=None, hi=None):
    while True:
        raw = input(f"    {label} [auto{' ' + unit if unit else ''}]: ").strip()
        if raw == "":
            return None
        try:
            v = float(raw)
            if lo is not None and v < lo:
                print(f"      x  Must be >= {lo}. Try again.")
                continue
            if hi is not None and v > hi:
                print(f"      x  Must be <= {hi}. Try again.")
                continue
            return v
        except ValueError:
            print("      x  Please enter a number or press Enter for auto.")


def _prompt_str(label, default, choices):
    while True:
        raw = input(f"    {label} [{default}] ({'/'.join(choices)}): ").strip().lower()
        if raw == "":
            return default
        if raw in choices:
            return raw
        print(f"      x  Choose one of: {choices}. Try again.")


# ------------------------------------------------------------------------------
# SUBJECT PARAMETERS
# ------------------------------------------------------------------------------
def collect_subject_params():
    """Interactively collect subject anthropometric parameters."""
    print("\n  -- Subject Parameters (press Enter for defaults) --")
    age    = _prompt_int("Age",     75, lo=1,   hi=120)
    height = _prompt_float("Height", 1.65, "m", lo=0.5, hi=2.5)
    sex    = _prompt_str("Sex",    "male", ["male", "female"])
    weight = _prompt_optional_float("Weight", "kg", lo=25.0, hi=250.0)
    return {"age": age, "height": height, "sex": sex, "weight": weight}


# ------------------------------------------------------------------------------
# SCENARIO LOADER
# ------------------------------------------------------------------------------
def load_scenario_module(scenario_id: int):
    """Dynamically import scenarios/scenario_XX.py and return its module."""
    # Add the directory containing this dispatcher to sys.path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    module_name = f"{SCENARIO_MODULE_PREFIX}{scenario_id}"
    try:
        mod = importlib.import_module(module_name)
        return mod
    except ModuleNotFoundError as exc:
        print(f"\n  [ERROR] Scenario module '{module_name}' not found.")
        print(f"          Expected path: {base_dir}/scenarios/scenario_{scenario_id}.py")
        print(f"          Detail: {exc}")
        return None


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    _print_banner()
    _print_catalogue()

    # -- Select scenario -------------------------------------------------------
    valid_ids = sorted(SCENARIO_CATALOGUE.keys())
    while True:
        raw = input("  Select scenario ID (or 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("  Exiting.")
            sys.exit(0)
        try:
            sid = int(raw)
            if sid in valid_ids:
                break
            print(f"  x  Not a valid ID. Choose from: {valid_ids}")
        except ValueError:
            print("  x  Please enter a number.")

    desc, cat = SCENARIO_CATALOGUE[sid]
    print(f"\n  Selected scenario {sid}: {desc}")
    print(f"  Category: {cat}\n")

    # -- Collect subject parameters --------------------------------------------
    subject = collect_subject_params()
    print(
        f"\n  >> Subject: age={subject['age']}yr  height={subject['height']}m  "
        f"sex={subject['sex']}  weight={'auto' if subject['weight'] is None else str(subject['weight'])+'kg'}\n"
    )

    # -- Load & run scenario module --------------------------------------------
    mod = load_scenario_module(sid)
    if mod is None:
        print("  Cannot continue without scenario module. Exiting.")
        sys.exit(1)

    if not hasattr(mod, "run"):
        print(f"  [ERROR] Scenario module must expose a `run(subject_params)` function.")
        sys.exit(1)

    print(f"  Launching scenario {sid} ...\n")
    mod.run(subject)


if __name__ == "__main__":
    main()
