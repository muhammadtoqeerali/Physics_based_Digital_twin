# -*- coding: utf-8 -*-
"""
fall_core.py
============
Parameterised fall simulation engine.

All physics classes (IMU, Marker kinematics, Dynamics, Validators, etc.)
live here. Scenario-specific behaviour is injected via ScenarioConfig from
fall_scenario_library.py.

Public API
----------
  run_simulation(scenario_config, subject_params) -> dict
      Full simulation + export. Returns summary dict.

  build_controller(env, mj_model, mj_data, age_params, scenario_config)
      Returns a configured EnhancedBiofidelicController.

Dependencies
------------
  backward_fall_walking_best.py  (all physics / controller classes are
  imported from there so there is zero code duplication)

The engine adds:
  1. Scenario-aware phase management
  2. Scenario-specific perturbation dispatch
  3. Initial condition setup (elevated, backward, sit-down)
  4. Faint / trip / slip physics hooks
"""

import sys, os, re
import numpy as np
import torch

# -- Resolve project root ------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# If running from project dir, also include parent
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# -- Import shared physics / controller classes from existing validated code ---
# We import specific names so they stay unambiguous.
from backward_fall_walking_best import (
    # MuJoCo helpers
    MJOBJ_BODY, MJOBJ_ACTUATOR, MJOBJ_GEOM, MJOBJ_SITE,
    _body_world_velocity, _contact_wrench_world,
    _norm_name, _tokenize_name, _body_name_matches_side,
    _safe_name2id, resolve_imu_mount_configuration,

    # Subject profiling
    AnthropometricModel, resolve_subject_weight_kg,
    estimate_reference_bmi,

    # IMU + Kinematics
    IMU_HARDWARE_SPEC, IMUValidator,
    MarkerKinematicsExporter, DynamicsContactAnalyzer,
    PaperAlignmentExporter, FallValidator, SISFallValidator, KFallValidator,

    # Controller
    EnhancedBiofidelicController, BiofidelicFallController,
    GaitPhaseDetector, FallTypeLibrary,

    # Dashboard
    PhysicsDashboard,

    # Embeddings
    infer_z_walk_stable, infer_z_backward_fall, infer_z_lie_rest, infer_z_stand,

    # biofidelic helpers
    MYOSUITE_INTEGRATION,
    get_age_reference_band, get_age_style,
    print_age_behavior_audit, compute_subject_control_targets,
    LAST_Z_WALK_DIAGNOSTICS,

    # Event detection
    detect_fall_events,
    compute_preperturb_walk_metrics,

    # Config constants
    SEED, Z_INTERPOLATION_STEPS, ACTION_SMOOTHING,
    STABILITY_WARMUP_STEPS, XCOM_NEGATIVE_CONFIRM_STEPS,
    XCOM_LOG_START_OFFSET_STEPS,
)

import mujoco
import mujoco.viewer
from humenv import make_humenv
from humenv.rewards import LocomotionReward
from metamotivo.fb_cpr.huggingface import FBcprModel
from biofidelic_profile import (
    phase_timing, get_age_style_v2, apply_age_effects_v2,
    perturbation_force_config, weakening_config,
    print_subject_profile,
)

from fall_scenario_library import (
    ScenarioConfig, InitialState, PerturbationType, FallDirection,
    apply_slip_to_floor, apply_trip_blocking, apply_faint_collapse,
    elevate_initial_position, set_backward_velocity, apply_sit_down_force,
)

# ------------------------------------------------------------------------------
# GLOBAL MODEL CACHE  (avoid re-loading Meta Motivo between scenarios)
# ------------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def _get_motivo_model():
    if "model" not in _MODEL_CACHE:
        print("  [Core] Loading Meta Motivo model (first run only)...")
        m = FBcprModel.from_pretrained("facebook/metamotivo-M-1")
        m.eval()
        _MODEL_CACHE["model"] = m
        print(f"  [Core] Model on {next(m.parameters()).device}")
    return _MODEL_CACHE["model"]


def _pelvis_avatar_frame(mj_data, pelvis_id: int):
    """Return (forward3, lateral3, yaw_rad) from the pelvis body frame."""
    if pelvis_id < 0:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 0.0
    R = np.asarray(mj_data.xmat[pelvis_id], dtype=float).reshape(3, 3)
    f = R[:, 0].copy()
    f[2] = 0.0
    fn = float(np.linalg.norm(f))
    if fn < 1e-9:
        f = np.array([1.0, 0.0, 0.0])
    else:
        f /= fn
    lat = np.array([-f[1], f[0], 0.0], dtype=float)
    yaw = float(np.arctan2(f[1], f[0]))
    return f, lat, yaw


def _local_dir_to_world(local_dir: np.ndarray, fwd3: np.ndarray, lat3: np.ndarray):
    """Map local [forward, lateral, vertical] to world XYZ."""
    ld = np.asarray(local_dir, dtype=float).reshape(3)
    out = ld[0] * fwd3 + ld[1] * lat3 + np.array([0.0, 0.0, ld[2]], dtype=float)
    n = float(np.linalg.norm(out))
    return out / n if n > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)


def _infer_z_sit_attempt_forward():
    """Forward-lean upright posture used before sit-down collapse."""
    model = _get_motivo_model()
    print("  [Embeddings] Inferring z_sit_attempt …")
    env, _ = make_humenv(task="move-ego-0-0")
    pelvis_id = mujoco.mj_name2id(env.unwrapped.model, MJOBJ_BODY, "Pelvis")
    torso_id  = mujoco.mj_name2id(env.unwrapped.model, MJOBJ_BODY, "Torso")
    head_id   = mujoco.mj_name2id(env.unwrapped.model, MJOBJ_BODY, "Head")
    sit_obs = []
    for trial in range(24):
        torch.manual_seed(SEED + 7000 + trial)
        z = model.sample_z(1)
        obs, _ = env.reset()
        for _ in range(100):
            d = env.unwrapped.data
            if torso_id >= 0 and pelvis_id >= 0:
                fwd3, _, _ = _pelvis_avatar_frame(d, pelvis_id)
                d.xfrc_applied[torso_id, 0] += 16.0 * fwd3[0]
                d.xfrc_applied[torso_id, 1] += 16.0 * fwd3[1]
                d.xfrc_applied[torso_id, 4] += -5.0
            with torch.no_grad():
                act = model.act(torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0), z).squeeze(0).numpy()
            obs, _, term, trunc, _ = env.step(act)
            ph = float(env.unwrapped.data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            if ph > 0.80 and head_id >= 0 and pelvis_id >= 0:
                vec = env.unwrapped.data.xpos[head_id] - env.unwrapped.data.xpos[pelvis_id]
                nv = float(np.linalg.norm(vec))
                if nv > 1e-9:
                    lean = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))))
                    if 5.0 <= lean <= 32.0:
                        sit_obs.append(obs['proprio'].copy())
            if term or trunc:
                break
    env.close()
    if len(sit_obs) >= 40:
        obs_t = torch.tensor(np.array(sit_obs[-300:]), dtype=torch.float32)
        with torch.no_grad():
            z_sit = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_sit = infer_z_stand()
    print(f"  [Embeddings] z_sit_attempt from {len(sit_obs)} states")
    return z_sit


def _infer_z_forward_prone():
    """Forward/prone collapse embedding for scenario 20-like tasks."""
    model = _get_motivo_model()
    print("  [Embeddings] Inferring z_fall_forward …")
    env, _ = make_humenv(task="lieonground-up")
    pelvis_id = mujoco.mj_name2id(env.unwrapped.model, MJOBJ_BODY, "Pelvis")
    head_id   = mujoco.mj_name2id(env.unwrapped.model, MJOBJ_BODY, "Head")
    torso_id  = mujoco.mj_name2id(env.unwrapped.model, MJOBJ_BODY, "Torso")
    prone_obs = []
    zero = None
    for _trial in range(18):
        obs, _ = env.reset()
        if zero is None:
            zero = np.zeros(env.action_space.shape)
        for _ in range(180):
            d = env.unwrapped.data
            if torso_id >= 0 and pelvis_id >= 0:
                fwd3, lat3, _ = _pelvis_avatar_frame(d, pelvis_id)
                d.xfrc_applied[torso_id, 0] += 120.0 * fwd3[0]
                d.xfrc_applied[torso_id, 1] += 120.0 * fwd3[1]
                d.xfrc_applied[torso_id, 3:6] += np.array([-30.0 * lat3[0], -30.0 * lat3[1], 0.0])
            if pelvis_id >= 0:
                d.xfrc_applied[pelvis_id, 2] -= 45.0
            obs, _, term, trunc, _ = env.step(zero)
            if pelvis_id >= 0 and head_id >= 0:
                p = d.xpos[pelvis_id]
                h = d.xpos[head_id]
                qn = float(np.linalg.norm(d.qvel))
                vec = h - p
                nv = float(np.linalg.norm(vec))
                tilt = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))) if nv > 1e-9 else 0.0
                if float(p[2]) < 0.24 and float(h[2]) < 0.42 and tilt > 65.0 and qn < 2.4:
                    prone_obs.append(obs['proprio'].copy())
            if term or trunc:
                break
    env.close()
    if len(prone_obs) >= 60:
        obs_t = torch.tensor(np.array(prone_obs[-300:]), dtype=torch.float32)
        with torch.no_grad():
            z_fwd = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_fwd = infer_z_backward_fall()
    print(f"  [Embeddings] z_fall_forward from {len(prone_obs)} states")
    return z_fwd


def _infer_z_prone_settle():
    """Stable prone settle embedding for forward-fall tasks."""
    model = _get_motivo_model()
    print("  [Embeddings] Inferring z_rest_prone …")
    env, _ = make_humenv(task="lieonground-up")
    pelvis_id = mujoco.mj_name2id(env.unwrapped.model, MJOBJ_BODY, "Pelvis")
    head_id   = mujoco.mj_name2id(env.unwrapped.model, MJOBJ_BODY, "Head")
    prone_obs = []
    zero = np.zeros(env.action_space.shape)
    for _trial in range(12):
        obs, _ = env.reset()
        for _ in range(180):
            obs, _, term, trunc, _ = env.step(zero)
            d = env.unwrapped.data
            if pelvis_id >= 0 and head_id >= 0:
                p = d.xpos[pelvis_id]
                h = d.xpos[head_id]
                qn = float(np.linalg.norm(d.qvel))
                vec = h - p
                nv = float(np.linalg.norm(vec))
                tilt = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))) if nv > 1e-9 else 0.0
                if float(p[2]) < 0.20 and float(h[2]) < 0.36 and qn < 1.4 and tilt > 72.0:
                    prone_obs.append(obs['proprio'].copy())
            if term or trunc:
                break
    env.close()
    if len(prone_obs) >= 40:
        obs_t = torch.tensor(np.array(prone_obs[-300:]), dtype=torch.float32)
        with torch.no_grad():
            z_rest = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_rest = _infer_z_forward_prone()
    print(f"  [Embeddings] z_rest_prone from {len(prone_obs)} states")
    return z_rest


# ------------------------------------------------------------------------------
# SCENARIO-AWARE EMBEDDING INFERENCE
# ------------------------------------------------------------------------------
def infer_embeddings(scenario: ScenarioConfig, age: int, height: float,
                     sex: str, body_mass: float):
    """Return (z_stand, z_pre, z_fall, z_rest).

    Sitting transitions are direction-sensitive. Scenario 20 must not re-use
    the backward-fall embeddings from the validated backward-slip task.
    """
    state = scenario.initial_state

    print("  [Embeddings] Inferring z_stand …")
    z_stand = infer_z_stand()

    if state in (InitialState.SITTING, InitialState.SITTING_DOWN, InitialState.GETTING_UP):
        if scenario.fall_direction == FallDirection.FORWARD:
            z_pre  = _infer_z_sit_attempt_forward()
            z_fall = _infer_z_forward_prone()
            z_rest = _infer_z_prone_settle()
            print("  [Embeddings] Sitting-forward scenario ? using forward sit/prone embeddings.")
        else:
            print("  [Embeddings] Inferring z_fall …")
            z_fall = infer_z_backward_fall()
            print("  [Embeddings] Inferring z_rest …")
            z_rest = infer_z_lie_rest()
            z_pre = z_fall
            print("  [Embeddings] Sitting non-forward scenario ? using low-posture fall embedding as pre-embed.")
    elif state == InitialState.JOGGING:
        print("  [Embeddings] Inferring z_fall …")
        z_fall = infer_z_backward_fall()
        print("  [Embeddings] Inferring z_rest …")
        z_rest = infer_z_lie_rest()
        print("  [Embeddings] Inferring z_jog …")
        z_pre = infer_z_walk_stable(n_samples=1000, age=age)
    elif state == InitialState.MOVING_BACKWARD:
        print("  [Embeddings] Inferring z_fall …")
        z_fall = infer_z_backward_fall()
        print("  [Embeddings] Inferring z_rest …")
        z_rest = infer_z_lie_rest()
        print("  [Embeddings] Backward-moving ? re-using z_walk (reversed guidance).")
        z_pre = infer_z_walk_stable(n_samples=1000, age=age)
    elif state == InitialState.ELEVATED:
        print("  [Embeddings] Inferring z_fall …")
        z_fall = infer_z_backward_fall()
        print("  [Embeddings] Inferring z_rest …")
        z_rest = infer_z_lie_rest()
        print("  [Embeddings] Elevated ? using z_stand as pre-embed.")
        z_pre = z_stand
    else:
        print("  [Embeddings] Inferring z_fall …")
        z_fall = infer_z_backward_fall()
        print("  [Embeddings] Inferring z_rest …")
        z_rest = infer_z_lie_rest()
        print("  [Embeddings] Inferring z_walk …")
        z_pre = infer_z_walk_stable(n_samples=1000, age=age)

    return z_stand, z_pre, z_fall, z_rest


# ------------------------------------------------------------------------------
# INITIAL STATE SETUP
# ------------------------------------------------------------------------------
def setup_initial_conditions(mj_data, scenario: ScenarioConfig, obs, env):
    """Apply scenario-specific initial state modifications.

    Must be called right after env.reset().
    """
    if abs(scenario.initial_height_offset) > 0.01:
        elevate_initial_position(
            mj_data,
            scenario.initial_height_offset,
            scenario.initial_pitch_deg,
        )
        mujoco.mj_forward(env.unwrapped.model, mj_data)

    if scenario.initial_backward_speed > 0.01:
        set_backward_velocity(mj_data, scenario.initial_backward_speed)


# ------------------------------------------------------------------------------
# SCENARIO-SPECIFIC PERTURBATION DISPATCH
# ------------------------------------------------------------------------------
class PerturbationState:
    """Mutable state carried through the perturbation loop."""
    def __init__(self):
        self.trip_active:      bool  = False
        self.trip_step_count:  int   = 0
        self.slip_applied:     bool  = False
        self.faint_started:    bool  = False
        self.faint_step:       int   = 0
        self.original_gear:    np.ndarray = np.array([])
        self.locked_fwd:       np.ndarray | None = None
        self.locked_lat:       np.ndarray | None = None


def apply_scenario_perturbation(
    step: int,
    phase: str,
    controller: "EnhancedBiofidelicController",
    mj_model,
    mj_data,
    scenario: ScenarioConfig,
    age_params: dict,
    body_mass: float,
    pert_state: PerturbationState,
    phase_boundaries: np.ndarray,
    perturb_start_index: int,
):
    """Central dispatch for all perturbation types."""
    bw = body_mass * 9.81
    ptype = scenario.perturbation_type

    # -- Slip: modify floor friction once -------------------------------------
    if (ptype == PerturbationType.SLIP
            and not pert_state.slip_applied
            and phase in ("perturb", "react", "fall")):
        apply_slip_to_floor(
            mj_model,
            scenario.slip_friction_coeff if scenario.slip_friction_coeff else 0.10,
            lateral_only=scenario.slip_lateral_only,
        )
        pert_state.slip_applied = True
        print(f"  [Step {step}] Floor friction reduced ? slip scenario active")

    # -- Trip: block the foot --------------------------------------------------
    if ptype == PerturbationType.TRIP:
        if not pert_state.trip_active:
            # Wait for heel-strike (gait phase detector)
            gait_ok = controller.gait_detector.is_in_window("forward_stumble")
            if gait_ok or (step - perturb_start_index) > 20:
                pert_state.trip_active    = True
                pert_state.trip_step_count = 0
                print(f"  [Step {step}] Trip blocking activated on {scenario.trip_foot}")
        if pert_state.trip_active and pert_state.trip_step_count < scenario.trip_duration_steps:
            apply_trip_blocking(
                mj_model, mj_data,
                scenario.trip_foot,
                scenario.trip_blocking_bw,
                body_mass,
            )
            pert_state.trip_step_count += 1
        return  # trip doesn't use xfrc_applied on torso/pelvis

    # -- Faint: progressive muscle collapse -----------------------------------
    if ptype == PerturbationType.FAINT:
        if not pert_state.faint_started and phase in ("perturb", "react", "fall"):
            pert_state.faint_started = True
            pert_state.original_gear = mj_model.actuator_gear[:, 0].copy()
            print(f"  [Step {step}] Fainting collapse initiated")

        if pert_state.faint_started:
            # Optionally keep arm actuators active (protective_arms scenario 29)
            arm_idxs = controller.arm_actuators if scenario.protective_arms else []
            saved_arm = mj_model.actuator_gear[arm_idxs, 0].copy() if arm_idxs else None

            apply_faint_collapse(
                mj_model,
                pert_state.original_gear,
                pert_state.faint_step,
                scenario.faint_collapse_steps,
                scenario.faint_residual_gear,
                direction_force_body="Torso",
                direction_force=scenario.faint_direction_force,
                body_mass=body_mass,
                mj_data=mj_data,
            )
            pert_state.faint_step += 1

            if arm_idxs and saved_arm is not None:
                # Restore arm gears - they should stay active
                mj_model.actuator_gear[arm_idxs, 0] = np.maximum(
                    saved_arm, pert_state.original_gear[arm_idxs] * 0.25
                )
        return

    # -- Sit-down gravity assist -----------------------------------------------
    if scenario.initial_state in (InitialState.SITTING_DOWN,
                                   InitialState.SITTING,
                                   InitialState.GETTING_UP) and phase == "perturb":
        pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")
        pelvis_z  = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.9
        if getattr(scenario, "use_avatar_frame", False) and pelvis_id >= 0 and pert_state.locked_fwd is None:
            pert_state.locked_fwd, pert_state.locked_lat, yaw = _pelvis_avatar_frame(mj_data, pelvis_id)
            print(f"  [Step {step}] Locked avatar heading for scenario {scenario.scenario_id}: yaw={np.degrees(yaw):+.1f} deg")
        sit_bias = np.asarray(scenario.sit_direction_bias, dtype=float)
        if getattr(scenario, "use_avatar_frame", False) and pert_state.locked_fwd is not None:
            sit_bias = _local_dir_to_world(sit_bias, pert_state.locked_fwd, pert_state.locked_lat)
        apply_sit_down_force(
            mj_model, mj_data, pelvis_id, pelvis_z,
            scenario.sit_target_pelvis_z,
            sit_bias,
            body_mass,
            step - perturb_start_index,
            scenario.sit_crouch_steps,
        )
        # During second half of perturb phase also add the directional fall force
        pert_progress = float(step - perturb_start_index) / max(scenario.sit_crouch_steps, 1)
        if pert_progress > 0.45:
            _apply_directional_push(controller, mj_model, mj_data, scenario, body_mass,
                                     ramp=np.clip((pert_progress - 0.45) / 0.55, 0.0, 1.0),
                                     pert_state=pert_state, age_params=age_params)
        return

    # -- Default: external push ------------------------------------------------
    pert_steps_done = step - perturb_start_index
    total_perturb   = scenario.force_ramp_steps
    ramp_progress   = float(np.clip(pert_steps_done / max(total_perturb, 1), 0.0, 1.0))
    _apply_directional_push(controller, mj_model, mj_data, scenario, body_mass, ramp=ramp_progress, pert_state=pert_state, age_params=age_params)


def _apply_directional_push(controller, mj_model, mj_data,
                              scenario: ScenarioConfig, body_mass: float,
                              ramp: float, pert_state: PerturbationState | None = None,
                              age_params: dict | None = None):
    """Apply a ramped xfrc_applied force in either world or avatar-local frame."""
    bw = body_mass * 9.81
    magnitude = scenario.force_bw_fraction * bw
    force_factor = 0.5 * (1.0 - np.cos(float(np.clip(ramp, 0, 1)) * np.pi))

    body_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, scenario.application_point)
    if body_id < 0:
        body_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")
    if body_id < 0:
        return

    d = np.asarray(scenario.force_direction, dtype=float)
    if getattr(scenario, "use_avatar_frame", False):
        pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")
        if pert_state is not None and pert_state.locked_fwd is None and pelvis_id >= 0:
            pert_state.locked_fwd, pert_state.locked_lat, _ = _pelvis_avatar_frame(mj_data, pelvis_id)
        fwd3 = pert_state.locked_fwd if (pert_state is not None and pert_state.locked_fwd is not None) else np.array([1.0, 0.0, 0.0])
        lat3 = pert_state.locked_lat if (pert_state is not None and pert_state.locked_lat is not None) else np.array([0.0, 1.0, 0.0])
        d = _local_dir_to_world(d, fwd3, lat3)
    else:
        d_norm = float(np.linalg.norm(d))
        if d_norm < 1e-9:
            return
        d = d / d_norm

    age_scale = 1.0
    if age_params:
        sf  = float(age_params.get("strength_factor", age_params.get("strength_factor_raw", 1.0)))
        bal = float(age_params.get("balance_impairment", 0.0))
        if scenario.scenario_id == 20:
            age_scale = float(np.clip(1.08 + 0.30 * (sf - 0.85) - 0.20 * bal, 0.92, 1.22))
    applied = magnitude * force_factor * age_scale * d
    mj_data.xfrc_applied[body_id, :3] += applied

    torque_mag = force_factor * 15.0 * age_scale
    if getattr(scenario, "use_avatar_frame", False):
        fwd3 = pert_state.locked_fwd if (pert_state is not None and pert_state.locked_fwd is not None) else np.array([1.0, 0.0, 0.0])
        lat3 = pert_state.locked_lat if (pert_state is not None and pert_state.locked_lat is not None) else np.array([0.0, 1.0, 0.0])
        if scenario.fall_direction == FallDirection.BACKWARD:
            mj_data.xfrc_applied[body_id, 3:6] += torque_mag * lat3
        elif scenario.fall_direction == FallDirection.FORWARD:
            mj_data.xfrc_applied[body_id, 3:6] += -torque_mag * lat3
        elif scenario.fall_direction == FallDirection.LATERAL_LEFT:
            mj_data.xfrc_applied[body_id, 3:6] += -torque_mag * fwd3
        elif scenario.fall_direction == FallDirection.LATERAL_RIGHT:
            mj_data.xfrc_applied[body_id, 3:6] += torque_mag * fwd3
    else:
        if scenario.fall_direction == FallDirection.BACKWARD:
            mj_data.xfrc_applied[body_id, 4] += torque_mag
        elif scenario.fall_direction == FallDirection.FORWARD:
            mj_data.xfrc_applied[body_id, 4] += -torque_mag
        elif scenario.fall_direction == FallDirection.LATERAL_LEFT:
            mj_data.xfrc_applied[body_id, 3] += -torque_mag
        elif scenario.fall_direction == FallDirection.LATERAL_RIGHT:
            mj_data.xfrc_applied[body_id, 3] += torque_mag


# ------------------------------------------------------------------------------
# MAIN SIMULATION ENTRY POINT
# ------------------------------------------------------------------------------
def run_simulation(scenario: ScenarioConfig, subject_params: dict) -> dict:
    """Run a complete fall simulation for the given scenario + subject.

    Parameters
    ----------
    scenario       : ScenarioConfig from fall_scenario_library
    subject_params : dict with keys age, height, sex, weight (weight may be None)

    Returns
    -------
    summary dict with keys: scenario_id, classification, authenticity_score,
                            imu_csv, validation_report, ...
    """
    from datetime import datetime

    age    = int(subject_params["age"])
    height = float(subject_params["height"])
    sex    = str(subject_params.get("sex", "male")).lower()
    weight = subject_params.get("weight")

    resolved_weight, target_bmi, weight_src = resolve_subject_weight_kg(
        height, age, sex, explicit_weight=weight
    )
    weight_display = (
        f"{weight:.1f}kg (user)" if weight is not None
        else f"{resolved_weight:.1f}kg (auto BMI {target_bmi:.1f})"
    )

    print("\n" + "=" * 70)
    print(f"  Scenario {scenario.scenario_id}: {scenario.description}")
    print(f"  Subject: age={age}yr  height={height}m  weight={weight_display}  sex={sex}")
    print("=" * 70)

    # Print subject profile
    print_subject_profile(age, sex, height, resolved_weight)

    # -- Build phase plan -----------------------------------------------------
    base_phases = phase_timing(age, sex)
    phases = scenario.resolve_phases(base_phases)
    total_steps = sum(phases.values())
    print(f"  Phases: {phases}  |  Total steps: {total_steps}")

    # -- Load Meta Motivo -----------------------------------------------------
    motivo_model = _get_motivo_model()

    # -- Infer task embeddings -------------------------------------------------
    z_stand, z_pre, z_fall, z_rest = infer_embeddings(
        scenario, age, height, sex, resolved_weight
    )

    # -- Create environment ----------------------------------------------------
    task_str = "move-ego-0-2" if scenario.initial_state in (
        InitialState.WALKING, InitialState.JOGGING, InitialState.MOVING_BACKWARD
    ) else "move-ego-0-0"
    env, _ = make_humenv(task=task_str)
    obs, _ = env.reset()
    mj_model = env.unwrapped.model
    mj_data  = env.unwrapped.data

    # -- IMU mount -------------------------------------------------------------
    myosuite_mount = resolve_imu_mount_configuration(
        mj_model,
        requested_xml_path=MYOSUITE_INTEGRATION.get("model_xml", ""),
    )

    # -- Anthropometric model --------------------------------------------------
    print(f"\n[A] Applying AnthropometricModel …")
    anthro     = AnthropometricModel(mj_model, age=age, height=height,
                                     weight=resolved_weight, sex=sex)
    age_params = anthro.apply_age_effects()
    body_mass  = float(np.sum(mj_model.body_mass))

    # -- Tune force config per scenario ---------------------------------------
    FORCE_CONFIG = perturbation_force_config(body_mass, age, sex, scenario.force_bw_fraction)
    WEAKENING_CONFIG = weakening_config(age, sex, body_mass)

    # -- IMU -------------------------------------------------------------------
    print("[B] Initialising IMUValidator …")
    imu = IMUValidator(
        mj_model, mj_data,
        sensor_body=myosuite_mount.get("sensor_body", IMU_HARDWARE_SPEC["proxy_body"]),
        sensor_site=myosuite_mount.get("sensor_site"),
        sensor_offset_local=myosuite_mount.get("sensor_offset_local"),
        age=age, height=height,
        target_output_hz=IMU_HARDWARE_SPEC["sampling_hz"],
        mount_label=myosuite_mount.get("mount_label", IMU_HARDWARE_SPEC["mount_label"]),
    )

    # -- Controller ------------------------------------------------------------
    print("[D] Initialising EnhancedBiofidelicController …")
    controller = EnhancedBiofidelicController(env, mj_model, mj_data,
                                               anthropometry=age_params)
    controller.current_phase = "stand"

    # Speed target from scenario
    if scenario.initial_state == InitialState.JOGGING:
        controller.walk_target_speed = scenario.jogging_speed
    elif scenario.initial_state == InitialState.MOVING_BACKWARD:
        controller.walk_target_speed = max(0.4, scenario.initial_backward_speed * 1.3)

    # Lie-down reward orientation
    lie_reward = controller.get_lie_down_reward(scenario.lie_orient)

    # -- Layer-1 / Layer-2 exporters -------------------------------------------
    marker_exporter   = MarkerKinematicsExporter(mj_model, mj_data, export_hz=imu.output_hz)
    dynamics_analyzer = DynamicsContactAnalyzer(mj_model, mj_data,
                                                 body_mass=body_mass,
                                                 leg_length=age_params.get("leg_length", height * 0.53))
    paper_exporter    = PaperAlignmentExporter(
        marker_exporter, dynamics_analyzer,
        subject_meta={
            "age_years":    age,
            "height_m":     height,
            "sex":          sex,
            "body_mass_kg": body_mass,
            "fall_type":    scenario.description,
            "scenario_id":  scenario.scenario_id,
        },
    )
    dashboard = PhysicsDashboard(
        mj_model, mj_data, body_mass=body_mass,
        leg_length=age_params.get("leg_length", height * 0.53),
        upright_h=float(mj_data.xpos[mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")][2]),
    )

    # -- Apply initial conditions ----------------------------------------------
    setup_initial_conditions(mj_data, scenario, obs, env)

    # -- Phase boundaries (cumulative step counts) -----------------------------
    phase_names  = ["stand", "walk", "perturb", "react", "fall"]
    phase_bounds = np.cumsum([0] + [phases.get(n, 0) for n in phase_names])

    # -- Perturbation state ----------------------------------------------------
    pert_state = PerturbationState()
    pert_state.original_gear = mj_model.actuator_gear[:, 0].copy()

    # -- Walk speed blend ------------------------------------------------------
    from backward_fall_walking_best import LAST_Z_WALK_DIAGNOSTICS
    if LAST_Z_WALK_DIAGNOSTICS:
        natural_speed = float(LAST_Z_WALK_DIAGNOSTICS.get("mean_vx", controller.walk_target_speed))
        natural_ds    = float(LAST_Z_WALK_DIAGNOSTICS.get("double_support_frac",
                                controller.age_style.get("expected_double_support", 0.25)))
        age_style_now = get_age_style_v2(age, height, sex, body_mass)
        ct = compute_subject_control_targets(age_style_now, natural_speed, natural_ds)
        controller.walk_target_speed      = ct["control_speed"]
        controller.walk_control_expected_ds = ct["control_ds"]

    # -- XCoM history ----------------------------------------------------------
    xcom_history, support_center_history, xcom_timestamps = [], [], []
    _xcom_neg_streak = 0
    _xcom_log_start  = phase_bounds[1] + XCOM_LOG_START_OFFSET_STEPS

    # -- Walk-stability counter (for gait-gate before perturbation) ------------
    walk_stable_counter   = 0
    perturb_start_index   = None
    react_start_index     = None
    fall_start_index      = None
    actual_walk_start     = None
    metrics_log           = []
    _rest_stable_ctr      = 0
    _last_imu_peak        = 0.0
    dynamics_frame        = None

    print(f"\n[4/4] Running simulation (scenario {scenario.scenario_id}) …\n")
    print(f"      Initial state  : {scenario.initial_state}")
    print(f"      Fall direction : {scenario.fall_direction}")
    print(f"      Perturbation   : {scenario.perturbation_type}")
    print()

    # -------------------------------------------------------------------------
    # MAIN SIMULATION LOOP
    # -------------------------------------------------------------------------
    viewer_ctx = (
        mujoco.viewer.launch_passive(mj_model, mj_data)
        if scenario.viewer_enabled else _NullContext()
    )
    with viewer_ctx as viewer:
        if scenario.viewer_enabled:
            viewer.cam.distance  = 4.5
            viewer.cam.elevation = -10
            viewer.cam.azimuth   = 90

        for step in range(total_steps):
            # -- Clear per-step forces -----------------------------------------
            controller.clear_forces()

            # -- Phase assignment ----------------------------------------------
            if step < phase_bounds[1]:
                phase = "stand"
                if step == 0:
                    controller.set_target_z(z_stand, blend_steps=10)
            else:
                # Walk / pre-perturbation
                if actual_walk_start is None:
                    actual_walk_start = step
                    controller.set_target_z(z_pre, blend_steps=30)
                    controller.start_walk_phase()
                    if scenario.initial_state == InitialState.MOVING_BACKWARD:
                        set_backward_velocity(mj_data, scenario.initial_backward_speed)
                    env.unwrapped.set_task(task_str)
                    print(f"  [Step {step}] Pre-perturbation phase started ({scenario.initial_state})")

                if perturb_start_index is None:
                    # Gait gate: wait for stable gait before triggering perturbation
                    phase = "walk"
                    walk_diag = compute_preperturb_walk_metrics(
                        controller, mj_model, mj_data, body_mass, age_params
                    )
                    gait_ok = controller.gait_detector.is_in_window(
                        _map_fall_dir_to_window(scenario.fall_direction)
                    )
                    if walk_diag["ready"] or (step - actual_walk_start) >= (phases.get("walk", 300) + 45):
                        perturb_start_index = step
                        controller.set_target_z(z_fall, blend_steps=40)
                        print(f"  [Step {step}] Perturbation started "
                              f"[type={scenario.perturbation_type}  dir={scenario.fall_direction}]")
                else:
                    perturb_elapsed = step - perturb_start_index
                    perturb_total   = phases.get("perturb", 60)
                    react_total     = phases.get("react", 18)

                    if react_start_index is None and perturb_elapsed >= perturb_total:
                        react_start_index = step
                        print(f"  [Step {step}] Reaction delay phase")

                    if fall_start_index is None and react_start_index is not None:
                        react_elapsed = step - react_start_index
                        if react_elapsed >= react_total:
                            fall_start_index = step
                            controller.clear_forces()
                            controller.finalize_z_transition()
                            controller.update_muscle_weakening(1.0)
                            print(f"  [Step {step}] Full collapse phase")

                    if fall_start_index is not None:
                        phase = "fall"
                    elif react_start_index is not None:
                        phase = "react"
                    else:
                        phase = "perturb"

            # -- Z interpolation -----------------------------------------------
            z_current = controller.update_z_interpolation()
            controller.current_phase = phase

            # -- Perturbation dispatch -----------------------------------------
            if phase in ("perturb", "react"):
                apply_scenario_perturbation(
                    step, phase, controller, mj_model, mj_data,
                    scenario, age_params, body_mass, pert_state,
                    phase_bounds, perturb_start_index or step,
                )
            elif phase == "fall":
                controller.update_muscle_weakening(1.0)

            # -- Action -------------------------------------------------------
            if phase in ("stand", "walk"):
                action = controller.apply_ankle_hip_strategy(obs, z_current)
                controller.apply_age_posture_bias_only()
            else:
                action = controller.get_protective_action(obs, z_current,
                                                           scenario.fall_direction)
            if phase == "walk" and scenario.initial_state not in (
                    InitialState.SITTING_DOWN, InitialState.GETTING_UP,
                    InitialState.SITTING, InitialState.ELEVATED):
                controller.apply_walk_guidance()

            # -- Step environment ----------------------------------------------
            obs, _, terminated, truncated, _ = env.step(action)

            # -- Rest mode -----------------------------------------------------
            if phase == "fall":
                pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")
                vel6 = np.zeros(6)
                mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vel6, 0)
                ph  = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
                slp = float(np.linalg.norm(vel6[3:5]))
                ang = float(np.linalg.norm(vel6[:3]))
                ncon = int(mj_data.ncon)
                if ph < 0.20 and slp < 0.30 and ang < 2.5 and ncon >= 3:
                    _rest_stable_ctr += 1
                else:
                    _rest_stable_ctr = 0
                if _rest_stable_ctr >= 5 and not controller.rest_mode:
                    controller.activate_rest_mode(z_rest)
                    print(f"  [Step {step}] Rest mode engaged")
                if controller.rest_mode:
                    controller.apply_rest_stiction()

            # -- IMU + exporters -----------------------------------------------
            sim_time_now = float(mj_data.time)
            _last_imu_peak = imu.log_frame(sim_time_now)
            marker_exporter.capture_frame(sim_time_now, phase=phase)
            dynamics_frame = dynamics_analyzer.capture_frame(sim_time_now, phase=phase)

            if scenario.viewer_enabled:
                viewer.sync()

            # -- Console logging (every 30 steps or 5 in perturb/fall) ---------
            log_freq = 5 if phase in ("perturb", "react", "fall") else 30
            if step % log_freq == 0:
                pelvis_id  = mujoco.mj_name2id(mj_model, MJOBJ_BODY, "Pelvis")
                pelvis_pos = mj_data.xpos[pelvis_id]
                live_imu   = imu.read_imu(sim_time_now)
                xcom_margin = None
                if phase in ("stand", "walk"):
                    try:
                        xc, sc, xm = controller.compute_xcom()
                        xcom_margin = xm
                        if step >= _xcom_log_start and phase == "walk":
                            xcom_history.append(np.asarray(xc).copy())
                            support_center_history.append(np.asarray(sc).copy())
                            xcom_timestamps.append(sim_time_now)
                    except Exception:
                        pass
                metrics = dashboard.report(
                    step=step, phase=phase,
                    leg_strength=controller.leg_strength,
                    xcom_margin=xcom_margin,
                    fall_predicted=(pelvis_pos[2] < 0.35),
                    imu_peak=_last_imu_peak,
                    sensor_impact=float(live_imu["impact"]),
                    control_source="scenario",
                    dynamics=dynamics_frame,
                )
                metrics.update({"step": step, "phase": phase})
                metrics_log.append(metrics)

            # Safety reset during stand phase
            if terminated and step < phase_bounds[1]:
                obs, _ = env.reset()
                controller.restore_strength()

    # -------------------------------------------------------------------------
    # POST-SIMULATION: VALIDATION + EXPORT
    # -------------------------------------------------------------------------
    print("\n  Post-simulation validation …")

    validator    = FallValidator()
    _perturb_t   = float(perturb_start_index or phase_bounds[2]) * float(imu.native_dt)
    event_summary = detect_fall_events(
        imu.data_buffer, dynamics_analyzer.frames,
        marker_exporter.frames, perturb_start_time=_perturb_t,
    )
    val_results = validator.validate_fall(
        imu.data_buffer,
        {"head_velocity": imu.data_buffer["pelvis_velocity"][-1]
                         if imu.data_buffer["pelvis_velocity"] else [0,0,0]},
        scenario.fall_direction,
        perturb_start_time=_perturb_t,
        event_summary=event_summary,
    )

    sisfall_val = SISFallValidator()
    sf_result   = sisfall_val.validate_sisfall_signature(
        imu.data_buffer, scenario.fall_direction,
        body_mass=body_mass, perturb_start_time=_perturb_t,
        event_summary=event_summary,
    )

    kfall_val  = KFallValidator()
    kf_result  = kfall_val.validate(imu, perturb_start_time=_perturb_t,
                                     event_summary=event_summary)

    print(f"\n  FallValidator score    : {val_results['overall_score']:.1%}  "
          f"({val_results['classification']})")
    print(f"  SISFall compliant      : {sf_result['sisfall_compliant']}")
    print(f"  KFall compliant        : {kf_result.get('kfall_compliant', False)}")

    # -- Scenario authenticity composite --------------------------------------
    walk_m = [m for m in metrics_log if m.get("phase") == "walk"]
    pre_score = 0.5
    if walk_m:
        wv = float(np.mean([abs(m.get("pelvis_vx", 0.0)) for m in walk_m]))
        pre_score = float(np.clip(wv / max(controller.walk_target_speed, 0.2), 0, 1))
    authenticity = (
        0.45 * val_results["overall_score"]
        + 0.25 * pre_score
        + 0.20 * kf_result.get("score", 0.0)
        + 0.10 * (1.0 if sf_result.get("sisfall_compliant", False) else 0.0)
    )
    print(f"  Scenario authenticity  : {authenticity:.1%}")

    # -- CSV / biomechanics export ---------------------------------------------
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    pfx = f"fall_scenario{scenario.scenario_id}_age{age}_{ts}"

    imu_result    = imu.export_to_csv(pfx + ".csv", metadata={
        "age": age, "height": height, "sex": sex, "weight": body_mass,
        "body_mass_kg": body_mass, "fall_type": scenario.description,
        "scenario_id": scenario.scenario_id,
    }) if scenario.save_imu_csv else {}

    mk_bundle     = marker_exporter.export_bundle(pfx)    if scenario.save_biomechanics else {}
    dyn_bundle    = dynamics_analyzer.export_bundle(pfx)  if scenario.save_biomechanics else {}
    paper_bundle  = paper_exporter.export_bundle(pfx)     if scenario.save_biomechanics else {}
    val_report    = validator.generate_report(val_results, pfx + "_validation.txt")

    env.close()
    controller.restore_strength()
    controller.clear_forces()

    summary = {
        "scenario_id":       scenario.scenario_id,
        "description":       scenario.description,
        "classification":    val_results["classification"],
        "overall_score":     val_results["overall_score"],
        "authenticity":      authenticity,
        "sisfall_compliant": sf_result["sisfall_compliant"],
        "kfall_compliant":   kf_result.get("kfall_compliant", False),
        "imu_csv":           imu_result.get("filename", ""),
        "validation_report": pfx + "_validation.txt",
        "event_summary":     event_summary,
    }
    print(f"\n  All outputs written with prefix: {pfx}")
    return summary


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def _map_fall_dir_to_window(fall_direction: str) -> str:
    return {
        FallDirection.BACKWARD:      "backward_walking",
        FallDirection.FORWARD:       "forward_stumble",
        FallDirection.LATERAL_LEFT:  "lateral_left",
        FallDirection.LATERAL_RIGHT: "lateral_right",
    }.get(fall_direction, "backward_walking")


class _NullContext:
    """Context manager that does nothing (used when viewer is disabled)."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def sync(self): pass
