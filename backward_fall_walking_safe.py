# -*- coding: utf-8 -*-
"""
Realistic Backward Fall While Walking - Biofidelic Implementation v25-armquietsettle
Based on: "Human Digital Twin for Realistic Fall Simulation Using Meta Motivo"
Paper: DETC2025-169046

Builds on v6 (93.2% HIGH_CONFIDENCE, all 6 checks passing).
v8 adds two non-invasive improvements - no effect on controller stability:

  ZMP  - Zero Moment Point from contact forces (Vukobratovic & Borovac 2004).
         Displayed alongside XCoM each step as a second stability indicator.
  IMU  - Physics pipeline: anti-aliasing pre-filter ? noise+bias ? soft-tissue
         artifact ? hardware saturation (+/-16g accel, +/-2000 deg/s gyro) ?
         on-chip low-pass filter. Improves CSV signal realism.

  Excluded from v7 (caused instability):
    Hill muscle dynamics  - conflicts with muscle-weakening gear modifications
    Compliant contact     - destabilizes MuJoCo contact solver
    Cascaded neural delays - zero-init queues destabilize initial standing
"""
import sys, os
import re
import numpy as np
import torch
import mujoco
import mujoco.viewer
from collections import deque, defaultdict
from pathlib import Path
from biofidelic_profile_myosuite import get_age_style_v2, get_age_reference_band_v2, apply_age_effects_v2, perturbation_force_config, weakening_config, phase_timing, get_segment_mass_fractions, print_subject_profile
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print('  [Warning] scipy not found - FallValidator jerk/peak checks will be skipped.')
import humenv
from humenv import make_humenv
from humenv.rewards import LocomotionReward, LieDownReward
from metamotivo.fb_cpr.huggingface import FBcprModel
if hasattr(mujoco, 'mjtObj'):
    MJOBJ_BODY = mujoco.mjtObj.mjOBJ_BODY
    MJOBJ_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR
    MJOBJ_GEOM = mujoco.mjtObj.mjOBJ_GEOM
    MJOBJ_SITE = mujoco.mjtObj.mjOBJ_SITE
else:
    MJOBJ_BODY = mujoco.mjOBJ_BODY
    MJOBJ_ACTUATOR = mujoco.mjOBJ_ACTUATOR
    MJOBJ_GEOM = mujoco.mjOBJ_GEOM
    MJOBJ_SITE = getattr(mujoco, 'mjOBJ_SITE', None)

def _norm_name(s):
    return ''.join((ch.lower() for ch in str(s or '') if ch.isalnum()))

def _tokenize_name(s):
    s = str(s or '').replace('-', '_')
    return [tok.lower() for tok in re.split('[^A-Za-z0-9]+', s) if tok]

def _body_world_velocity(mj_model, mj_data, body_id):
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, body_id, vel6, 0)
    return vel6[3:].copy()

def _contact_rotation_world(contact):
    return np.asarray(contact.frame, dtype=float).reshape(3, 3).T

def _contact_wrench_world(contact, wrench_local):
    R = _contact_rotation_world(contact)
    f_world = R @ np.asarray(wrench_local[:3], dtype=float)
    t_world = R @ np.asarray(wrench_local[3:], dtype=float)
    return (f_world, t_world)

def _body_name_matches_side(name, side):
    norm = _norm_name(name)
    toks = _tokenize_name(name)
    left_hit = any((t in ('l', 'left') for t in toks)) or norm.endswith('l') or '_l' in str(name).lower()
    right_hit = any((t in ('r', 'right') for t in toks)) or norm.endswith('r') or '_r' in str(name).lower()
    if side == 'left':
        return left_hit and (not right_hit)
    if side == 'right':
        return right_hit and (not left_hit)
    return True

def _safe_name2id(mj_model, obj_type, name):
    if obj_type is None or not name:
        return -1
    try:
        return int(mujoco.mj_name2id(mj_model, obj_type, str(name)))
    except Exception:
        return -1

def _resolve_site_id(mj_model, candidates):
    if MJOBJ_SITE is None:
        return -1
    best_id = -1
    best_score = -1000000000.0
    for sid in range(getattr(mj_model, 'nsite', 0)):
        sname = mujoco.mj_id2name(mj_model, MJOBJ_SITE, sid) or ''
        norm = _norm_name(sname)
        score = 0.0
        for cand in candidates:
            key = _norm_name(cand)
            if not key:
                continue
            if norm == key:
                score += 100.0
            elif key in norm:
                score += 20.0
        if score > best_score:
            best_score = score
            best_id = sid
    return best_id if best_score >= 15.0 else -1

def _resolve_body_id(mj_model, candidates):
    best_id = -1
    best_score = -1000000000.0
    for bid in range(mj_model.nbody):
        bname = mujoco.mj_id2name(mj_model, MJOBJ_BODY, bid) or ''
        norm = _norm_name(bname)
        score = 0.0
        for cand in candidates:
            key = _norm_name(cand)
            if not key:
                continue
            if norm == key:
                score += 100.0
            elif key in norm:
                score += 20.0
        if score > best_score:
            best_score = score
            best_id = bid
    return best_id if best_score >= 10.0 else -1

def _load_reference_mujoco_model(xml_path):
    xml_path = str(xml_path or '').strip()
    if not xml_path:
        return None
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)
    return mujoco.MjModel.from_xml_path(xml_path)

def resolve_imu_mount_configuration(mj_model, requested_xml_path=''):
    """Resolve a better IMU mount only when an explicit MyoSuite XML is available.

    Without a provided reference XML, the simulator should keep the original Torso/L1-L2
    proxy so behaviour stays comparable to the baseline run.
    """
    report = {'requested_xml_path': str(requested_xml_path or '').strip(), 'reference_loaded': False, 'runtime_mount_source': 'baseline_proxy', 'sensor_body': IMU_HARDWARE_SPEC['proxy_body'], 'sensor_site': None, 'sensor_offset_local': None, 'mount_label': IMU_HARDWARE_SPEC['mount_label'], 'reference_mount': None, 'notes': []}
    site_candidates = ['imu', 'imu_site', 'imu_l1_l2', 'l1_l2', 'lumbar_imu', 'back_imu', 'sacrum_imu', 'pelvis_imu', 'sensor']
    body_candidates = ['L1_L2', 'L1L2', 'lumbar', 'lumbar_body', 'sacrum']
    xml_path = report['requested_xml_path']
    if not xml_path:
        report['notes'].append('No MyoSuite XML provided; keeping the original Torso L1-L2 proxy.')
        return report
    try:
        ref_model = _load_reference_mujoco_model(xml_path)
        report['reference_loaded'] = True
        ref_site_id = _resolve_site_id(ref_model, site_candidates)
        if ref_site_id >= 0:
            ref_body_id = int(ref_model.site_bodyid[ref_site_id])
            ref_body_name = mujoco.mj_id2name(ref_model, MJOBJ_BODY, ref_body_id) or ''
            ref_site_name = mujoco.mj_id2name(ref_model, MJOBJ_SITE, ref_site_id) or ''
            report['reference_mount'] = {'body': ref_body_name, 'site': ref_site_name, 'offset_local': np.asarray(ref_model.site_pos[ref_site_id], dtype=float).copy()}
            runtime_same_site = _safe_name2id(mj_model, MJOBJ_SITE, ref_site_name)
            runtime_same_body = _safe_name2id(mj_model, MJOBJ_BODY, ref_body_name)
            if runtime_same_site >= 0:
                report['sensor_body'] = ref_body_name
                report['sensor_site'] = ref_site_name
                report['sensor_offset_local'] = np.asarray(mj_model.site_pos[runtime_same_site], dtype=float).copy()
                report['runtime_mount_source'] = 'runtime_site_matching_reference'
                report['mount_label'] = f'{ref_site_name} site'
            elif runtime_same_body >= 0:
                report['sensor_body'] = ref_body_name
                report['sensor_offset_local'] = np.asarray(ref_model.site_pos[ref_site_id], dtype=float).copy()
                report['runtime_mount_source'] = 'reference_site_projected_to_runtime_body'
                report['mount_label'] = f'{ref_site_name} guided proxy'
            else:
                report['notes'].append('Reference XML loaded, but no matching body/site exists in the humenv runtime model; keeping baseline Torso proxy.')
        else:
            ref_body_id = _resolve_body_id(ref_model, body_candidates)
            if ref_body_id >= 0:
                ref_body_name = mujoco.mj_id2name(ref_model, MJOBJ_BODY, ref_body_id) or ''
                report['reference_mount'] = {'body': ref_body_name, 'site': None, 'offset_local': None}
                runtime_same_body = _safe_name2id(mj_model, MJOBJ_BODY, ref_body_name)
                if runtime_same_body >= 0:
                    report['sensor_body'] = ref_body_name
                    report['runtime_mount_source'] = 'runtime_body_matching_reference'
                    report['mount_label'] = f'{ref_body_name} body proxy'
                else:
                    report['notes'].append('Reference XML loaded, but no matching lumbar body exists in the humenv runtime model; keeping baseline Torso proxy.')
            else:
                report['notes'].append('Reference XML loaded, but no lumbar/L1-L2 marker was found; keeping baseline Torso proxy.')
        report['notes'].append('Reference MyoSuite XML is used for guidance only; the Meta Motivo runtime body was not replaced.')
    except Exception as exc:
        report['notes'].append(f'MyoSuite XML not loaded: {exc}; keeping the original Torso proxy.')
    return report
SEED = 42
PHASES = {'stand': 150, 'walk': 300, 'perturb': 60, 'react': 15, 'fall': 400}
TOTAL_STEPS = sum(PHASES.values())
FORCE_CONFIG = {'magnitude': 450.0, 'ramp_up': 30, 'direction': np.array([-1.0, 0.0, 0.3]), 'application_point': 'Pelvis'}
WEAKENING_CONFIG = {'initial_factor': 1.0, 'min_factor': 0.05, 'decay_time': 45, 'recovery_time': 120}
Z_INTERPOLATION_STEPS = 20
ACTION_SMOOTHING = 0.7
IMU_HARDWARE_SPEC = {'microcontroller': 'STM32F722RET6 (ARM Cortex-M7, 216 MHz)', 'accelerometer': 'LIS3DH tri-axis MEMS, +/-16g, 1 mg resolution', 'gyroscope': 'LSM6DS3 tri-axis, +/-2000 dps, 0.07 dps resolution', 'sampling_hz': 100.0, 'mount_label': 'Lower back (vertebrae L1-L2)', 'proxy_body': 'Torso', 'proxy_fraction_torso_to_pelvis': 0.4}
MYOSUITE_INTEGRATION = {'enable_reference_xml': str(os.environ.get('MYOSUITE_ENABLE_REFERENCE_XML', '0')).strip().lower() in ('1', 'true', 'yes', 'on'), 'model_xml': str(os.environ.get('MYOSUITE_MODEL_XML', '')).strip(), 'mode': str(os.environ.get('MYOSUITE_MODE', 'reference')).strip().lower() or 'reference'}
PAPER_ALIGNMENT_EXPORT = {'enable_opensim_bridge': True, 'enable_pose_dataset_bridge': True, 'pose_views': ('frontal', 'sagittal', 'oblique'), 'pose_preview_frames': 4, 'render_pose_images': False}
STABILITY_WARMUP_STEPS = 60
XCOM_NEGATIVE_CONFIRM_STEPS = 2
XCOM_LOG_START_OFFSET_STEPS = 30
LAST_Z_WALK_DIAGNOSTICS = {}

# Runtime context injected by shared-core callers so the legacy helper
# functions in this module do not depend on import-time side effects.
model = None

def bind_runtime_context(*, motivo_model=None, age=None, height=None, sex=None, resolved_weight=None):
    """Bind runtime globals expected by the legacy embedding helpers.

    The original scenario-34 code relied on module-level globals. Shared-core
    scenarios must set them explicitly before calling infer_z_* helpers.
    """
    global model, SIM_AGE, SIM_HEIGHT, SIM_SEX, SIM_RESOLVED_WEIGHT
    if motivo_model is not None:
        model = motivo_model
    if age is not None:
        SIM_AGE = int(age)
    if height is not None:
        SIM_HEIGHT = float(height)
    if sex is not None:
        SIM_SEX = str(sex).lower()
    if resolved_weight is not None:
        SIM_RESOLVED_WEIGHT = float(resolved_weight)

def _require_runtime_model():
    if model is None:
        raise RuntimeError(
            'Meta Motivo model is not bound in backward_fall_walking_safe.py. '
            'Call bind_runtime_context(motivo_model=...) before infer_z_*.'
        )
    return model

def _prompt_float(label, default, unit='', lo=None, hi=None):
    while True:
        raw = input(f"    {label} [{default}{(' ' + unit if unit else '')}]: ").strip()
        if raw == '':
            return default
        try:
            val = float(raw)
            if lo is not None and val < lo:
                print(f'      x  Must be >= {lo}. Try again.')
                continue
            if hi is not None and val > hi:
                print(f'      x  Must be <= {hi}. Try again.')
                continue
            return val
        except ValueError:
            print('      x  Please enter a number. Try again.')

def _prompt_str(label, default, choices):
    while True:
        raw = input(f"    {label} [{default}] ({'/'.join(choices)}): ").strip().lower()
        if raw == '':
            return default
        if raw in choices:
            return raw
        print(f'      x  Choose one of: {choices}. Try again.')
BASE_MODEL_HEIGHT_M = 1.75
BASE_LEG_LENGTH_RATIO = 0.53

def _prompt_optional_float(label, unit='', lo=None, hi=None):
    while True:
        raw = input(f"    {label} [auto{(' ' + unit if unit else '')}]: ").strip()
        if raw == '':
            return None
        try:
            val = float(raw)
            if lo is not None and val < lo:
                print(f'      x  Must be >= {lo}. Try again.')
                continue
            if hi is not None and val > hi:
                print(f'      x  Must be <= {hi}. Try again.')
                continue
            return val
        except ValueError:
            print('      x  Please enter a number or press Enter for auto. Try again.')

def estimate_reference_bmi(age_years, sex='male'):
    age = float(age_years)
    sex = str(sex or 'male').lower()
    if age < 18:
        bmi = 19.0 if sex == 'female' else 19.5
    elif age < 40:
        bmi = 22.0 if sex == 'female' else 23.0
    elif age < 65:
        bmi = 23.0 if sex == 'female' else 24.0
    else:
        bmi = 25.5 if sex == 'female' else 26.0
    return float(np.clip(bmi, 18.5, 29.0))

def resolve_subject_weight_kg(height_m, age_years, sex='male', explicit_weight=None):
    target_bmi = estimate_reference_bmi(age_years, sex)
    if explicit_weight is None:
        weight_kg = target_bmi * float(height_m) * float(height_m)
        source = 'auto_bmi_profile'
    else:
        weight_kg = float(explicit_weight)
        target_bmi = weight_kg / max(float(height_m) * float(height_m), 1e-09)
        source = 'user_input'
    return (float(weight_kg), float(target_bmi), source)

def get_age_style(age_years, height_m=None, sex=None, body_mass_kg=None):
    height_m = SIM_HEIGHT if height_m is None else float(height_m)
    sex = (SIM_SEX if sex is None else sex).lower()
    body_mass_kg = SIM_RESOLVED_WEIGHT if body_mass_kg is None else float(body_mass_kg)
    return get_age_style_v2(age_years, height_m=height_m, sex=sex, body_mass_kg=body_mass_kg)

def get_age_reference_band(age_years, sex='male', height_m=None, body_mass_kg=None):
    height_m = SIM_HEIGHT if height_m is None else float(height_m)
    body_mass_kg = SIM_RESOLVED_WEIGHT if body_mass_kg is None else float(body_mass_kg)
    return get_age_reference_band_v2(age_years, sex, height_m, body_mass_kg)

def print_age_behavior_audit(age_years, sex, age_style, adapted_speed, natural_speed, anthro):
    ref = get_age_reference_band(age_years, sex, height_m=anthro.get('height_m', SIM_HEIGHT))
    sp_lo, sp_hi = ref['comfortable_speed_band_mps']
    ds_lo, ds_hi = ref['double_support_band']
    status = 'OK'
    if adapted_speed < 0.7 * sp_lo:
        status = 'LOW_WALK_TARGET'
    print('    ? AgeBehaviourAudit ACTIVE')
    print(f"      audit band               = {ref['label']}")
    print(f'      lit comfortable speed    = {sp_lo:.2f} to {sp_hi:.2f} m/s')
    print(f"      model style target       = {age_style['target_walk_speed']:.2f} m/s")
    print(f'      adapted controller speed = {adapted_speed:.2f} m/s')
    print(f'      z_walk natural speed     = {natural_speed:.2f} m/s')
    print(f'      lit double-support band  = {ds_lo:.0%} to {ds_hi:.0%}')
    print(f"      model expected dbl_sup   = {age_style['expected_double_support']:.0%}")
    print(f"      strength_factor          = {float(anthro.get('strength_factor', 1.0)):.3f}")
    print(f"      reaction_delay           = {float(anthro.get('reaction_delay', 0.0)):.3f} s")
    print(f"      balance_impairment       = {float(anthro.get('balance_impairment', 0.0)):.3f}")
    print(f'      age audit status         = {status}')

def compute_subject_control_targets(age_style, natural_speed, natural_ds):
    """
    Blend the policy-natural speed with the literature-facing subject target.

    v34 methodology:
      - young / middle-aged adults should not be trapped near the slow policy
        natural speed, otherwise they drift and circle instead of committing to
        forward locomotion;
      - older adults still remain closer to the natural policy speed to avoid
        forcing an unrealistically aggressive gait.
    """
    natural_speed = float(max(0.18, natural_speed))
    natural_ds = float(np.clip(natural_ds, 0.05, 0.6))
    control_gain = float(age_style.get('control_speed_gain', 1.0))
    target_speed = float(age_style.get('target_walk_speed', natural_speed))
    blend = float(np.clip(age_style.get('policy_speed_blend', 0.55), 0.0, 1.0))
    desired_speed = blend * natural_speed + (1.0 - blend) * target_speed
    desired_speed *= control_gain
    policy_floor_ratio = float(np.clip(0.86 - 0.4 * blend, 0.58, 0.72))
    speed_floor = max(age_style.get('min_adapted_speed', 0.18), policy_floor_ratio * target_speed, 0.65 * natural_speed)
    speed_ceiling = max(speed_floor + 1e-06, float(age_style.get('max_speed_ratio', 1.8)) * natural_speed, 0.92 * target_speed)
    control_speed = float(np.clip(desired_speed, speed_floor, speed_ceiling))
    control_ds = float(np.clip(0.2 * natural_ds + 0.8 * age_style.get('expected_double_support', 0.3), 0.18, 0.52))
    return {'control_speed': control_speed, 'control_ds': control_ds, 'control_gain': control_gain}

def infer_z_walk_stable(n_samples=1000, age=None):
    global LAST_Z_WALK_DIAGNOSTICS
    '\n    Infer robust walking embedding using weighted regression\n    as described in Meta Motivo paper (Sec 8.5.3).\n\n    v3: reward threshold lowered to 0.3 (was 0.6) and trials increased\n    to 80 to collect enough diverse walking states. Includes a best-of-N\n    fallback if reward inference still yields insufficient data.\n    '
    model_ref = _require_runtime_model()
    age = SIM_AGE if age is None else age
    age_style = get_age_style(age, SIM_HEIGHT, SIM_SEX, body_mass_kg=SIM_RESOLVED_WEIGHT)
    latent_style = get_age_style(35, height_m=1.7, sex='male', body_mass_kg=70.0)
    print(f"  Inferring z_walk (stable locomotion | reference_straight_walk_speed={latent_style['target_walk_speed']:.2f} m/s | subject_age_style={age_style['label']} | subject_target_speed={age_style['target_walk_speed']:.2f} m/s)...")
    reward_fn = LocomotionReward(move_speed=latent_style['target_walk_speed'], move_angle=0, stand_height=1.4)
    env_tmp, _ = make_humenv(task='move-ego-0-2')
    observations, rewards = ([], [])
    for trial in range(80):
        torch.manual_seed(SEED + trial)
        z = model_ref.sample_z(1)
        obs, _ = env_tmp.reset()
        traj_obs, traj_rew = ([], [])
        for step in range(120):
            obs_t = torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model_ref.act(obs_t, z).squeeze(0).numpy()
            obs, _, term, trunc, _ = env_tmp.step(action)
            r = reward_fn.compute(env_tmp.unwrapped.model, env_tmp.unwrapped.data)
            traj_obs.append(obs['proprio'].copy())
            traj_rew.append(r)
            if term or trunc:
                break
        if np.mean(traj_rew) > 0.3:
            observations.extend(traj_obs)
            rewards.extend(traj_rew)
    env_tmp.close()
    if len(observations) < 50:
        print('      Warning: few walking samples collected - using best-z fallback')
        best_z = None
        best_rew = -np.inf
        env_fb, _ = make_humenv(task='move-ego-0-2')
        for trial in range(30):
            torch.manual_seed(SEED + trial + 9000)
            z = model_ref.sample_z(1)
            obs_fb, _ = env_fb.reset()
            trial_rewards = []
            for _ in range(60):
                obs_t = torch.tensor(obs_fb['proprio'], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    act = model_ref.act(obs_t, z).squeeze(0).numpy()
                obs_fb, _, t, tr, _ = env_fb.step(act)
                trial_rewards.append(reward_fn.compute(env_fb.unwrapped.model, env_fb.unwrapped.data))
                if t or tr:
                    break
            if np.mean(trial_rewards) > best_rew:
                best_rew = np.mean(trial_rewards)
                best_z = z.clone()
        env_fb.close()
        print(f'      Best-z fallback reward: {best_rew:.3f}')
        LAST_Z_WALK_DIAGNOSTICS = {'mean_reward': float(best_rew), 'mean_vx': 0.0, 'straightness': 0.0, 'double_support_frac': 0.5, 'gait_score': float(best_rew), 'age_target_speed': float(age_style['target_walk_speed'])}
        return best_z
    obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
    rew_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        z_base = model_ref.reward_inference(obs_tensor, rew_tensor)
        z_candidates = [('base', z_base)]
        torch.manual_seed(SEED + 777)
        for perturb_scale in [0.05, 0.1, 0.15, 0.2]:
            noise = torch.randn_like(z_base) * perturb_scale
            z_candidates.append((f'noise={perturb_scale}', z_base + noise))
        for sample_idx in range(6):
            torch.manual_seed(SEED + 3000 + sample_idx)
            z_candidates.append((f'sample={sample_idx + 1}', model_ref.sample_z(1)))
    print('      Selecting best z_walk via rollout evaluation (expanded multi-objective gait search)...')
    best_z = None
    best_score = -np.inf
    best_reward = -np.inf
    eval_env, _ = make_humenv(task='move-ego-0-2')
    pelvis_id = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'Pelvis')
    foot_left = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'FootL')
    foot_right = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'FootR')
    if foot_left < 0:
        foot_left = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'L_Ankle')
    if foot_right < 0:
        foot_right = mujoco.mj_name2id(eval_env.unwrapped.model, MJOBJ_BODY, 'R_Ankle')

    def _support_flags(model, data, left_id, right_id):
        lsup = False
        rsup = False
        plane_type = getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999)
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = (c.geom1, c.geom2)
            names = ((mujoco.mj_id2name(model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(model, MJOBJ_GEOM, g2) or '').lower())
            is_ground = any(('floor' in n or 'ground' in n or 'plane' in n for n in names)) or any((model.geom_type[g] == plane_type for g in (g1, g2)))
            if not is_ground:
                continue
            ng = g2 if 'floor' in names[0] or 'ground' in names[0] or 'plane' in names[0] or (model.geom_type[g1] == plane_type) else g1
            bg = int(model.geom_bodyid[ng])
            if bg == left_id:
                lsup = True
            elif bg == right_id:
                rsup = True
        return (lsup, rsup)
    for idx, (cand_label, z_cand) in enumerate(z_candidates):
        obs_e, _ = eval_env.reset()
        rollout_rewards = []
        xs, ys, hs, vxy = ([], [], [], [])
        dbl = []
        for _ in range(75):
            obs_t = torch.tensor(obs_e['proprio'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act = model.act(obs_t, z_cand).squeeze(0).numpy()
            obs_e, _, t, tr, _ = eval_env.step(act)
            rollout_rewards.append(reward_fn.compute(eval_env.unwrapped.model, eval_env.unwrapped.data))
            if pelvis_id >= 0:
                p = eval_env.unwrapped.data.xpos[pelvis_id].copy()
                v = _body_world_velocity(eval_env.unwrapped.model, eval_env.unwrapped.data, pelvis_id)
                xs.append(float(p[0]))
                ys.append(float(p[1]))
                hs.append(float(p[2]))
                vxy.append(float(np.linalg.norm(v[:2])))
            if foot_left >= 0 and foot_right >= 0:
                lsup, rsup = _support_flags(eval_env.unwrapped.model, eval_env.unwrapped.data, foot_left, foot_right)
                dbl.append(float(lsup and rsup))
            if t or tr:
                break
        mean_r = float(np.mean(rollout_rewards)) if rollout_rewards else 0.0
        mean_speed = float(np.mean(vxy)) if vxy else 0.0
        if len(xs) > 1:
            dxy = np.array([xs[-1] - xs[0], ys[-1] - ys[0]], dtype=float)
            net_disp = float(np.linalg.norm(dxy))
            heading_xy = dxy / net_disp if net_disp > 1e-06 else np.array([1.0, 0.0], dtype=float)
            step_xy = np.column_stack([np.diff(xs), np.diff(ys)]) if len(xs) > 2 else np.zeros((0, 2), dtype=float)
            path_len = float(np.sum(np.linalg.norm(step_xy, axis=1))) if step_xy.size else 0.0
            straightness = net_disp / max(path_len, 1e-06) if path_len > 1e-06 else 0.0
            lat_axis = np.array([-heading_xy[1], heading_xy[0]], dtype=float)
            lateral_offsets = [float(np.dot(np.array([x - xs[0], y - ys[0]], dtype=float), lat_axis)) for x, y in zip(xs, ys)]
            lateral_rms = float(np.sqrt(np.mean(np.square(lateral_offsets)))) if lateral_offsets else 0.0
        else:
            heading_xy = np.array([1.0, 0.0], dtype=float)
            net_disp = 0.0
            straightness = 0.0
            lateral_rms = 1.0
        ds_frac = float(np.mean(dbl)) if dbl else 0.5
        speed_score = float(np.clip(mean_speed / max(0.7 * age_style['target_walk_speed'], 0.2), 0.0, 1.0))
        ds_target = float(np.clip(age_style.get('expected_double_support', latent_style.get('expected_double_support', 0.26)), 0.18, 0.45))
        ds_score = float(np.clip(1.0 - abs(ds_frac - ds_target) / 0.2, 0.0, 1.0))
        lateral_score = float(np.clip(1.0 - lateral_rms / max(net_disp, 0.35), 0.0, 1.0))
        gait_score = 0.28 * mean_r + 0.28 * straightness + 0.22 * speed_score + 0.12 * ds_score + 0.1 * lateral_score
        print(f'        z_candidate {idx + 1}/{len(z_candidates)} ({cand_label})  reward={mean_r:.3f}  vxy={mean_speed:.3f}  straight={straightness:.3f}  lat_rms={lateral_rms:.3f}  ds={ds_frac:.2f}  score={gait_score:.3f}')
        if gait_score > best_score:
            best_score = gait_score
            best_reward = mean_r
            best_z = z_cand.clone()
            LAST_Z_WALK_DIAGNOSTICS = {'mean_reward': float(mean_r), 'mean_vx': float(mean_speed), 'straightness': float(straightness), 'double_support_frac': float(ds_frac), 'gait_score': float(gait_score), 'age_target_speed': float(age_style['target_walk_speed']), 'heading_xy': heading_xy.tolist(), 'net_disp': float(net_disp), 'lateral_rms': float(lateral_rms)}
    eval_env.close()
    print(f"      Best z_walk selected (score={best_score:.3f} | reward={best_reward:.3f} | vxy={LAST_Z_WALK_DIAGNOSTICS.get('mean_vx', 0.0):.3f})")
    z_walk = best_z
    print(f'      z_walk inferred from {len(observations)} states')
    return z_walk

def infer_z_backward_fall():
    """
    Infer embedding for backward fall using LieDownReward with orient='up'
    and goal inference from collapsed poses (Sec 4.2 of paper)
    """
    model_ref = _require_runtime_model()
    print('  Inferring z_fall (backward collapse)...')
    env_tmp, _ = make_humenv(task='lieonground-up')
    collapsed_obs = []
    for trial in range(10):
        obs, _ = env_tmp.reset()
        zero_action = np.zeros(env_tmp.action_space.shape)
        for step in range(100):
            obs, _, term, trunc, _ = env_tmp.step(zero_action)
            pelvis_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, 'Pelvis')
            pelvis_z = env_tmp.unwrapped.data.xpos[pelvis_id][2]
            if pelvis_z < 0.3:
                collapsed_obs.append(obs['proprio'].copy())
            if term or trunc:
                break
    env_tmp.close()
    if len(collapsed_obs) > 50:
        obs_tensor = torch.tensor(np.array(collapsed_obs[-200:]), dtype=torch.float32)
        with torch.no_grad():
            z_fall = model_ref.goal_inference(obs_tensor).mean(dim=0, keepdim=True)
    else:
        print('      Warning: using fallback goal inference')
        z_fall = model_ref.sample_z(1)
    print(f'      z_fall inferred from {len(collapsed_obs)} collapsed states')
    return z_fall

def infer_z_lie_rest():
    """Infer a stable, low-velocity supine pose for post-impact rest mode."""
    model_ref = _require_runtime_model()
    print('  Inferring z_rest (stable lie-on-ground)...')
    env_tmp, _ = make_humenv(task='lieonground-up')
    stable_obs = []
    pelvis_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, 'Pelvis')
    head_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, 'Head')
    torso_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, 'Torso')
    for trial in range(12):
        obs, _ = env_tmp.reset()
        zero_action = np.zeros(env_tmp.action_space.shape)
        for step in range(140):
            obs, _, term, trunc, _ = env_tmp.step(zero_action)
            pelvis_z = float(env_tmp.unwrapped.data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            torso_z = float(env_tmp.unwrapped.data.xpos[torso_id][2]) if torso_id >= 0 else pelvis_z
            head_z = float(env_tmp.unwrapped.data.xpos[head_id][2]) if head_id >= 0 else torso_z
            qvel_norm = float(np.linalg.norm(env_tmp.unwrapped.data.qvel))
            trunk_lean_deg = 0.0
            if pelvis_id >= 0 and head_id >= 0:
                vec = env_tmp.unwrapped.data.xpos[head_id] - env_tmp.unwrapped.data.xpos[pelvis_id]
                nv = float(np.linalg.norm(vec))
                if nv > 1e-08:
                    trunk_lean_deg = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))))
            is_low = pelvis_z < 0.18 and torso_z < 0.24 and (head_z < 0.4)
            is_quiet = qvel_norm < 1.2
            is_supine_like = trunk_lean_deg > 72.0
            if is_low and is_quiet and is_supine_like:
                stable_obs.append(obs['proprio'].copy())
            if term or trunc:
                break
    env_tmp.close()
    if len(stable_obs) > 50:
        obs_tensor = torch.tensor(np.array(stable_obs[-240:]), dtype=torch.float32)
        with torch.no_grad():
            z_rest = model_ref.goal_inference(obs_tensor).mean(dim=0, keepdim=True)
    else:
        print('      Warning: few supine-rest samples - blending fallback from z_fall')
        z_rest = infer_z_backward_fall()
    print(f'      z_rest inferred from {len(stable_obs)} stable lying states')
    return z_rest

def infer_z_stand():
    """Infer stable standing embedding"""
    model_ref = _require_runtime_model()
    print('  Inferring z_stand...')
    reward_fn = LocomotionReward(move_speed=0.0, move_angle=0, stand_height=1.4)
    env_tmp, _ = make_humenv(task='move-ego-0-0')
    observations, rewards = ([], [])
    for trial in range(30):
        torch.manual_seed(SEED + trial + 1000)
        z = model_ref.sample_z(1)
        obs, _ = env_tmp.reset()
        for step in range(80):
            obs_t = torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model_ref.act(obs_t, z).squeeze(0).numpy()
            obs, _, term, trunc, _ = env_tmp.step(action)
            r = reward_fn.compute(env_tmp.unwrapped.model, env_tmp.unwrapped.data)
            observations.append(obs['proprio'].copy())
            rewards.append(r)
            if term or trunc:
                break
    env_tmp.close()
    obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
    rew_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        z_stand = model_ref.reward_inference(obs_tensor, rew_tensor).mean(dim=0, keepdim=True)
    return z_stand

class GaitPhaseDetector:
    """
    Detects gait phase from contact state + body-frame point kinematics.

    Layer-2 note:
      - avoids direct use of mjData.cvel for user-facing linear velocity
      - uses ground contacts when available, falling back to height thresholds
    """

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.foot_left = self._resolve_body(['L_Foot', 'FootL', 'LeftFoot', 'L_Ankle', 'AnkleL'], side='left', family='foot')
        self.foot_right = self._resolve_body(['R_Foot', 'FootR', 'RightFoot', 'R_Ankle', 'AnkleR'], side='right', family='foot')
        self.phase_history = deque(maxlen=10)
        self.last_phase = 'unknown'

    def _resolve_body(self, candidates, side=None, family=None):
        best_id = -1
        best_score = -1000000000.0
        fam_kw = {'foot': ['foot', 'ankle', 'toe', 'heel'], 'shank': ['shin', 'knee', 'lowerleg']}.get(family, [])
        for bid in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or ''
            norm = _norm_name(name)
            score = 0.0
            for c in candidates:
                key = _norm_name(c)
                if norm == key:
                    score += 100.0
                elif key and key in norm:
                    score += 20.0
            if family and any((k in norm for k in fam_kw)):
                score += 10.0
            if side and _body_name_matches_side(name, side):
                score += 15.0
            elif side:
                score -= 20.0
            if score > best_score:
                best_score = score
                best_id = bid
        return best_id if best_score >= 10.0 else -1

    def _foot_world_velocity(self, body_id):
        if body_id < 0:
            return np.zeros(3)
        return _body_world_velocity(self.mj_model, self.mj_data, body_id)

    def _foot_in_ground_contact(self, body_id):
        if body_id < 0:
            return False
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = (c.geom1, c.geom2)
            b1 = self.mj_model.geom_bodyid[g1]
            b2 = self.mj_model.geom_bodyid[g2]
            names = ((mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g2) or '').lower())
            ground = any(('floor' in n or 'ground' in n or 'plane' in n for n in names)) or any((self.mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2)))
            if not ground:
                continue
            if b1 == body_id or b2 == body_id:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
                if wrench[0] > 1.0:
                    return True
        return False

    def detect_phase(self):
        if self.foot_left < 0 or self.foot_right < 0:
            return 'unknown'
        left_pos = self.mj_data.xpos[self.foot_left]
        right_pos = self.mj_data.xpos[self.foot_right]
        left_vel = self._foot_world_velocity(self.foot_left)
        right_vel = self._foot_world_velocity(self.foot_right)
        left_on_ground = self._foot_in_ground_contact(self.foot_left) or left_pos[2] < 0.06
        right_on_ground = self._foot_in_ground_contact(self.foot_right) or right_pos[2] < 0.06
        if left_on_ground and right_on_ground:
            if left_vel[2] < -0.08 or right_vel[2] < -0.08:
                phase = 'heel_strike'
            else:
                phase = 'double_support'
        elif left_on_ground and (not right_on_ground):
            phase = 'mid_stance' if right_vel[2] < 0.05 else 'terminal_stance'
        elif right_on_ground and (not left_on_ground):
            phase = 'mid_stance' if left_vel[2] < 0.05 else 'terminal_stance'
        else:
            phase = 'swing'
        self.phase_history.append(phase)
        self.last_phase = phase
        return phase

    def get_optimal_perturbation_window(self, fall_type):
        windows = {'backward_walking': ['mid_stance', 'terminal_stance'], 'forward_stumble': ['swing', 'heel_strike'], 'lateral_left': ['mid_stance'], 'lateral_right': ['mid_stance'], 'slip_induced': ['heel_strike', 'mid_stance']}
        return windows.get(fall_type, ['mid_stance'])

    def is_in_window(self, fall_type):
        return self.detect_phase() in self.get_optimal_perturbation_window(fall_type)

class BiofidelicFallController:
    """
    Manages realistic fall dynamics including:
    - Reaction time delays
    - Progressive muscle weakening
    - Smooth policy transitions
    - Orientation-aware LieDownReward factory
    - GaitPhaseDetector for perturbation timing
    """

    def __init__(self, env, mj_model, mj_data):
        self.env = env
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.original_gear = mj_model.actuator_gear[:, 0].copy()
        self.original_ctrlrange = mj_model.actuator_ctrlrange.copy()
        self.leg_actuators = self._get_leg_actuators()
        self.arm_actuators = self._get_arm_actuators()
        self.torso_actuators = self._get_torso_actuators()
        self.current_z = None
        self.target_z = None
        self.z_blend = 0.0
        self.action_smoothing = deque(maxlen=3)
        self.leg_strength = 1.0
        self.force_applied = 0.0
        self.original_dof_damping = mj_model.dof_damping.copy()
        self._ground_geoms = self._detect_ground_geoms()
        self._geom_friction_orig = {g: mj_model.geom_friction[g].copy() for g in range(mj_model.ngeom)}
        self._configure_contact_solver()
        self.rest_mode = False
        self.rest_mode_hard_lock = False
        self.rest_counter = 0
        self.impact_brake_mode = False
        self.impact_anchor_xy = None
        self.rest_anchor_xy = None
        self.impact_brake_mode = False
        self.impact_anchor_xy = None
        self._rest_arm_scale = 0.04
        self._rest_leg_scale = 0.24
        self._rest_torso_scale = 0.28
        self.gait_detector = GaitPhaseDetector(mj_model, mj_data)
        self.perturbation_applied = False
        self.waiting_for_phase = False
        self._lie_orient = 'up'

    def _get_leg_actuators(self):
        indices = []
        keywords = ['hip', 'knee', 'ankle', 'foot', 'leg']
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_ACTUATOR, i)
            if name and any((k in name.lower() for k in keywords)):
                indices.append(i)
        return indices

    def _get_arm_actuators(self):
        indices = []
        keywords = ['shoulder', 'elbow', 'wrist', 'hand', 'arm']
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_ACTUATOR, i)
            if name and any((k in name.lower() for k in keywords)):
                indices.append(i)
        return indices

    def _get_torso_actuators(self):
        indices = []
        keywords = ['torso', 'spine', 'abdomen', 'chest']
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_ACTUATOR, i)
            if name and any((k in name.lower() for k in keywords)):
                indices.append(i)
        return indices

    def _detect_ground_geoms(self):
        plane_type = getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', None)
        ground = set()
        for g in range(self.mj_model.ngeom):
            gname = (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g) or '').lower()
            is_ground = 'floor' in gname or 'ground' in gname or 'plane' in gname or (plane_type is not None and self.mj_model.geom_type[g] == plane_type)
            if is_ground:
                ground.add(g)
        return ground

    def _configure_contact_solver(self):
        opt = getattr(self.mj_model, 'opt', None)
        if opt is None:
            return
        if hasattr(opt, 'iterations'):
            opt.iterations = max(int(opt.iterations), 120)
        if hasattr(opt, 'ls_iterations'):
            opt.ls_iterations = max(int(opt.ls_iterations), 50)
        if hasattr(opt, 'noslip_iterations'):
            opt.noslip_iterations = max(int(opt.noslip_iterations), 16)

    def _geom_body_name(self, gid):
        bid = int(self.mj_model.geom_bodyid[int(gid)])
        return mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or f'body_{bid}'

    def _is_foot_like_body(self, body_name):
        n = _norm_name(body_name)
        return any((k in n for k in ('foot', 'ankle', 'toe', 'heel')))

    def has_nonfoot_ground_contact(self, min_normal_n=25.0):
        for i in range(int(self.mj_data.ncon)):
            c = self.mj_data.contact[i]
            g1, g2 = (int(c.geom1), int(c.geom2))
            if g1 in self._ground_geoms and g2 not in self._ground_geoms:
                non_ground = g2
            elif g2 in self._ground_geoms and g1 not in self._ground_geoms:
                non_ground = g1
            else:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            normal_n = float(max(0.0, wrench[0]))
            if normal_n < float(min_normal_n):
                continue
            if not self._is_foot_like_body(self._geom_body_name(non_ground)):
                return True
        return False

    def activate_impact_brake(self):
        if self.impact_brake_mode:
            return
        self.impact_brake_mode = True
        self.impact_anchor_xy = self.mj_data.qpos[:2].copy() if self.mj_data.qpos.shape[0] >= 2 else None

    def apply_impact_brake(self):
        if not self.impact_brake_mode or self.mj_data.qvel.shape[0] < 6:
            return
        self.mj_data.qvel[0] *= 0.35
        self.mj_data.qvel[1] *= 0.35
        self.mj_data.qvel[2] *= 0.92
        self.mj_data.qvel[3] *= 0.78
        self.mj_data.qvel[4] *= 0.78
        self.mj_data.qvel[5] *= 0.82
        if self.impact_anchor_xy is not None and self.mj_data.qpos.shape[0] >= 2:
            self.mj_data.qpos[0] = 0.92 * float(self.mj_data.qpos[0]) + 0.08 * float(self.impact_anchor_xy[0])
            self.mj_data.qpos[1] = 0.92 * float(self.mj_data.qpos[1]) + 0.08 * float(self.impact_anchor_xy[1])

    def activate_rest_mode(self, z_rest=None):
        if self.rest_mode:
            return
        self.activate_impact_brake()
        self.rest_mode = True
        self.rest_mode_hard_lock = False
        self.rest_counter = 0
        self.rest_anchor_xy = self.mj_data.qpos[:2].copy() if self.mj_data.qpos.shape[0] >= 2 else None
        self.clear_forces()
        if z_rest is not None:
            self.set_target_z(z_rest, blend_steps=18)
        self.mj_model.dof_damping[:] = self.original_dof_damping * 2.4
        for g in range(self.mj_model.ngeom):
            fr = self._geom_friction_orig[g].copy()
            if g in self._ground_geoms:
                fr[0] = max(fr[0], 7.0)
                fr[1] = max(fr[1], 0.18)
                fr[2] = max(fr[2], 0.05)
            else:
                fr[0] = max(fr[0], 3.5)
                fr[1] = max(fr[1], 0.1)
                fr[2] = max(fr[2], 0.03)
            self.mj_model.geom_friction[g] = fr
        for idx in self.leg_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * self._rest_leg_scale
        for idx in self.arm_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * self._rest_arm_scale
        for idx in self.torso_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * self._rest_torso_scale

    def harden_rest_mode(self):
        if not self.rest_mode:
            return
        self.rest_mode_hard_lock = True

    def apply_rest_stiction(self):
        if not self.rest_mode or self.mj_data.qvel.shape[0] < 6:
            return
        self.mj_data.qvel[0] *= 0.01
        self.mj_data.qvel[1] *= 0.01
        self.mj_data.qvel[2] *= 0.85
        self.mj_data.qvel[3] *= 0.45
        self.mj_data.qvel[4] *= 0.45
        self.mj_data.qvel[5] *= 0.6
        if self.rest_mode_hard_lock:
            self.mj_data.qvel[0] = 0.0
            self.mj_data.qvel[1] = 0.0
            self.mj_data.qvel[5] = 0.0
            if self.rest_anchor_xy is not None and self.mj_data.qpos.shape[0] >= 2:
                self.mj_data.qpos[0] = float(self.rest_anchor_xy[0])
                self.mj_data.qpos[1] = float(self.rest_anchor_xy[1])

    def restore_passive_contact(self):
        self.mj_model.dof_damping[:] = self.original_dof_damping
        for g in range(self.mj_model.ngeom):
            self.mj_model.geom_friction[g] = self._geom_friction_orig[g].copy()
        self.rest_mode = False
        self.rest_mode_hard_lock = False
        self.rest_counter = 0

    def set_target_z(self, z_target, blend_steps=Z_INTERPOLATION_STEPS):
        if self.current_z is None:
            self.current_z = z_target.clone()
            self.target_z = z_target.clone()
            self.z_blend = 1.0
        else:
            self.target_z = z_target.clone()
            self.z_blend = 0.0
            self.blend_steps = blend_steps
            self.blend_counter = 0

    def update_z_interpolation(self):
        if self.z_blend < 1.0:
            self.blend_counter += 1
            t = self.blend_counter / self.blend_steps
            t_smooth = t * t * (3 - 2 * t)
            self.z_blend = t_smooth
            return (1 - self.z_blend) * self.current_z + self.z_blend * self.target_z
        return self.target_z

    def finalize_z_transition(self):
        self.current_z = self.target_z.clone()
        self.z_blend = 1.0

    def apply_external_force(self, magnitude, direction, ramp_progress):
        body_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, FORCE_CONFIG['application_point'])
        if body_id >= 0:
            force_factor = 0.5 * (1 - np.cos(ramp_progress * np.pi))
            self.force_applied = magnitude * force_factor
            self.mj_data.xfrc_applied[body_id, :3] = self.force_applied * direction
            self.mj_data.xfrc_applied[body_id, 3:] = [0, force_factor * 20, 0]

    def clear_forces(self):
        self.mj_data.xfrc_applied[:] = 0
        self.force_applied = 0

    def update_muscle_weakening(self, phase_progress, weakening_type='exponential'):
        phase_progress = float(np.clip(phase_progress, 0.0, 1.0))
        if weakening_type == 'exponential':
            fatigue_k = float(WEAKENING_CONFIG.get('fatigue_coefficient', 1.0))
            effort_scale = float(WEAKENING_CONFIG.get('active_effort_scale', 1.8))
            decay_time = float(max(WEAKENING_CONFIG.get('decay_time', 45), 1.0))
            phase_window_steps = float(max(PHASES.get('perturb', 60) + PHASES.get('react', 15), 1))
            effort_integral = phase_progress * effort_scale * (phase_window_steps / decay_time)
            self.leg_strength = WEAKENING_CONFIG['initial_factor'] * np.exp(-fatigue_k * effort_integral)
            self.leg_strength = max(self.leg_strength, WEAKENING_CONFIG['min_factor'])
        else:
            self.leg_strength = WEAKENING_CONFIG['initial_factor'] - phase_progress * (WEAKENING_CONFIG['initial_factor'] - WEAKENING_CONFIG['min_factor'])
        for idx in self.leg_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * self.leg_strength
        torso_strength = 0.5 + 0.5 * self.leg_strength
        for idx in self.torso_actuators:
            self.mj_model.actuator_gear[idx, 0] = self.original_gear[idx] * torso_strength

    def restore_strength(self):
        self.restore_passive_contact()
        self.mj_model.actuator_gear[:, 0] = self.original_gear.copy()
        self.leg_strength = 1.0

    def get_action(self, obs, z):
        obs_t = torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model.act(obs_t, z).squeeze(0).numpy()
        if self.rest_mode:
            if self.rest_mode_hard_lock:
                action[:] = 0.0
            else:
                action[self.leg_actuators] *= 0.18
                action[self.arm_actuators] *= 0.02
                action[self.torso_actuators] *= 0.08
        elif self.current_phase in ('stand', 'walk'):
            if self.current_phase == 'stand':
                leg_gain = float(self.age_style.get('stand_leg_gain', self.age_style.get('walk_leg_gain', 1.0)))
            else:
                leg_gain = float(self.age_style.get('walk_leg_gain', 1.0))
                pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
                if pelvis_id >= 0:
                    vel = _body_world_velocity(self.mj_model, self.mj_data, pelvis_id)
                    speed_err = self.walk_target_speed - float(vel[0])
                    leg_gain *= float(np.clip(1.0 + 0.06 * speed_err, 0.98, 1.06))
                leg_gain *= float(np.clip(getattr(self, 'walk_leg_gain_boost', 1.0), 0.95, 1.22))
            action[self.leg_actuators] *= leg_gain
            action[self.arm_actuators] *= self.age_style['arm_gain']
        else:
            pelvis_low = bool(self._pelvis_id >= 0 and self.mj_data.xpos[self._pelvis_id][2] < 0.18)
            grounded = bool(int(self.mj_data.ncon) >= 3 or self.has_nonfoot_ground_contact(min_normal_n=15.0))
            if pelvis_low and grounded:
                action[self.arm_actuators] *= 0.02
                action[self.torso_actuators] *= 0.18
                action[self.leg_actuators] *= 0.3
        self.action_smoothing.append(action)
        if len(self.action_smoothing) >= 2:
            if self.rest_mode:
                weights = np.array([0.6, 0.4])
            else:
                weights = self.age_style['smoothing_weights'] if self.current_phase in ('stand', 'walk') else np.array([0.3, 0.7])
            action = np.average(list(self.action_smoothing)[-2:], weights=weights, axis=0)
        return action

    def compute_zmp(self):
        """
        Contact-frame forces are converted to world frame before computing
        the ground-plane load centroid. On a flat floor this is effectively a
        CoP-style ZMP approximation and is much more interpretable than using
        raw contact-frame components directly.
        """
        total_fz = 0.0
        zmp_x_sum = 0.0
        zmp_y_sum = 0.0
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = (c.geom1, c.geom2)
            names = ((mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g2) or '').lower())
            is_ground = any(('floor' in n or 'ground' in n or 'plane' in n for n in names)) or any((self.mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2)))
            if not is_ground:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            f_world, _ = _contact_wrench_world(c, wrench)
            fz = max(0.0, float(f_world[2]))
            if fz <= 0.5:
                continue
            total_fz += fz
            zmp_x_sum += float(c.pos[0]) * fz
            zmp_y_sum += float(c.pos[1]) * fz
        pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        if total_fz < 1.0:
            return (self.mj_data.xpos[pelvis_id][:2].copy(), False, -0.5)
        zmp_2d = np.array([zmp_x_sum / total_fz, zmp_y_sum / total_fz])
        support_center = self._estimate_support_polygon_center()
        margin = 0.15 - float(np.linalg.norm(zmp_2d - support_center))
        return (zmp_2d, margin >= 0.0, float(margin))

    def get_lie_down_reward(self, orient='up'):
        """
        Create an orientation-aware reward function based on fall direction.
        Supine  (orient='up'/'backward')  -> _create_supine_reward()
        Prone   (orient='down'/'forward') -> _create_prone_reward()
        Lateral (orient='left'/'right')   -> _create_lateral_reward()
        """
        self._lie_orient = orient
        if orient in ('up', 'backward'):
            return self._create_supine_reward()
        elif orient in ('down', 'forward'):
            return self._create_prone_reward()
        elif orient in ('left', 'right'):
            return self._create_lateral_reward(orient)
        else:
            return LieDownReward()

    def _create_supine_reward(self):
        """Custom reward for supine (backward) fall - lying on back."""

        def reward_fn(mdl, dat):
            pelvis_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Pelvis')
            torso_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Torso')
            head_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Head')
            pelvis_z = dat.xpos[pelvis_id][2]
            torso_z = dat.xpos[torso_id][2]
            head_z = dat.xpos[head_id][2]
            height_reward = -0.5 * (pelvis_z + torso_z)
            flatness = -abs(torso_z - (head_z + pelvis_z) / 2)
            torso_supine = 1.0 if torso_z < head_z else -1.0
            return height_reward + 0.3 * flatness + 0.2 * torso_supine
        return reward_fn

    def _create_prone_reward(self):
        """Custom reward for prone (forward) fall - face down with pivot."""

        def reward_fn(mdl, dat):
            pelvis_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Pelvis')
            head_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Head')
            pelvis_pos = dat.xpos[pelvis_id]
            head_pos = dat.xpos[head_id]
            forward_displacement = head_pos[0] - pelvis_pos[0]
            height = (pelvis_pos[2] + head_pos[2]) / 2
            orientation = -abs(head_pos[2] - pelvis_pos[2] + 0.1)
            pivot_reward = max(0, forward_displacement)
            return -height + 0.5 * orientation + pivot_reward
        return reward_fn

    def _create_lateral_reward(self, side):
        """Custom reward for lateral (side) fall."""

        def reward_fn(mdl, dat):
            pelvis_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Pelvis')
            pelvis_pos = dat.xpos[pelvis_id]
            lateral = pelvis_pos[1] if side == 'left' else -pelvis_pos[1]
            return -pelvis_pos[2] + 0.3 * lateral
        return reward_fn

class AnthropometricModel:
    """
    Customizable digital human twin with age and height parameters.

    Weight can optionally be user-provided; otherwise an age/sex BMI prior is
    resolved into a subject mass while keeping the same model structure.
    """
    WINTER_MASS_FRACTIONS = {'head': 0.081, 'trunk': 0.497, 'thorax': 0.497, 'torso': 0.497, 'spine': 0.497, 'abdomen': 0.146, 'pelvis': 0.142, 'upperarm': 0.028, 'forearm': 0.016, 'hand': 0.006, 'thigh': 0.1, 'shank': 0.0465, 'foot': 0.0145}

    def __init__(self, mj_model, age=35, height=1.75, weight=None, sex='male'):
        self.mj_model = mj_model
        self.age = age
        self.height = height
        self.sex = sex
        self.WINTER_MASS_FRACTIONS = get_segment_mass_fractions(self.sex)
        self.original_body_mass = mj_model.body_mass.copy()
        self.original_body_pos = mj_model.body_pos.copy()
        self.original_gear = mj_model.actuator_gear[:, 0].copy()
        self.original_geom_pos = mj_model.geom_pos.copy()
        self.original_geom_size = mj_model.geom_size.copy()
        self.original_jnt_pos = mj_model.jnt_pos.copy()
        self.original_inertia = mj_model.body_inertia.copy()
        self.original_body_ipos = mj_model.body_ipos.copy() if hasattr(mj_model, 'body_ipos') else None
        self.original_geom_fromto = mj_model.geom_fromto.copy() if hasattr(mj_model, 'geom_fromto') else None
        self.default_body_mass = float(np.sum(self.original_body_mass))
        self.use_default_weight = weight is None
        self.weight = self.default_body_mass if weight is None else float(weight)
        self.height_scale = height / 1.75
        self.mass_scale = 1.0 if self.use_default_weight else self.weight / 70.0
        self.apply_scaling()
        self.apply_age_effects()

    def _get_winter_fraction(self, body_name):
        if body_name is None:
            return None
        bname = body_name.lower()
        priority_order = ['upperarm', 'forearm', 'abdomen', 'thorax', 'pelvis', 'trunk', 'torso', 'spine', 'thigh', 'shank', 'foot', 'head', 'hand']
        for key in priority_order:
            if key in bname:
                return self.WINTER_MASS_FRACTIONS[key]
        return None

    def _segment_scale_profile(self):
        hs = float(np.clip(self.height / 1.75, 0.9, 1.1))
        age_frac = float(np.clip((self.age - 30.0) / 50.0, 0.0, 1.0))
        female = str(self.sex).lower() == 'female'
        pelvis_width = float(np.clip(1.0 + 0.35 * (hs - 1.0) + (0.04 if female else 0.0), 0.92, 1.1))
        shoulder_width = float(np.clip(1.0 + 0.45 * (hs - 1.0) + (0.03 if female else 0.0) - 0.02 * age_frac, 0.92, 1.1))
        trunk_length = float(np.clip(1.0 + 0.55 * (hs - 1.0) - 0.02 * age_frac, 0.92, 1.1))
        neck_head = float(np.clip(1.0 + 0.2 * (hs - 1.0), 0.95, 1.06))
        upperarm = float(np.clip(1.0 + 0.75 * (hs - 1.0), 0.92, 1.1))
        forearm = float(np.clip(1.0 + 0.78 * (hs - 1.0), 0.92, 1.1))
        thigh = float(np.clip(1.0 + 0.82 * (hs - 1.0), 0.92, 1.1))
        shank = float(np.clip(1.0 + 0.84 * (hs - 1.0), 0.92, 1.1))
        foot = float(np.clip(1.0 + 0.7 * (hs - 1.0), 0.92, 1.08))
        return {'pelvis_width': pelvis_width, 'shoulder_width': shoulder_width, 'trunk_length': trunk_length, 'neck_head': neck_head, 'upperarm': upperarm, 'forearm': forearm, 'thigh': thigh, 'shank': shank, 'foot': foot}

    def _body_scale_vector(self, body_name):
        lname = (body_name or '').lower()
        prof = self._segment_scale_profile()
        if 'pelvis' in lname:
            return np.array([1.0, prof['pelvis_width'], 1.0], dtype=float)
        if any((k in lname for k in ('torso', 'trunk', 'spine', 'abdomen', 'thorax'))):
            return np.array([1.0, prof['shoulder_width'], prof['trunk_length']], dtype=float)
        if 'head' in lname or 'neck' in lname:
            return np.array([1.0, 1.0, prof['neck_head']], dtype=float)
        if 'shoulder' in lname:
            return np.array([1.0, prof['shoulder_width'], prof['trunk_length']], dtype=float)
        if 'elbow' in lname or 'upperarm' in lname:
            return np.array([prof['upperarm'], prof['upperarm'], prof['upperarm']], dtype=float)
        if any((k in lname for k in ('hand', 'wrist', 'forearm'))):
            return np.array([prof['forearm'], prof['forearm'], prof['forearm']], dtype=float)
        if 'hip' in lname:
            return np.array([1.0, prof['pelvis_width'], 1.0], dtype=float)
        if 'knee' in lname or 'thigh' in lname:
            return np.array([prof['thigh'], prof['thigh'], prof['thigh']], dtype=float)
        if 'ankle' in lname or 'shank' in lname:
            return np.array([prof['shank'], prof['shank'], prof['shank']], dtype=float)
        if any((k in lname for k in ('toe', 'foot', 'heel'))):
            return np.array([prof['foot'], prof['foot'], prof['foot']], dtype=float)
        return np.array([self.height_scale, self.height_scale, self.height_scale], dtype=float)

    def apply_allometric_kinematic_scaling(self):
        matched_bodies = 0
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i)
            scale_vec = self._body_scale_vector(name)
            self.mj_model.body_pos[i] = self.original_body_pos[i] * scale_vec
            if self.original_body_ipos is not None:
                self.mj_model.body_ipos[i] = self.original_body_ipos[i] * scale_vec
            if np.max(np.abs(scale_vec - 1.0)) > 1e-06:
                matched_bodies += 1
        for j in range(self.mj_model.njnt):
            bid = int(self.mj_model.jnt_bodyid[j])
            bname = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid)
            scale_vec = self._body_scale_vector(bname)
            self.mj_model.jnt_pos[j] = self.original_jnt_pos[j] * scale_vec
        for g in range(self.mj_model.ngeom):
            bid = int(self.mj_model.geom_bodyid[g])
            bname = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid)
            scale_vec = self._body_scale_vector(bname)
            self.mj_model.geom_pos[g] = self.original_geom_pos[g] * scale_vec
            if self.original_geom_fromto is not None:
                fromto = self.original_geom_fromto[g].copy()
                if np.linalg.norm(fromto) > 0:
                    self.mj_model.geom_fromto[g, :3] = fromto[:3] * scale_vec
                    self.mj_model.geom_fromto[g, 3:] = fromto[3:] * scale_vec
        return matched_bodies

    def apply_scaling(self):
        kinematic_matches = self.apply_allometric_kinematic_scaling()
        matched_bodies = set()
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i)
            frac = self._get_winter_fraction(name)
            orig_m = self.original_body_mass[i]
            if self.use_default_weight:
                self.mj_model.body_mass[i] = orig_m
                if frac is not None and orig_m > 1e-06:
                    matched_bodies.add(i)
            elif frac is not None and orig_m > 1e-06:
                target_mass = self.weight * frac
                scale_factor = target_mass / 70.0 / (orig_m / 70.0) if orig_m > 0 else self.mass_scale
                scale_factor = float(np.clip(scale_factor, 0.3, 5.0))
                self.mj_model.body_mass[i] = orig_m * scale_factor
                matched_bodies.add(i)
            else:
                self.mj_model.body_mass[i] = orig_m * self.mass_scale
            mass_ratio = self.mj_model.body_mass[i] / max(orig_m, 1e-09)
            inertia_scale = mass_ratio * self.height_scale ** 2.2
            self.mj_model.body_inertia[i] = self.original_inertia[i] * inertia_scale
        for geom_id in range(self.mj_model.ngeom):
            geom_type = self.mj_model.geom_type[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                self.mj_model.geom_size[geom_id][0] *= self.height_scale
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                self.mj_model.geom_size[geom_id][0] *= self.height_scale
                self.mj_model.geom_size[geom_id][1] *= self.height_scale
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self.mj_model.geom_size[geom_id] *= self.height_scale
        if not self.use_default_weight:
            current_total_mass = float(np.sum(self.mj_model.body_mass))
            if current_total_mass > 1e-09:
                mass_fix = float(self.weight / current_total_mass)
                self.mj_model.body_mass[:] *= mass_fix
                self.mj_model.body_inertia[:] *= mass_fix
        mode = 'default model body mass' if self.use_default_weight else 'custom mass scaling'
        print(f'      [Anthropometry] {mode}: mass-matched={len(matched_bodies)}/{self.mj_model.nbody} | kinematic-anchor-rebuild={kinematic_matches}/{self.mj_model.nbody}')

    def apply_age_effects(self):
        total_mass = float(np.sum(self.mj_model.body_mass))
        params = apply_age_effects_v2(mj_model=self.mj_model, age=self.age, sex=self.sex, height=self.height, body_mass_kg=total_mass, original_gear=self.original_gear)
        params['default_body_mass_kg'] = self.default_body_mass
        return params

class IMUValidator:
    """
    Hardware target:
      - MCU:  STM32F722RET6
      - Acc:  LIS3DH  +/-16 g
      - Gyro: LSM6DS3 +/-2000 dps
      - Output: 100 Hz
      - Placement: lower back (vertebrae L1-L2 proxy in the MuJoCo body tree)

    Physics notes:
      1. MuJoCo c-quantities are 6D spatial vectors with angular first, linear second.
         For user-facing kinematics we therefore prefer mj_objectVelocity /
         mj_objectAcceleration over directly indexing cvel/cacc.
      2. The model does not expose an anatomical L1-L2 site, so we use a rigid-body proxy:
         Torso body + offset interpolated toward Pelvis.
      3. If native env stepping is ~30 Hz, exported 100 Hz data is a uniform resampling
         of the native stream, not a true hardware-equivalent 100 Hz measurement.
    """
    IMU_CLIP_MS2 = 16.0 * 9.81

    def __init__(self, mj_model, mj_data, sensor_body='Torso', age=35, height=1.75, target_output_hz=100.0, mount_label=None, sensor_offset_local=None, sensor_site=None):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.age = age
        self.height = height
        self.target_output_hz = float(target_output_hz)
        self.mount_label = mount_label or IMU_HARDWARE_SPEC['mount_label']
        self.sensor_site = None
        site_id = _safe_name2id(mj_model, MJOBJ_SITE, sensor_site) if sensor_site else -1
        if site_id >= 0:
            body_id = int(mj_model.site_bodyid[site_id])
            sensor_body = mujoco.mj_id2name(mj_model, MJOBJ_BODY, body_id) or sensor_body
            self.sensor_site = mujoco.mj_id2name(mj_model, MJOBJ_SITE, site_id) or str(sensor_site)
            if sensor_offset_local is None:
                sensor_offset_local = np.asarray(mj_model.site_pos[site_id], dtype=float).copy()
        else:
            body_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, sensor_body)
            if body_id < 0:
                fallback_body = 'Pelvis'
                body_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, fallback_body)
                sensor_body = fallback_body
        if body_id < 0:
            raise ValueError('No suitable MuJoCo body found for IMU mounting (Torso/Pelvis missing).')
        self.sensor_body = sensor_body
        self.body_id = int(body_id)
        self.body_name = mujoco.mj_id2name(mj_model, MJOBJ_BODY, self.body_id) or sensor_body
        self.pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'Pelvis')
        opt_dt = float(mj_model.opt.timestep)
        try:
            ns = int(mj_model.opt.iterations) if hasattr(mj_model.opt, 'iterations') else 0
        except Exception:
            ns = 0
        if opt_dt > 0 and opt_dt < 0.01:
            ns = max(1, round(0.0333 / opt_dt))
        else:
            ns = 1
        self.dt = float(opt_dt * ns)
        if self.dt < 0.0001 or self.dt > 0.5:
            self.dt = 1.0 / 30.0
        self.native_dt = self.dt
        self.native_hz = 1.0 / self.native_dt if self.native_dt > 0 else 0.0
        self.output_hz = float(target_output_hz)
        self.output_dt = 1.0 / self.output_hz
        self.native_is_100hz = abs(self.native_hz - self.output_hz) < 1.0
        self.output_is_100hz = abs(self.output_hz - 100.0) < 1e-09
        self.output_mode = 'native' if self.native_is_100hz else 'resampled_from_native'
        self.effective_bandwidth_hz = 0.5 * self.native_hz
        self.sensor_offset_local = np.asarray(sensor_offset_local, dtype=float).copy() if sensor_offset_local is not None else self._infer_l1l2_proxy_offset()
        self._last_read_time = None
        self._last_read_sample = None
        self._last_export_verification = None
        mount_desc = f'site={self.sensor_site} | body={self.body_name}' if self.sensor_site else f'body={self.body_name}'
        print(f'      [IMU] {mount_desc} (id={self.body_id})  opt_dt={opt_dt:.5f}s  nsubsteps~{ns}  native_dt={self.native_dt:.5f}s  native_hz={self.native_hz:.2f}  output_hz={self.output_hz:.0f}  mode={self.output_mode}')
        self.data_buffer = {'timestamp': [], 'accelerometer': [], 'gyroscope': [], 'quaternion': [], 'pelvis_height': [], 'pelvis_velocity': [], 'impact_force': [], 'soft_tissue_artifact': [], 'sensor_confidence': [], 'accel_raw': [], 'sensor_world_pos': [], 'sensor_world_vel': []}
        age_factor = max(1.0, 1.0 + (age - 60) * 0.02) if age > 60 else 1.0
        self.accel_bias = np.random.normal(0, 0.03 * age_factor, 3)
        self.gyro_bias = np.random.normal(0, 0.008 * age_factor, 3)
        self.accel_noise = 0.08 * age_factor
        self.gyro_noise = 0.015 * age_factor
        self.sta_frequency = 0.25
        self.sta_amplitude = 0.05 * (height / 1.75)
        self.sta_phase = np.random.uniform(0, 2 * np.pi)
        self.ground_truth_buffer = []
        self.GYRO_SAT = np.deg2rad(2000.0)
        self._lp_alpha = 0.76
        self._lp_accel = np.array([0.0, 0.0, 9.81])
        self._lp_gyro = np.zeros(3)
        self._aa_alpha = 0.87
        self._aa_accel = np.array([0.0, 0.0, 9.81])
        self._aa_gyro = np.zeros(3)
        self._prev_v_world = None

    def _get_rot_matrix(self):
        body_quat = self.mj_data.xquat[self.body_id]
        rot_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rot_matrix, body_quat)
        return rot_matrix.reshape(3, 3)

    def _infer_l1l2_proxy_offset(self):
        if self.body_name.lower() == 'torso' and self.pelvis_id >= 0:
            R = self._get_rot_matrix()
            torso_pos = self.mj_data.xpos[self.body_id].copy()
            pelvis_pos = self.mj_data.xpos[self.pelvis_id].copy()
            frac = float(IMU_HARDWARE_SPEC.get('proxy_fraction_torso_to_pelvis', 0.4))
            offset_world = frac * (pelvis_pos - torso_pos)
            return R.T @ offset_world
        return np.zeros(3)

    def _sensor_world_position(self):
        R = self._get_rot_matrix()
        origin_world = self.mj_data.xpos[self.body_id].copy()
        return origin_world + R @ self.sensor_offset_local

    def _pelvis_world_velocity(self):
        if self.pelvis_id < 0:
            return np.zeros(3)
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, self.pelvis_id, vel6, 0)
        return vel6[3:].copy()

    def _body_kinematics_local(self):
        vel6 = np.zeros(6)
        acc6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, self.body_id, vel6, 1)
        if hasattr(mujoco, 'mj_objectAcceleration'):
            try:
                mujoco.mj_objectAcceleration(self.mj_model, self.mj_data, MJOBJ_BODY, self.body_id, acc6, 1)
            except Exception:
                acc6[:3] = self.mj_data.cacc[self.body_id][:3]
                acc6[3:] = self.mj_data.cacc[self.body_id][3:]
        else:
            acc6[:3] = self.mj_data.cacc[self.body_id][:3]
            acc6[3:] = self.mj_data.cacc[self.body_id][3:]
        return (vel6, acc6)

    def get_sensor_mount_info(self):
        geom_names = []
        for geom_id in range(self.mj_model.ngeom):
            if self.mj_model.geom_bodyid[geom_id] == self.body_id:
                gname = mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, geom_id) or f'geom_{geom_id}'
                geom_names.append(gname)
        model_pos = self.mj_model.body_pos[self.body_id].copy() if hasattr(self.mj_model, 'body_pos') else np.zeros(3)
        return {'sensor_body': self.body_name, 'sensor_site': self.sensor_site, 'sensor_body_id': int(self.body_id), 'mount_type': f'site:{self.sensor_site}' if self.sensor_site else f'{self.mount_label} proxy', 'body_origin_model_frame_m': model_pos, 'sensor_offset_local_m': self.sensor_offset_local.copy(), 'world_pos_m': self._sensor_world_position().copy(), 'world_quat_wxyz': self.mj_data.xquat[self.body_id].copy(), 'attached_geoms': geom_names}

    def get_sampling_report(self):
        return {'native_dt_s': float(self.native_dt), 'native_hz': float(self.native_hz), 'native_is_100hz': bool(self.native_is_100hz), 'output_dt_s': float(self.output_dt), 'output_hz': float(self.output_hz), 'output_is_100hz': bool(self.output_is_100hz), 'output_mode': self.output_mode, 'effective_bandwidth_hz': float(self.effective_bandwidth_hz), 'true_hardware_equivalent_100hz': bool(self.native_is_100hz)}

    def get_runtime_processing_report(self):
        return {'requested_mount_label': IMU_HARDWARE_SPEC['mount_label'], 'actual_mount_body': self.body_name, 'actual_mount_site': self.sensor_site, 'actual_mount_body_id': int(self.body_id), 'actual_mount_proxy': bool(self.sensor_site is None), 'actual_sensor_offset_local_m': self.sensor_offset_local.copy(), 'actual_sensor_world_pos_m': self._sensor_world_position().copy(), 'actual_kinematics_source': 'mj_objectVelocity + mj_objectAcceleration', 'actual_point_kinematics': 'rigid-body point correction v = v0 + xr, a = a0 + xr + x(xr)', 'actual_native_dt_s': float(self.native_dt), 'actual_native_hz': float(self.native_hz), 'actual_output_dt_s': float(self.output_dt), 'actual_output_hz': float(self.output_hz), 'actual_output_mode': self.output_mode, 'actual_resampler': 'linear interpolation', 'actual_effective_bandwidth_hz': float(self.effective_bandwidth_hz), 'actual_antialias_alpha': float(self._aa_alpha), 'actual_lowpass_alpha': float(self._lp_alpha), 'actual_accel_bias_mps2': self.accel_bias.copy(), 'actual_gyro_bias_rads': self.gyro_bias.copy(), 'actual_accel_noise_std_mps2': float(self.accel_noise), 'actual_gyro_noise_std_rads': float(self.gyro_noise), 'actual_sta_frequency_hz': float(self.sta_frequency), 'actual_sta_amplitude_m': float(self.sta_amplitude), 'actual_accel_saturation_g': 16.0, 'actual_gyro_saturation_dps': 2000.0, 'hardware_equivalent_100hz': bool(self.native_is_100hz)}

    def verify_output_stream(self):
        t_native = np.asarray(self.data_buffer['timestamp'], dtype=float)
        if len(t_native) < 2:
            return {'available': False}
        t_out = np.arange(t_native[0], t_native[-1] + 1e-09, self.output_dt)
        native_dt = np.diff(t_native)
        out_dt = np.diff(t_out)
        res = {'available': True, 'native_uniform_ok': bool(np.max(np.abs(native_dt - self.native_dt)) < 0.0005), 'output_uniform_ok': bool(np.max(np.abs(out_dt - self.output_dt)) < 1e-09) if len(out_dt) else True, 'native_hz': float(self.native_hz), 'output_hz': float(self.output_hz), 'upsample_factor': float(self.output_hz / max(self.native_hz, 1e-09)), 'effective_bandwidth_hz': float(self.effective_bandwidth_hz), 'true_hardware_equivalent_100hz': bool(self.native_is_100hz), 'interpolation': 'linear'}
        self._last_export_verification = res
        return res

    def print_configuration_report(self, age=None, height=None, sex=None):
        mount = self.get_sensor_mount_info()
        rates = self.get_sampling_report()
        runtime = self.get_runtime_processing_report()
        attached = ', '.join(mount['attached_geoms']) if mount['attached_geoms'] else 'none listed'
        print('\n  +---------------------------------------------------------------------+')
        print('  |                    SUBJECT + IMU CONFIGURATION                     |')
        print('  +---------------------------------------------------------------------+')
        print(f'  |  Subject age        : {(age if age is not None else self.age):>8}')
        print(f'  |  Subject height [m] : {(height if height is not None else self.height):>8.3f}')
        print(f"  |  Subject sex        : {(sex if sex is not None else 'n/a'):>8}")
        print('  +---------------------------------------------------------------------+')
        print('  |  REQUESTED HARDWARE TARGET                                         |')
        print('  +---------------------------------------------------------------------+')
        print(f"  |  MCU                : {IMU_HARDWARE_SPEC['microcontroller']}")
        print(f"  |  Accelerometer      : {IMU_HARDWARE_SPEC['accelerometer']}")
        print(f"  |  Gyroscope          : {IMU_HARDWARE_SPEC['gyroscope']}")
        print(f"  |  Requested mount    : {IMU_HARDWARE_SPEC['mount_label']}")
        print(f"  |  Requested rate     : {IMU_HARDWARE_SPEC['sampling_hz']:.2f} Hz")
        print('  +---------------------------------------------------------------------+')
        print('  |  ACTUAL SIMULATION IMPLEMENTATION                                  |')
        print('  +---------------------------------------------------------------------+')
        print(f"  |  Actual mount body  : {runtime['actual_mount_body']} (body id={runtime['actual_mount_body_id']})")
        print(f"  |  Actual mount type  : {mount['mount_type']}")
        print(f"  |  Body origin model  : {np.round(mount['body_origin_model_frame_m'], 5)} m")
        print(f"  |  Sensor offset loc. : {np.round(runtime['actual_sensor_offset_local_m'], 5)} m")
        print(f"  |  Sensor pos world   : {np.round(runtime['actual_sensor_world_pos_m'], 5)} m")
        print(f'  |  Attached geoms     : {attached}')
        print(f"  |  Kinematics source  : {runtime['actual_kinematics_source']}")
        print('  +---------------------------------------------------------------------+')
        print('  |  ACTUAL SIGNAL PROCESSING CHAIN                                    |')
        print('  +---------------------------------------------------------------------+')
        print(f"  |  Native dt / Hz     : {runtime['actual_native_dt_s']:.5f} s / {runtime['actual_native_hz']:.2f} Hz")
        print(f"  |  Output dt / Hz     : {runtime['actual_output_dt_s']:.5f} s / {runtime['actual_output_hz']:.2f} Hz")
        print(f"  |  Output mode        : {runtime['actual_output_mode']}")
        print(f"  |  Resampler          : {runtime['actual_resampler']}")
        print(f"  |  Phys. bandwidth    : ~{runtime['actual_effective_bandwidth_hz']:.2f} Hz max")
        print(f"  |  Anti-alias alpha   : {runtime['actual_antialias_alpha']:.3f}")
        print(f"  |  Low-pass alpha     : {runtime['actual_lowpass_alpha']:.3f}")
        print(f"  |  Accel bias         : {np.round(runtime['actual_accel_bias_mps2'], 5)} m/s^2")
        print(f"  |  Gyro bias          : {np.round(runtime['actual_gyro_bias_rads'], 5)} rad/s")
        print(f"  |  Accel noise std    : {runtime['actual_accel_noise_std_mps2']:.5f} m/s^2")
        print(f"  |  Gyro noise std     : {runtime['actual_gyro_noise_std_rads']:.5f} rad/s")
        print(f"  |  STA freq / amp     : {runtime['actual_sta_frequency_hz']:.3f} Hz / {runtime['actual_sta_amplitude_m']:.5f} m")
        print(f"  |  Accel saturation   : +/-{runtime['actual_accel_saturation_g']:.1f} g")
        print(f"  |  Gyro saturation    : +/-{runtime['actual_gyro_saturation_dps']:.1f} dps")
        print(f"  |  Native == 100 Hz   : {('YES' if rates['native_is_100hz'] else 'NO')}")
        print(f"  |  100 Hz output      : {('YES' if rates['output_is_100hz'] else 'NO')} ({rates['output_mode']})")
        print('  +---------------------------------------------------------------------+')
        print('    [IMU reality] Requested 100 Hz hardware is being approximated by a')
        print('                  30 Hz native simulation stream plus linear resampling.')
        print('                  That is numerically consistent for export, but not')
        print('                  physically identical to a true native 100 Hz IMU.')

    def read_imu(self, sim_time=None, add_artifacts=True, use_cache=True):
        if sim_time is None:
            sim_time = float(self.mj_data.time)
        if use_cache and self._last_read_time is not None and (abs(sim_time - self._last_read_time) < 1e-12):
            cached = {}
            for k, v in self._last_read_sample.items():
                cached[k] = v.copy() if isinstance(v, np.ndarray) else v
            return cached
        body_quat = self.mj_data.xquat[self.body_id].copy()
        R = self._get_rot_matrix()
        gravity_world = np.array([0.0, 0.0, -9.81])
        vel6_local, acc6_local = self._body_kinematics_local()
        omega_local = vel6_local[:3].copy()
        v_origin_local = vel6_local[3:].copy()
        alpha_local = acc6_local[:3].copy()
        a_origin_local = acc6_local[3:].copy()
        r_local = self.sensor_offset_local.copy()
        v_sensor_local = v_origin_local + np.cross(omega_local, r_local)
        a_sensor_local = a_origin_local + np.cross(alpha_local, r_local) + np.cross(omega_local, np.cross(omega_local, r_local))
        v_world = R @ v_sensor_local
        acc_true = a_sensor_local - R.T @ gravity_world
        if self._prev_v_world is not None:
            a_world_fd = (v_world - self._prev_v_world) / self.dt
            acc_raw = R.T @ (a_world_fd - gravity_world)
            acc_norm = float(np.linalg.norm(acc_raw))
            if acc_norm > self.IMU_CLIP_MS2:
                acc_raw = acc_raw * (self.IMU_CLIP_MS2 / acc_norm)
        else:
            acc_raw = np.array([0.0, 0.0, 9.81])
        self._prev_v_world = v_world.copy()
        gyro_true = omega_local.copy()
        self._aa_accel = self._aa_alpha * self._aa_accel + (1 - self._aa_alpha) * acc_true
        self._aa_gyro = self._aa_alpha * self._aa_gyro + (1 - self._aa_alpha) * gyro_true
        acc_noisy = self._aa_accel + self.accel_bias + np.random.normal(0, self.accel_noise, 3)
        gyro_noisy = self._aa_gyro + self.gyro_bias + np.random.normal(0, self.gyro_noise, 3)
        sta = 0.0
        if add_artifacts:
            sta = self.sta_amplitude * np.sin(2 * np.pi * self.sta_frequency * sim_time + self.sta_phase)
            acc_noisy += np.array([0.1, 0.1, 1.0]) * sta * np.random.normal(1, 0.3)
        acc_sat = np.clip(acc_noisy, -self.IMU_CLIP_MS2, self.IMU_CLIP_MS2)
        gyro_sat = np.clip(gyro_noisy, -self.GYRO_SAT, self.GYRO_SAT)
        self._lp_accel = self._lp_alpha * self._lp_accel + (1 - self._lp_alpha) * acc_sat
        self._lp_gyro = self._lp_alpha * self._lp_gyro + (1 - self._lp_alpha) * gyro_sat
        pelvis_height = float(self.mj_data.xpos[self.pelvis_id][2]) if self.pelvis_id >= 0 else float(self._sensor_world_position()[2])
        pelvis_vel_world = self._pelvis_world_velocity()
        contact_force = self._estimate_impact_force()
        confidence = max(0.2, 1.0 - contact_force / 25000.0)
        sample = {'timestamp': sim_time, 'accel': self._lp_accel.copy(), 'accel_raw': acc_raw.copy(), 'gyro': self._lp_gyro.copy(), 'quat': body_quat.copy(), 'height': pelvis_height, 'pelvis_velocity': pelvis_vel_world.copy(), 'sensor_velocity': v_world.copy(), 'sensor_world_pos': self._sensor_world_position().copy(), 'impact': contact_force, 'accel_true': acc_true, 'gyro_true': gyro_true, 'soft_tissue_artifact': sta, 'sensor_confidence': confidence}
        self._last_read_time = float(sim_time)
        self._last_read_sample = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sample.items()}
        return sample

    def _estimate_impact_force(self):
        peak_force = 0.0
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1 = c.geom1
            g2 = c.geom2
            b1 = self.mj_model.geom_bodyid[g1]
            b2 = self.mj_model.geom_bodyid[g2]
            if b1 == self.body_id or b2 == self.body_id:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.mj_model, self.mj_data, i, force)
                peak_force = max(peak_force, float(max(0.0, force[0])))
        return min(peak_force, 12000.0)

    def log_frame(self, sim_time=None):
        if sim_time is None:
            sim_time = float(self.mj_data.time)
        imu_data = self.read_imu(sim_time, use_cache=False)
        self.data_buffer['timestamp'].append(imu_data['timestamp'])
        self.data_buffer['accelerometer'].append(imu_data['accel'])
        self.data_buffer['accel_raw'].append(imu_data['accel_raw'])
        self.data_buffer['gyroscope'].append(imu_data['gyro'])
        self.data_buffer['quaternion'].append(imu_data['quat'])
        self.data_buffer['pelvis_height'].append(imu_data['height'])
        self.data_buffer['pelvis_velocity'].append(imu_data['pelvis_velocity'])
        self.data_buffer['sensor_world_vel'].append(imu_data['sensor_velocity'])
        self.data_buffer['sensor_world_pos'].append(imu_data['sensor_world_pos'])
        self.data_buffer['impact_force'].append(imu_data['impact'])
        self.data_buffer['soft_tissue_artifact'].append(imu_data['soft_tissue_artifact'])
        self.data_buffer['sensor_confidence'].append(imu_data['sensor_confidence'])
        self.ground_truth_buffer.append({'accel_true': imu_data['accel_true'], 'gyro_true': imu_data['gyro_true']})
        return float(np.linalg.norm(imu_data['accel_raw']))

    def _resample_buffers_100hz(self):
        t = np.array(self.data_buffer['timestamp'], dtype=float)
        if len(t) < 2:
            return None
        t_new = np.arange(t[0], t[-1] + 1e-09, self.output_dt)

        def interp_vec(key, dim):
            arr = np.array(self.data_buffer[key], dtype=float)
            out = np.zeros((len(t_new), dim), dtype=float)
            for k in range(dim):
                out[:, k] = np.interp(t_new, t, arr[:, k])
            return out

        def interp_scalar(values):
            arr = np.array(values, dtype=float)
            return np.interp(t_new, t, arr)
        gt_acc = np.array([gt['accel_true'] for gt in self.ground_truth_buffer], dtype=float)
        gt_gyro = np.array([gt['gyro_true'] for gt in self.ground_truth_buffer], dtype=float)
        res = {'timestamp': t_new, 'accelerometer': interp_vec('accelerometer', 3), 'accel_raw': interp_vec('accel_raw', 3), 'gyroscope': interp_vec('gyroscope', 3), 'pelvis_height': interp_scalar(self.data_buffer['pelvis_height']), 'pelvis_velocity': interp_vec('pelvis_velocity', 3), 'sensor_world_pos': interp_vec('sensor_world_pos', 3), 'sensor_world_vel': interp_vec('sensor_world_vel', 3), 'impact_force': interp_scalar(self.data_buffer['impact_force']), 'soft_tissue_artifact': interp_scalar(self.data_buffer['soft_tissue_artifact']), 'sensor_confidence': interp_scalar(self.data_buffer['sensor_confidence']), 'accel_true': np.vstack([np.interp(t_new, t, gt_acc[:, k]) for k in range(3)]).T, 'gyro_true': np.vstack([np.interp(t_new, t, gt_gyro[:, k]) for k in range(3)]).T}
        return res

    def export_to_csv(self, filename, metadata=None):
        import csv
        from datetime import datetime
        if metadata is None:
            metadata = {}
        resampled = self._resample_buffers_100hz()
        if resampled is None:
            return {'filename': filename, 'frames': 0, 'falls_detected': 0}
        verification = self.verify_output_stream()
        n = len(resampled['timestamp'])
        headers = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 'accel_raw_x', 'accel_raw_y', 'accel_raw_z', 'gyro_x', 'gyro_y', 'gyro_z', 'sensor_pos_x', 'sensor_pos_y', 'sensor_pos_z', 'sensor_vel_x', 'sensor_vel_y', 'sensor_vel_z', 'pelvis_height', 'impact_force', 'impact_magnitude', 'jerk_mag', 'fall_detected', 'accel_true_x', 'accel_true_y', 'accel_true_z', 'sensor_error_mag', 'soft_tissue_artifact', 'sensor_confidence']
        mount = self.get_sensor_mount_info()
        rates = self.get_sampling_report()
        meta = {'simulation_date': datetime.now().isoformat(), 'age': metadata.get('age', 35), 'height_m': metadata.get('height', 1.75), 'sex': metadata.get('sex', 'unknown'), 'weight_kg': metadata.get('weight', 70.0), 'fall_type': metadata.get('fall_type', 'backward_walking'), 'microcontroller': IMU_HARDWARE_SPEC['microcontroller'], 'accelerometer': IMU_HARDWARE_SPEC['accelerometer'], 'gyroscope': IMU_HARDWARE_SPEC['gyroscope'], 'target_mount_label': IMU_HARDWARE_SPEC['mount_label'], 'sampling_rate_hz': f'{self.output_hz:.1f}', 'native_sampling_rate_hz': f'{self.native_hz:.3f}', 'native_sampling_is_100hz': rates['native_is_100hz'], 'output_sampling_is_100hz': rates['output_is_100hz'], 'sampling_mode': rates['output_mode'], 'effective_bandwidth_hz': f"{rates['effective_bandwidth_hz']:.3f}", 'resample_verification': verification, 'sensor_body': mount['sensor_body'], 'sensor_body_id': mount['sensor_body_id'], 'sensor_mount_type': mount['mount_type'], 'sensor_body_origin_model_frame_m': np.array2string(np.asarray(mount['body_origin_model_frame_m']), precision=5, separator=', '), 'sensor_offset_local_m': np.array2string(np.asarray(mount['sensor_offset_local_m']), precision=5, separator=', '), 'sensor_world_frame_pos_m': np.array2string(np.asarray(mount['world_pos_m']), precision=5, separator=', '), 'sensor_attached_geoms': ';'.join(mount['attached_geoms']) if mount['attached_geoms'] else 'none', 'total_frames': n, 'note': 'accel=gravity-compensated at lower-back proxy, accel_raw=world-FD reconstructed proper acceleration, 100 Hz stream is linearly resampled if native_hz < 100'}
        raw_arr = np.array(resampled['accel_raw'], dtype=float)
        jerk_mags = [0.0]
        for j in range(1, n):
            da = raw_arr[j] - raw_arr[j - 1]
            jerk_mags.append(float(np.linalg.norm(da) / self.output_dt))
        rows, falls_detected = ([], 0)
        for i in range(n):
            ax, ay, az = [float(resampled['accelerometer'][i][k]) for k in range(3)]
            rx, ry, rz = [float(resampled['accel_raw'][i][k]) for k in range(3)]
            gx, gy, gz = [float(resampled['gyroscope'][i][k]) for k in range(3)]
            spx, spy, spz = [float(resampled['sensor_world_pos'][i][k]) for k in range(3)]
            svx, svy, svz = [float(resampled['sensor_world_vel'][i][k]) for k in range(3)]
            ph = float(resampled['pelvis_height'][i])
            imp = float(resampled['impact_force'][i])
            mag = float(np.sqrt(rx ** 2 + ry ** 2 + rz ** 2))
            jerk = jerk_mags[i]
            fall = int(ph < 0.4 and (jerk > 15.0 or imp > 50.0))
            falls_detected += fall
            ax_t = float(resampled['accel_true'][i][0])
            ay_t = float(resampled['accel_true'][i][1])
            az_t = float(resampled['accel_true'][i][2])
            err_mag = float(np.sqrt((ax - ax_t) ** 2 + (ay - ay_t) ** 2 + (az - az_t) ** 2))
            sta = float(resampled['soft_tissue_artifact'][i])
            conf = float(resampled['sensor_confidence'][i])
            rows.append([round(resampled['timestamp'][i], 4), round(ax, 6), round(ay, 6), round(az, 6), round(rx, 6), round(ry, 6), round(rz, 6), round(gx, 6), round(gy, 6), round(gz, 6), round(spx, 6), round(spy, 6), round(spz, 6), round(svx, 6), round(svy, 6), round(svz, 6), round(ph, 6), round(imp, 4), round(mag, 6), round(jerk, 4), fall, round(ax_t, 6), round(ay_t, 6), round(az_t, 6), round(err_mag, 6), round(sta, 6), round(conf, 3)])
        meta['falls_detected_frames'] = falls_detected
        with open(filename, 'w', newline='') as f:
            for key, val in meta.items():
                f.write(f'# {key}: {val}\n')
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f'  IMU data saved -> {filename}  ({n} frames | {falls_detected} fall-detected)')
        return {'filename': filename, 'frames': n, 'falls_detected': falls_detected}

def rotmat_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion in MuJoCo/OpenSim-friendly wxyz order."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    q = np.empty(4, dtype=float)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12))
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12))
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12))
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    q /= max(np.linalg.norm(q), 1e-12)
    if q[0] < 0.0:
        q *= -1.0
    return q

class MarkerKinematicsExporter:
    """
    Layer 1 biomechanics export for motion-analysis style validation.

    This version is more paper-aligned than the first pass:
      - broader anatomical marker set
      - much stronger fuzzy body resolution for humenv naming differences
      - virtual-marker trajectories + segment kinematics + joint kinematics
      - TRC export for marker-based motion-analysis tooling
      - MOT export for joint-angle style inspection
      - PNG visual summaries for quick qualitative review

    Notes:
      * The paper mentions site/marker-based biomechanical analysis. In the
        current compiled humenv workflow, the safest retrofit is to attach
        virtual anatomical markers rigidly to existing bodies.
      * These markers are dynamically correct with respect to the simulated
        body, even though they are not yet injected as MJCF <site> objects.
    """
    DEFAULT_MARKERS = {'HEAD': {'candidates': ['Head'], 'offset_mode': 'superior', 'family': 'head'}, 'C7': {'candidates': ['Torso', 'Chest', 'Spine'], 'offset_mode': 'posterior_superior', 'family': 'torso'}, 'CLAV': {'candidates': ['Torso', 'Chest', 'Spine'], 'offset_mode': 'anterior_superior', 'family': 'torso'}, 'STRN': {'candidates': ['Torso', 'Chest', 'Spine'], 'offset_mode': 'anterior_center', 'family': 'torso'}, 'T10': {'candidates': ['Torso', 'Chest', 'Spine'], 'offset_mode': 'posterior_center', 'family': 'torso'}, 'SACR': {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'posterior_center', 'family': 'pelvis'}, 'SHO_L': {'candidates': ['L_Shoulder', 'LeftShoulder', 'Shoulder_L', 'UpperArmL', 'ShoulderL', 'ArmL'], 'offset_mode': 'proximal', 'family': 'upperarm', 'side': 'left'}, 'SHO_R': {'candidates': ['R_Shoulder', 'RightShoulder', 'Shoulder_R', 'UpperArmR', 'ShoulderR', 'ArmR'], 'offset_mode': 'proximal', 'family': 'upperarm', 'side': 'right'}, 'ELB_L': {'candidates': ['L_Elbow', 'LeftElbow', 'Elbow_L', 'LowerArmL', 'ForeArmL', 'ElbowL'], 'offset_mode': 'proximal', 'family': 'lowerarm', 'side': 'left'}, 'ELB_R': {'candidates': ['R_Elbow', 'RightElbow', 'Elbow_R', 'LowerArmR', 'ForeArmR', 'ElbowR'], 'offset_mode': 'proximal', 'family': 'lowerarm', 'side': 'right'}, 'WRI_L': {'candidates': ['L_Wrist', 'L_Hand', 'LeftWrist', 'Wrist_L', 'HandL', 'WristL'], 'offset_mode': 'proximal', 'family': 'hand', 'side': 'left'}, 'WRI_R': {'candidates': ['R_Hand', 'R_Wrist', 'RightWrist', 'Wrist_R', 'HandR', 'WristR'], 'offset_mode': 'proximal', 'family': 'hand', 'side': 'right'}, 'ASI_L': {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'left_anterior', 'family': 'pelvis', 'side': 'left'}, 'ASI_R': {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'right_anterior', 'family': 'pelvis', 'side': 'right'}, 'PSI_L': {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'left_posterior', 'family': 'pelvis', 'side': 'left'}, 'PSI_R': {'candidates': ['Pelvis', 'Hips'], 'offset_mode': 'right_posterior', 'family': 'pelvis', 'side': 'right'}, 'HIP_L': {'candidates': ['L_Hip', 'LeftHip', 'Hip_L', 'ThighL', 'UpperLegL', 'HipL'], 'offset_mode': 'proximal', 'family': 'thigh', 'side': 'left'}, 'HIP_R': {'candidates': ['R_Hip', 'RightHip', 'Hip_R', 'ThighR', 'UpperLegR', 'HipR'], 'offset_mode': 'proximal', 'family': 'thigh', 'side': 'right'}, 'KNE_L': {'candidates': ['L_Knee', 'LeftKnee', 'Knee_L', 'L_Shin', 'ShinL', 'LowerLegL', 'KneeL'], 'offset_mode': 'proximal', 'family': 'shank', 'side': 'left'}, 'KNE_R': {'candidates': ['R_Knee', 'RightKnee', 'Knee_R', 'R_Shin', 'ShinR', 'LowerLegR', 'KneeR'], 'offset_mode': 'proximal', 'family': 'shank', 'side': 'right'}, 'ANK_L': {'candidates': ['L_Ankle', 'LeftAnkle', 'Ankle_L', 'L_Foot', 'FootL', 'AnkleL'], 'offset_mode': 'proximal', 'family': 'foot', 'side': 'left'}, 'ANK_R': {'candidates': ['R_Ankle', 'RightAnkle', 'Ankle_R', 'R_Foot', 'FootR', 'AnkleR'], 'offset_mode': 'proximal', 'family': 'foot', 'side': 'right'}, 'HEE_L': {'candidates': ['L_Heel', 'LeftHeel', 'Heel_L', 'L_Ankle', 'L_Foot', 'FootL', 'HeelL'], 'offset_mode': 'posterior_inferior', 'family': 'foot', 'side': 'left'}, 'HEE_R': {'candidates': ['R_Heel', 'RightHeel', 'Heel_R', 'R_Ankle', 'R_Foot', 'FootR', 'HeelR'], 'offset_mode': 'posterior_inferior', 'family': 'foot', 'side': 'right'}, 'TOE_L': {'candidates': ['L_Toe', 'LeftToe', 'Toe_L', 'L_Foot', 'FootL', 'ToeL'], 'offset_mode': 'anterior_inferior', 'family': 'foot', 'side': 'left'}, 'TOE_R': {'candidates': ['R_Toe', 'RightToe', 'Toe_R', 'R_Foot', 'FootR', 'ToeR'], 'offset_mode': 'anterior_inferior', 'family': 'foot', 'side': 'right'}}
    SEGMENT_EXPORTS = {'pelvis': ['Pelvis', 'Hips'], 'torso': ['Torso', 'Chest', 'Spine'], 'head': ['Head'], 'left_upperarm': ['L_Shoulder', 'LeftShoulder', 'Shoulder_L', 'UpperArmL', 'ShoulderL', 'ArmL'], 'right_upperarm': ['R_Shoulder', 'RightShoulder', 'Shoulder_R', 'UpperArmR', 'ShoulderR', 'ArmR'], 'left_forearm': ['L_Elbow', 'LeftElbow', 'Elbow_L', 'LowerArmL', 'ForeArmL', 'ElbowL'], 'right_forearm': ['R_Elbow', 'RightElbow', 'Elbow_R', 'LowerArmR', 'ForeArmR', 'ElbowR'], 'left_hand': ['L_Hand', 'L_Wrist', 'LeftWrist', 'HandL', 'WristL'], 'right_hand': ['R_Hand', 'R_Wrist', 'RightWrist', 'HandR', 'WristR'], 'left_thigh': ['L_Hip', 'LeftHip', 'Hip_L', 'ThighL', 'UpperLegL', 'HipL'], 'right_thigh': ['R_Hip', 'RightHip', 'Hip_R', 'ThighR', 'UpperLegR', 'HipR'], 'left_shank': ['L_Knee', 'LeftKnee', 'Knee_L', 'L_Shin', 'ShinL', 'LowerLegL', 'KneeL'], 'right_shank': ['R_Knee', 'RightKnee', 'Knee_R', 'R_Shin', 'ShinR', 'LowerLegR', 'KneeR'], 'left_foot': ['L_Foot', 'L_Ankle', 'L_Toe', 'L_Heel', 'FootL', 'ToeL', 'HeelL'], 'right_foot': ['R_Foot', 'R_Ankle', 'R_Toe', 'R_Heel', 'FootR', 'ToeR', 'HeelR']}
    SKELETON_EDGES = [('HEAD', 'C7'), ('C7', 'CLAV'), ('CLAV', 'STRN'), ('C7', 'T10'), ('T10', 'SACR'), ('CLAV', 'SHO_L'), ('SHO_L', 'ELB_L'), ('ELB_L', 'WRI_L'), ('CLAV', 'SHO_R'), ('SHO_R', 'ELB_R'), ('ELB_R', 'WRI_R'), ('SACR', 'ASI_L'), ('ASI_L', 'HIP_L'), ('HIP_L', 'KNE_L'), ('KNE_L', 'ANK_L'), ('ANK_L', 'HEE_L'), ('ANK_L', 'TOE_L'), ('SACR', 'ASI_R'), ('ASI_R', 'HIP_R'), ('HIP_R', 'KNE_R'), ('KNE_R', 'ANK_R'), ('ANK_R', 'HEE_R'), ('ANK_R', 'TOE_R'), ('ASI_L', 'ASI_R'), ('PSI_L', 'PSI_R')]
    POSE_MARKERS = ['HEAD', 'C7', 'SHO_L', 'SHO_R', 'ELB_L', 'ELB_R', 'WRI_L', 'WRI_R', 'SACR', 'HIP_L', 'HIP_R', 'KNE_L', 'KNE_R', 'ANK_L', 'ANK_R', 'HEE_L', 'HEE_R', 'TOE_L', 'TOE_R']
    OPENSIM_TRC_NAMES = {'HEAD': 'HEAD', 'C7': 'C7', 'CLAV': 'CLAV', 'STRN': 'STRN', 'T10': 'T10', 'SACR': 'SACR', 'ASI_L': 'LASI', 'ASI_R': 'RASI', 'PSI_L': 'LPSI', 'PSI_R': 'RPSI', 'SHO_L': 'LSHO', 'SHO_R': 'RSHO', 'ELB_L': 'LELB', 'ELB_R': 'RELB', 'WRI_L': 'LWRA', 'WRI_R': 'RWRA', 'HIP_L': 'LHIP', 'HIP_R': 'RHIP', 'KNE_L': 'LKNE', 'KNE_R': 'RKNE', 'ANK_L': 'LANK', 'ANK_R': 'RANK', 'HEE_L': 'LHEE', 'HEE_R': 'RHEE', 'TOE_L': 'LTOE', 'TOE_R': 'RTOE'}
    VISUAL_MARKERS = ['HEAD', 'C7', 'SHO_L', 'SHO_R', 'ELB_L', 'ELB_R', 'WRI_L', 'WRI_R', 'SACR', 'HIP_L', 'HIP_R', 'KNE_L', 'KNE_R', 'ANK_L', 'ANK_R', 'HEE_L', 'HEE_R', 'TOE_L', 'TOE_R']
    VISUAL_SKELETON_EDGES = [('HEAD', 'C7'), ('C7', 'SACR'), ('C7', 'SHO_L'), ('SHO_L', 'ELB_L'), ('ELB_L', 'WRI_L'), ('C7', 'SHO_R'), ('SHO_R', 'ELB_R'), ('ELB_R', 'WRI_R'), ('SHO_L', 'SHO_R'), ('SACR', 'HIP_L'), ('HIP_L', 'KNE_L'), ('KNE_L', 'ANK_L'), ('ANK_L', 'HEE_L'), ('ANK_L', 'TOE_L'), ('SACR', 'HIP_R'), ('HIP_R', 'KNE_R'), ('KNE_R', 'ANK_R'), ('ANK_R', 'HEE_R'), ('ANK_R', 'TOE_R'), ('HIP_L', 'HIP_R')]
    _SIDE_HINTS = {'left': ['left', '_l', 'l_', '-l', 'l-', 'lft', 'lf', 'lh', 'arml', 'handl', 'footl', 'thighl', 'shinl', 'toel', 'heell', 'wl', 'el', 'sl', 'hipl', 'kneel', 'anklel'], 'right': ['right', '_r', 'r_', '-r', 'r-', 'rgt', 'rt', 'rh', 'armr', 'handr', 'footr', 'thighr', 'shinr', 'toer', 'heelr', 'wr', 'er', 'sr', 'hipr', 'kneer', 'ankler']}
    _FAMILY_HINTS = {'head': ['head', 'skull', 'neck'], 'torso': ['torso', 'chest', 'spine', 'thorax', 'trunk', 'abdomen'], 'pelvis': ['pelvis', 'hip', 'hips', 'waist'], 'upperarm': ['upperarm', 'shoulder', 'arm', 'humerus'], 'lowerarm': ['lowerarm', 'forearm', 'elbow', 'ulna', 'radius'], 'hand': ['hand', 'wrist', 'palm'], 'thigh': ['thigh', 'upperleg', 'hip', 'femur'], 'shank': ['shin', 'lowerleg', 'knee', 'calf', 'tibia'], 'foot': ['foot', 'toe', 'heel', 'ankle']}

    def __init__(self, mj_model, mj_data, export_hz=None):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.export_hz = float(export_hz) if export_hz is not None else 100.0
        self.frames = []
        self._joint_exports = self._resolve_joint_exports()
        self._body_catalog = self._build_body_catalog()
        self.marker_defs = self._build_marker_definitions()
        self.segment_ids = self._resolve_segment_ids()

    def _normalize(self, s):
        return ''.join((ch.lower() for ch in str(s or '') if ch.isalnum()))

    def _body_name(self, body_id):
        return mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, body_id) or f'body_{body_id}'

    def _joint_name(self, joint_id):
        if hasattr(mujoco, 'mjtObj'):
            obj_joint = mujoco.mjtObj.mjOBJ_JOINT
        else:
            obj_joint = mujoco.mjOBJ_JOINT
        return mujoco.mj_id2name(self.mj_model, obj_joint, joint_id) or f'joint_{joint_id}'

    def _build_body_catalog(self):
        catalog = []
        for i in range(self.mj_model.nbody):
            name = self._body_name(i)
            norm = self._normalize(name)
            catalog.append({'id': i, 'name': name, 'norm': norm, 'tokens': self._tokenize(name)})
        return catalog

    def _tokenize(self, s):
        s = str(s or '').replace('-', '_')
        return [tok.lower() for tok in re.split('[^A-Za-z0-9]+', s) if tok]

    def _side_matches(self, rec, side):
        if not side:
            return True
        norm_name = rec['norm']
        tokens = rec.get('tokens', [])
        left_hit = any((t in ('l', 'left') for t in tokens)) or norm_name.endswith('l') or norm_name.startswith('l') or ('_l' in rec['name'].lower())
        right_hit = any((t in ('r', 'right') for t in tokens)) or norm_name.endswith('r') or norm_name.startswith('r') or ('_r' in rec['name'].lower())
        if side == 'left':
            return left_hit and (not right_hit)
        if side == 'right':
            return right_hit and (not left_hit)
        return True

    def _family_matches(self, rec, family):
        if not family:
            return True
        fam_score = self._score_family(rec['norm'], family)
        return fam_score > 0.0

    def _score_side(self, norm_name, side):
        if not side:
            return 0.0
        hints = self._SIDE_HINTS.get(side, [])
        score = 0.0
        for h in hints:
            if self._normalize(h) in norm_name:
                score += 1.0
        if side == 'left' and norm_name.endswith('l'):
            score += 0.75
        if side == 'right' and norm_name.endswith('r'):
            score += 0.75
        return score

    def _score_family(self, norm_name, family):
        if not family:
            return 0.0
        hints = self._FAMILY_HINTS.get(family, [])
        score = 0.0
        for h in hints:
            hnorm = self._normalize(h)
            if hnorm and hnorm in norm_name:
                score += 1.0
        if family == 'foot' and any((norm_name.endswith(s) for s in ('foot', 'toe', 'heel', 'ankle'))):
            score += 0.75
        elif family == 'shank' and any((k in norm_name for k in ('knee', 'shin', 'lowerleg', 'calf', 'tibia'))):
            score += 0.75
        elif family == 'thigh' and any((k in norm_name for k in ('thigh', 'upperleg', 'femur'))):
            score += 0.75
        elif family == 'upperarm' and any((k in norm_name for k in ('shoulder', 'upperarm', 'humerus'))):
            score += 0.75
        elif family == 'lowerarm' and any((k in norm_name for k in ('elbow', 'forearm', 'lowerarm', 'ulna', 'radius'))):
            score += 0.75
        elif family == 'hand' and any((k in norm_name for k in ('hand', 'wrist', 'palm'))):
            score += 0.75
        elif family == 'pelvis' and any((k in norm_name for k in ('pelvis', 'hips', 'waist'))):
            score += 0.75
        elif family == 'torso' and any((k in norm_name for k in ('torso', 'chest', 'spine', 'thorax', 'trunk', 'abdomen'))):
            score += 0.75
        elif family == 'head' and any((k in norm_name for k in ('head', 'skull', 'neck'))):
            score += 0.75
        return score

    def _candidate_score(self, rec, key, family=None, side=None):
        score = 0.0
        if rec['norm'] == key:
            score += 100.0
        if rec['norm'].endswith(key):
            score += 30.0
        if key in rec['norm']:
            score += 20.0
        score += 5.0 * self._score_family(rec['norm'], family)
        score += 7.0 * self._score_side(rec['norm'], side)
        if side and (not self._side_matches(rec, side)):
            score -= 50.0
        if family and (not self._family_matches(rec, family)):
            score -= 20.0
        return score

    def _find_body_id(self, candidates, family=None, side=None):
        best_id = -1
        best_score = -1000000000.0
        cand_keys = [self._normalize(c) for c in candidates if self._normalize(c)]
        for rec in self._body_catalog:
            for key in cand_keys:
                score = self._candidate_score(rec, key, family=family, side=side)
                if score > best_score:
                    best_score = score
                    best_id = rec['id']
        if best_id >= 0 and best_score >= 20.0:
            return best_id
        best_id = -1
        best_score = -1000000000.0
        for rec in self._body_catalog:
            score = 5.0 * self._score_family(rec['norm'], family) + 7.0 * self._score_side(rec['norm'], side)
            if side and (not self._side_matches(rec, side)):
                score -= 30.0
            if family and (not self._family_matches(rec, family)):
                score -= 10.0
            if family == 'pelvis' and rec['id'] == 0:
                score -= 0.5
            if score > best_score and score >= 5.0:
                best_score = score
                best_id = rec['id']
        return best_id

    def _body_geom_ids(self, body_id):
        return [gid for gid in range(self.mj_model.ngeom) if self.mj_model.geom_bodyid[gid] == body_id]

    def _body_local_extents(self, body_id):
        mins = []
        maxs = []
        geom_ids = self._body_geom_ids(body_id)
        if not geom_ids:
            return (np.array([-0.03, -0.03, -0.03]), np.array([0.03, 0.03, 0.03]))
        for gid in geom_ids:
            size = np.asarray(self.mj_model.geom_size[gid], dtype=float)
            pos = np.asarray(self.mj_model.geom_pos[gid], dtype=float)
            gtype = int(self.mj_model.geom_type[gid])
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                half = np.array([size[0], size[0], size[0]])
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, h = (size[0], size[1])
                half = np.array([r, r, h + r])
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r, h = (size[0], size[1])
                half = np.array([r, r, h])
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                half = size[:3]
            else:
                half = np.array([0.03, 0.03, 0.03])
            mins.append(pos - half)
            maxs.append(pos + half)
        return (np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0))

    def _offset_from_mode(self, body_id, mode):
        mins, maxs = self._body_local_extents(body_id)
        center = 0.5 * (mins + maxs)
        eps = 0.008
        left_y = maxs[1] - eps
        right_y = mins[1] + eps
        mapping = {'center': center, 'superior': np.array([center[0], center[1], maxs[2] - eps]), 'inferior': np.array([center[0], center[1], mins[2] + eps]), 'anterior_center': np.array([maxs[0] - eps, center[1], center[2]]), 'anterior_superior': np.array([maxs[0] - eps, center[1], maxs[2] - eps]), 'posterior_superior': np.array([mins[0] + eps, center[1], maxs[2] - eps]), 'posterior_center': np.array([mins[0] + eps, center[1], center[2]]), 'posterior_inferior': np.array([mins[0] + eps, center[1], mins[2] + eps]), 'anterior_inferior': np.array([maxs[0] - eps, center[1], mins[2] + eps]), 'left_anterior': np.array([maxs[0] - eps, left_y, center[2]]), 'right_anterior': np.array([maxs[0] - eps, right_y, center[2]]), 'left_posterior': np.array([mins[0] + eps, left_y, center[2]]), 'right_posterior': np.array([mins[0] + eps, right_y, center[2]]), 'proximal': np.array([mins[0] + eps, center[1], center[2]]), 'distal': np.array([maxs[0] - eps, center[1], center[2]])}
        return mapping.get(mode, center)

    def _build_marker_definitions(self):
        defs = {}
        for mname, spec in self.DEFAULT_MARKERS.items():
            body_id = self._find_body_id(spec['candidates'], family=spec.get('family'), side=spec.get('side'))
            if body_id < 0:
                continue
            defs[mname] = {'body_id': body_id, 'body_name': self._body_name(body_id), 'offset_local': self._offset_from_mode(body_id, spec.get('offset_mode', 'center')), 'offset_mode': spec.get('offset_mode', 'center'), 'family': spec.get('family'), 'side': spec.get('side')}
        return defs

    def _resolve_segment_ids(self):
        out = {}
        for label, candidates in self.SEGMENT_EXPORTS.items():
            family = 'pelvis' if 'pelvis' in label else 'torso' if 'torso' in label else 'head' if 'head' in label else 'upperarm' if 'upperarm' in label else 'lowerarm' if 'forearm' in label else 'hand' if 'hand' in label else 'thigh' if 'thigh' in label else 'shank' if 'shank' in label else 'foot'
            side = 'left' if label.startswith('left_') else 'right' if label.startswith('right_') else None
            bid = self._find_body_id(candidates, family=family, side=side)
            if bid >= 0:
                out[label] = bid
        return out

    def _resolve_joint_exports(self):
        exports = []
        for jid in range(self.mj_model.njnt):
            qadr = int(self.mj_model.jnt_qposadr[jid])
            dadr = int(self.mj_model.jnt_dofadr[jid]) if jid < len(self.mj_model.jnt_dofadr) else -1
            jtype = int(self.mj_model.jnt_type[jid])
            if jtype == getattr(mujoco.mjtJoint, 'mjJNT_FREE', -999):
                continue
            exports.append({'joint_id': jid, 'joint_name': self._joint_name(jid), 'qpos_adr': qadr, 'dof_adr': dadr, 'joint_type': jtype})
        return exports

    def _body_rot(self, body_id):
        return self.mj_data.xmat[body_id].reshape(3, 3).copy()

    def _marker_world(self, body_id, offset_local):
        R = self._body_rot(body_id)
        return self.mj_data.xpos[body_id].copy() + R @ np.asarray(offset_local, dtype=float)

    def _point_world_velocity(self, body_id, point_world):
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, body_id, vel6, 0)
        omega = vel6[:3].copy()
        v0 = vel6[3:].copy()
        r = np.asarray(point_world, dtype=float) - self.mj_data.xpos[body_id].copy()
        return v0 + np.cross(omega, r)

    def _joint_angle_deg(self, rec):
        qadr = rec['qpos_adr']
        jtype = rec['joint_type']
        if jtype == getattr(mujoco.mjtJoint, 'mjJNT_HINGE', -1):
            return float(np.degrees(self.mj_data.qpos[qadr]))
        if jtype == getattr(mujoco.mjtJoint, 'mjJNT_SLIDE', -3):
            return float(self.mj_data.qpos[qadr])
        if jtype == getattr(mujoco.mjtJoint, 'mjJNT_BALL', -2):
            quat = np.asarray(self.mj_data.qpos[qadr:qadr + 4], dtype=float)
            if quat.shape[0] == 4:
                return float(np.degrees(2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))))
        return float(self.mj_data.qpos[qadr])

    def _compute_equal_limits(self, pts_list, dims=(0, 1), pad=0.03):
        arr = np.array([p for p in pts_list if p is not None], dtype=float)
        if arr.size == 0:
            return None
        sub = arr[:, list(dims)]
        mins = np.min(sub, axis=0)
        maxs = np.max(sub, axis=0)
        center = 0.5 * (mins + maxs)
        half = 0.5 * np.max(maxs - mins) + pad
        return [(center[i] - half, center[i] + half) for i in range(2)]

    def _compute_equal_limits_3d(self, pts_list, pad=0.03):
        arr = np.array([p for p in pts_list if p is not None], dtype=float)
        if arr.size == 0:
            return None
        mins = np.min(arr, axis=0)
        maxs = np.max(arr, axis=0)
        center = 0.5 * (mins + maxs)
        half = 0.5 * np.max(maxs - mins) + pad
        return [(center[i] - half, center[i] + half) for i in range(3)]

    def _collect_plot_points(self, pts, names=None):
        names = names or [n for n in self.VISUAL_MARKERS if n in pts]
        return [np.asarray(pts[n], dtype=float) for n in names if n in pts]

    def _presentation_frame(self, pts):
        pts = {k: np.asarray(v, dtype=float) for k, v in pts.items() if v is not None}
        if 'SACR' in pts:
            origin = pts['SACR'].copy()
        elif 'HIP_L' in pts and 'HIP_R' in pts:
            origin = 0.5 * (pts['HIP_L'] + pts['HIP_R'])
        elif pts:
            origin = np.mean(np.stack(list(pts.values()), axis=0), axis=0)
        else:
            origin = np.zeros(3, dtype=float)
        lateral = None
        for a, b in [('HIP_L', 'HIP_R'), ('ASI_L', 'ASI_R'), ('SHO_L', 'SHO_R')]:
            if a in pts and b in pts:
                lateral = pts[b] - pts[a]
                break
        if lateral is None:
            lateral = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral = np.asarray(lateral, dtype=float)
        lateral[2] = 0.0
        lat_n = float(np.linalg.norm(lateral))
        if lat_n < 1e-09:
            lateral = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            lateral /= lat_n
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        forward = np.cross(up, lateral)
        fwd_n = float(np.linalg.norm(forward))
        if fwd_n < 1e-09:
            forward = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            forward /= fwd_n
        ref = None
        if 'C7' in pts:
            ref = pts['C7'] - origin
        elif 'HEAD' in pts:
            ref = pts['HEAD'] - origin
        elif 'SHO_L' in pts and 'SHO_R' in pts:
            ref = 0.5 * (pts['SHO_L'] + pts['SHO_R']) - origin
        if ref is not None:
            ref = np.asarray(ref, dtype=float).copy()
            ref[2] = 0.0
            if np.linalg.norm(ref) > 1e-09 and np.dot(forward, ref) < 0.0:
                forward *= -1.0
        lateral = np.cross(up, forward)
        lat_n = float(np.linalg.norm(lateral))
        if lat_n < 1e-09:
            lateral = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            lateral /= lat_n
        return (origin, forward, lateral, up)

    def _point_to_presentation_local(self, point, origin, forward, lateral, up):
        diff = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
        return np.array([float(np.dot(diff, forward)), float(np.dot(diff, lateral)), float(np.dot(diff, up))], dtype=float)

    def _points_to_presentation_local(self, pts):
        origin, forward, lateral, up = self._presentation_frame(pts)
        return {n: self._point_to_presentation_local(p, origin, forward, lateral, up) for n, p in pts.items()}

    def _marker_side(self, name):
        if str(name).endswith('_L'):
            return 'left'
        if str(name).endswith('_R'):
            return 'right'
        return 'center'

    def _edge_style(self, a, b):
        sa = self._marker_side(a)
        sb = self._marker_side(b)
        if sa == sb == 'left':
            return {'color': '#2ca02c', 'linewidth': 2.2}
        if sa == sb == 'right':
            return {'color': '#d62728', 'linewidth': 2.2}
        if sa == sb == 'center':
            return {'color': '#1f77b4', 'linewidth': 2.4}
        return {'color': '#7f7f7f', 'linewidth': 1.9}

    def _phase_indices(self, *phase_names):
        wanted = {str(p).lower() for p in phase_names}
        return [i for i, fr in enumerate(self.frames) if str(fr.get('phase', '')).lower() in wanted]

    def _mid_index(self, idxs, fallback=0):
        if not idxs:
            return int(fallback)
        return int(idxs[len(idxs) // 2])

    def _select_peak_pose_index(self):
        if not self.frames:
            return 0
        pelvis_h = np.array([fr['pelvis_height'] for fr in self.frames], dtype=float)
        trunk = np.array([fr['trunk_lean_deg'] for fr in self.frames], dtype=float)
        active = self._phase_indices('perturb', 'react', 'fall')
        if not active:
            return int(np.argmax(trunk))
        if len(active) >= 2:
            local_h = pelvis_h[active]
            drop_i = int(np.argmin(np.diff(local_h))) + 1
            anchor = active[min(drop_i, len(active) - 1)]
        else:
            anchor = active[0]
        lo = max(active[0], anchor - 3)
        hi = min(len(self.frames) - 1, anchor + 8)
        min_h = float(np.min(pelvis_h))
        cands = [i for i in range(lo, hi + 1) if pelvis_h[i] > min_h + 0.04]
        if not cands:
            cands = [i for i in active if pelvis_h[i] > min_h + 0.02] or list(active)

        def score(i):
            drop = max(0.0, pelvis_h[0] - pelvis_h[i])
            return 0.75 * trunk[i] + 35.0 * drop
        return int(max(cands, key=score))

    def _select_snapshot_indices(self):
        n = len(self.frames)
        if n == 0:
            return []
        stand_idx = self._mid_index(self._phase_indices('stand'), 0)
        walk_idx = self._mid_index(self._phase_indices('walk'), min(n - 1, max(1, n // 3)))
        pose_idx = self._select_peak_pose_index()
        rest_idx = n - 1
        idxs = []
        for idx in [stand_idx, walk_idx, pose_idx, rest_idx]:
            idx = int(np.clip(idx, 0, n - 1))
            if idx not in idxs:
                idxs.append(idx)
        while len(idxs) < 4 and len(idxs) < n:
            cand = int(round(len(idxs) * (n - 1) / max(3, n - 1)))
            if cand not in idxs:
                idxs.append(cand)
            else:
                break
        return idxs[:4]

    def _snapshot_label(self, idx):
        if not self.frames:
            return ''
        phase = str(self.frames[idx].get('phase', '')).upper() or 'FRAME'
        if idx == self._select_peak_pose_index():
            phase = 'LOSS OF BAL.'
        elif idx == len(self.frames) - 1:
            phase = 'REST'
        return f"{phase}\nt={self.frames[idx]['time']:.2f}s"

    def _quality_summary_lines(self):
        qc = self.quality_report()
        suspect = [k for k, v in qc.items() if v.get('status') != 'ok']
        derived_segments = [k for k, v in self.marker_summary().get('segments', {}).items() if v == 'derived_marker_triad']
        lines = ['Layer-1 export summary', f'Markers resolved : {len(self.marker_defs)}/{len(self.DEFAULT_MARKERS)}', f'Suspect markers : {len(suspect)}', f"Derived segments : {len(derived_segments)} ({(', '.join(derived_segments) if derived_segments else 'none')})", f'Joint channels   : {len(self._joint_exports)}', f'Frames captured  : {len(self.frames)}']
        if suspect:
            lines.append('Suspects        : ' + ', '.join(suspect[:6]) + ('...' if len(suspect) > 6 else ''))
        return lines

    def _derived_segment_record(self, label, markers, marker_velocities):

        def mk(name):
            return np.asarray(markers[name], dtype=float) if name in markers else None

        def mv(name):
            return np.asarray(marker_velocities[name], dtype=float) if name in marker_velocities else np.zeros(3)
        if label == 'left_foot' and all((k in markers for k in ('ANK_L', 'HEE_L', 'TOE_L'))):
            a, h, t = (mk('ANK_L'), mk('HEE_L'), mk('TOE_L'))
            x = t - h
            x = x / max(np.linalg.norm(x), 1e-09)
            z = a - 0.5 * (h + t)
            z = z / max(np.linalg.norm(z), 1e-09)
            y = np.cross(z, x)
            y = y / max(np.linalg.norm(y), 1e-09)
            z = np.cross(x, y)
            z = z / max(np.linalg.norm(z), 1e-09)
            R = np.column_stack([x, y, z])
            return {'body_name': 'derived_left_foot', 'position': (a + h + t) / 3.0, 'orientation_quat': rotmat_to_quat(R), 'angular_velocity': np.zeros(3), 'linear_velocity': (mv('ANK_L') + mv('HEE_L') + mv('TOE_L')) / 3.0, 'source': 'derived_markers'}
        if label == 'right_foot' and all((k in markers for k in ('ANK_R', 'HEE_R', 'TOE_R'))):
            a, h, t = (mk('ANK_R'), mk('HEE_R'), mk('TOE_R'))
            x = t - h
            x = x / max(np.linalg.norm(x), 1e-09)
            z = a - 0.5 * (h + t)
            z = z / max(np.linalg.norm(z), 1e-09)
            y = np.cross(z, x)
            y = y / max(np.linalg.norm(y), 1e-09)
            z = np.cross(x, y)
            z = z / max(np.linalg.norm(z), 1e-09)
            R = np.column_stack([x, y, z])
            return {'body_name': 'derived_right_foot', 'position': (a + h + t) / 3.0, 'orientation_quat': rotmat_to_quat(R), 'angular_velocity': np.zeros(3), 'linear_velocity': (mv('ANK_R') + mv('HEE_R') + mv('TOE_R')) / 3.0, 'source': 'derived_markers'}
        return None

    def quality_report(self):
        report = {}
        for mname, spec in self.DEFAULT_MARKERS.items():
            if mname not in self.marker_defs:
                report[mname] = {'status': 'missing'}
                continue
            rec = self.marker_defs[mname]
            norm = self._normalize(rec['body_name'])
            side_ok = self._side_matches({'name': rec['body_name'], 'norm': norm, 'tokens': self._tokenize(rec['body_name'])}, spec.get('side')) if spec.get('side') else True
            fam_ok = self._family_matches({'name': rec['body_name'], 'norm': norm, 'tokens': self._tokenize(rec['body_name'])}, spec.get('family')) if spec.get('family') else True
            body_name_l = str(rec['body_name']).lower()
            if spec.get('family') == 'pelvis' and 'pelvis' in body_name_l:
                fam_ok = True
                side_ok = True
            if spec.get('family') == 'upperarm' and 'shoulder' in body_name_l:
                fam_ok = True
            report[mname] = {'body': rec['body_name'], 'side_ok': bool(side_ok), 'family_ok': bool(fam_ok), 'status': 'ok' if side_ok and fam_ok else 'suspect'}
        return report

    def capture_frame(self, sim_time=None, phase=None):
        if sim_time is None:
            sim_time = float(self.mj_data.time)
        markers = {}
        marker_vels = {}
        for mname, spec in self.marker_defs.items():
            pos = self._marker_world(spec['body_id'], spec['offset_local'])
            vel = self._point_world_velocity(spec['body_id'], pos)
            markers[mname] = pos
            marker_vels[mname] = vel
        segments = {}
        for label, bid in self.segment_ids.items():
            derived = self._derived_segment_record(label, markers, marker_vels)
            if derived is not None:
                segments[label] = derived
                continue
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, bid, vel6, 0)
            segments[label] = {'body_name': self._body_name(bid), 'position': self.mj_data.xpos[bid].copy(), 'orientation_quat': self.mj_data.xquat[bid].copy(), 'angular_velocity': vel6[:3].copy(), 'linear_velocity': vel6[3:].copy(), 'source': 'body_frame'}
        joints = {}
        for rec in self._joint_exports:
            joints[rec['joint_name']] = {'qpos': float(self.mj_data.qpos[rec['qpos_adr']]), 'value_export': self._joint_angle_deg(rec), 'unit': 'deg' if rec['joint_type'] != getattr(mujoco.mjtJoint, 'mjJNT_SLIDE', -3) else 'm', 'qvel': float(self.mj_data.qvel[rec['dof_adr']]) if rec['dof_adr'] >= 0 and rec['dof_adr'] < len(self.mj_data.qvel) else 0.0}
        pelvis_body = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        head_body = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Head')
        pelvis = markers.get('SACR', self.mj_data.xpos[pelvis_body] if pelvis_body >= 0 else np.zeros(3))
        head = markers.get('HEAD', self.mj_data.xpos[head_body] if head_body >= 0 else np.array([0.0, 0.0, 1.0]))
        trunk_vec = np.asarray(head) - np.asarray(pelvis)
        trunk_norm = float(np.linalg.norm(trunk_vec))
        trunk_lean_deg = 0.0
        if trunk_norm > 1e-08:
            trunk_lean_deg = float(np.degrees(np.arccos(np.clip(np.dot(trunk_vec / trunk_norm, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))))
        frame = {'time': float(sim_time), 'phase': phase or '', 'markers': markers, 'marker_velocities': marker_vels, 'segments': segments, 'joints': joints, 'pelvis_height': float(pelvis[2]), 'trunk_lean_deg': trunk_lean_deg}
        self.frames.append(frame)
        return frame

    def marker_summary(self):
        qc = self.quality_report()
        suspect = {k: v for k, v in qc.items() if v.get('status') == 'suspect'}
        return {'num_markers': len(self.marker_defs), 'marker_names': list(self.marker_defs.keys()), 'segments': {k: 'derived_marker_triad' if k in ('left_foot', 'right_foot') else self._body_name(v) for k, v in self.segment_ids.items()}, 'num_joint_exports': len(self._joint_exports), 'resolved_bodies': {m: rec['body_name'] for m, rec in self.marker_defs.items()}, 'suspect_markers': suspect, 'num_suspect_markers': len(suspect)}

    def export_marker_csv(self, filename):
        import csv
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        marker_names = list(self.marker_defs.keys())
        trc_names = [self.OPENSIM_TRC_NAMES.get(n, n) for n in marker_names]
        headers = ['time', 'phase']
        for m in marker_names:
            headers += [f'{m}_x', f'{m}_y', f'{m}_z', f'{m}_vx', f'{m}_vy', f'{m}_vz']
        headers += ['pelvis_height', 'trunk_lean_deg']
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for fr in self.frames:
                row = [round(fr['time'], 6), fr['phase']]
                for m in marker_names:
                    p = fr['markers'][m]
                    v = fr['marker_velocities'][m]
                    row += [round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6), round(float(v[0]), 6), round(float(v[1]), 6), round(float(v[2]), 6)]
                row += [round(fr['pelvis_height'], 6), round(fr['trunk_lean_deg'], 6)]
                writer.writerow(row)
        return {'filename': filename, 'frames': len(self.frames), 'markers': len(marker_names)}

    def export_segment_csv(self, filename):
        import csv
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        segs = list(self.segment_ids.keys())
        headers = ['time', 'phase']
        for s in segs:
            headers += [f'{s}_px', f'{s}_py', f'{s}_pz', f'{s}_qw', f'{s}_qx', f'{s}_qy', f'{s}_qz', f'{s}_wx', f'{s}_wy', f'{s}_wz', f'{s}_vx', f'{s}_vy', f'{s}_vz']
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for fr in self.frames:
                row = [round(fr['time'], 6), fr['phase']]
                for s in segs:
                    rec = fr['segments'][s]
                    p = rec['position']
                    q = rec['orientation_quat']
                    w = rec['angular_velocity']
                    v = rec['linear_velocity']
                    row += [round(float(x), 6) for x in (*p, *q, *w, *v)]
                writer.writerow(row)
        return {'filename': filename, 'frames': len(self.frames), 'segments': len(segs)}

    def export_joint_csv(self, filename):
        import csv
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        joint_names = list(self.frames[0]['joints'].keys())
        headers = ['time', 'phase']
        for j in joint_names:
            headers += [f'{j}_value', f'{j}_qvel']
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for fr in self.frames:
                row = [round(fr['time'], 6), fr['phase']]
                for j in joint_names:
                    rec = fr['joints'][j]
                    row += [round(float(rec['value_export']), 6), round(float(rec['qvel']), 6)]
                writer.writerow(row)
        return {'filename': filename, 'frames': len(self.frames), 'joints': len(joint_names)}

    def export_trc(self, filename, data_rate=None):
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        marker_names = list(self.marker_defs.keys())
        trc_names = [self.OPENSIM_TRC_NAMES.get(n, n) for n in marker_names]
        times = np.array([fr['time'] for fr in self.frames], dtype=float)
        if len(times) > 1:
            dt = float(np.median(np.diff(times)))
            frame_rate = float(1.0 / max(dt, 1e-09))
        else:
            frame_rate = float(data_rate or self.export_hz)
        data_rate = float(data_rate or frame_rate)
        num_frames = len(self.frames)
        num_markers = len(marker_names)
        units = 'm'
        with open(filename, 'w') as f:
            f.write(f'PathFileType\t4\t(X/Y/Z)\t{Path(filename).name}\n')
            f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
            f.write(f'{data_rate:.6f}\t{frame_rate:.6f}\t{num_frames}\t{num_markers}\t{units}\t{data_rate:.6f}\t1\t{num_frames}\n')
            header = ['Frame#', 'Time']
            sub = ['', '']
            for i, name in enumerate(trc_names, start=1):
                header += [name, '', '']
                sub += [f'X{i}', f'Y{i}', f'Z{i}']
            f.write('\t'.join(header) + '\n')
            f.write('\t'.join(sub) + '\n')
            for idx, fr in enumerate(self.frames, start=1):
                row = [str(idx), f"{fr['time']:.6f}"]
                for name in marker_names:
                    p = fr['markers'][name]
                    row += [f'{float(p[0]):.6f}', f'{float(p[1]):.6f}', f'{float(p[2]):.6f}']
                f.write('\t'.join(row) + '\n')
        return {'filename': filename, 'frames': num_frames, 'markers': num_markers, 'data_rate': data_rate}

    def export_mot(self, filename, data_rate=None):
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        joint_names = list(self.frames[0]['joints'].keys())
        times = np.array([fr['time'] for fr in self.frames], dtype=float)
        if len(times) > 1:
            dt = float(np.median(np.diff(times)))
            data_rate = float(data_rate or 1.0 / max(dt, 1e-09))
        else:
            data_rate = float(data_rate or self.export_hz)
        with open(filename, 'w') as f:
            f.write(f'name {Path(filename).name}\n')
            f.write(f'datacolumns {1 + len(joint_names)}\n')
            f.write(f'datarows {len(self.frames)}\n')
            f.write(f'range {times[0]:.6f} {times[-1]:.6f}\n')
            f.write('endheader\n')
            f.write('time\t' + '\t'.join(joint_names) + '\n')
            for fr in self.frames:
                vals = [f"{fr['joints'][j]['value_export']:.6f}" for j in joint_names]
                f.write(f"{fr['time']:.6f}\t" + '\t'.join(vals) + '\n')
        return {'filename': filename, 'frames': len(self.frames), 'joints': len(joint_names), 'data_rate': data_rate}

    def _median_filter(self, arr, k=5):
        arr = np.asarray(arr, dtype=float)
        if k <= 1 or arr.size == 0:
            return arr.copy()
        h = k // 2
        out = np.empty_like(arr)
        for i in range(arr.size):
            lo = max(0, i - h)
            hi = min(arr.size, i + h + 1)
            out[i] = np.nanmedian(arr[lo:hi])
        return out

    def export_visuals(self, prefix):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            return {'available': False, 'error': f'matplotlib unavailable: {e}', 'plots': []}
        if not self.frames:
            return {'available': False, 'plots': []}
        times = np.array([fr['time'] for fr in self.frames], dtype=float)
        pelvis_h = np.array([fr['pelvis_height'] for fr in self.frames], dtype=float)
        trunk = np.array([fr['trunk_lean_deg'] for fr in self.frames], dtype=float)
        plots = []
        snapshot_idxs = self._select_snapshot_indices()
        names3d = [n for n in self.VISUAL_MARKERS if n in self.marker_defs]
        pose_idx = self._select_peak_pose_index()

        def world_pts_for_frame(idx):
            return {n: np.asarray(self.frames[idx]['markers'][n], dtype=float) for n in names3d}

        def local_pts_for_frame(idx):
            return self._points_to_presentation_local(world_pts_for_frame(idx))
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(times, pelvis_h, label='Pelvis height [m]', linewidth=2.0)
        ax.plot(times, trunk / 100.0, label='Trunk lean /100 [deg]', linewidth=2.0)
        for idx in snapshot_idxs:
            ax.axvline(times[idx], color='0.82', linestyle='--', linewidth=0.9)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Value')
        ax.set_title('Layer-1 kinematic summary')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fn1 = f'{prefix}_kinematics_summary.png'
        fig.tight_layout()
        fig.savefig(fn1, dpi=180)
        plt.close(fig)
        plots.append(fn1)
        if len(names3d) >= 8 and snapshot_idxs:
            fig = plt.figure(figsize=(12.5, 3.9))
            frames_local = []
            all_pts = []
            for idx in snapshot_idxs:
                pts_local = local_pts_for_frame(idx)
                frames_local.append(pts_local)
                all_pts.extend(self._collect_plot_points(pts_local, names3d))
            lims = self._compute_equal_limits(all_pts, dims=(0, 2), pad=0.05)
            for k, (idx, pts) in enumerate(zip(snapshot_idxs, frames_local), start=1):
                ax = fig.add_subplot(1, len(snapshot_idxs), k)
                self._plot_edges_2d(ax, pts, 0, 2, marker_size=22, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
                ax.set_title(self._snapshot_label(idx))
                ax.set_xlabel('Forward [m]')
                ax.set_ylabel('Up [m]')
                ax.grid(True, alpha=0.3)
                if lims:
                    ax.set_xlim(*lims[0])
                    ax.set_ylim(*lims[1])
                ax.set_aspect('equal', adjustable='box')
            fig.suptitle('Sagittal marker-skeleton snapshots (pelvis-aligned)')
            fig.tight_layout()
            fn2 = f'{prefix}_sagittal_snapshots.png'
            fig.savefig(fn2, dpi=180)
            plt.close(fig)
            plots.append(fn2)
        traj_markers = [n for n in ['HEAD', 'SACR', 'TOE_L', 'TOE_R', 'HEE_L', 'HEE_R'] if n in self.marker_defs]
        walk_frames = [fr for fr in self.frames if fr.get('phase') == 'walk']
        traj_frames = walk_frames if len(walk_frames) >= 40 else self.frames
        decim = max(1, len(traj_frames) // 180)
        if len(traj_markers) >= 2:
            fig = plt.figure(figsize=(6.4, 6.2))
            ax = fig.add_subplot(111)
            sac = np.array([fr['markers']['SACR'] for fr in traj_frames], dtype=float) if 'SACR' in self.marker_defs else None
            head = np.array([fr['markers']['HEAD'] for fr in traj_frames], dtype=float) if 'HEAD' in self.marker_defs else None
            lf = 0.5 * (np.array([fr['markers']['TOE_L'] for fr in traj_frames], dtype=float) + np.array([fr['markers']['HEE_L'] for fr in traj_frames], dtype=float)) if all((m in self.marker_defs for m in ('TOE_L', 'HEE_L'))) else None
            rf = 0.5 * (np.array([fr['markers']['TOE_R'] for fr in traj_frames], dtype=float) + np.array([fr['markers']['HEE_R'] for fr in traj_frames], dtype=float)) if all((m in self.marker_defs for m in ('TOE_R', 'HEE_R'))) else None
            if sac is not None:
                ax.plot(sac[::decim, 0], sac[::decim, 1], linewidth=2.6, label='SACR path')
                ax.scatter([sac[0, 0]], [sac[0, 1]], s=30, color=ax.lines[-1].get_color())
                ax.scatter([sac[-1, 0]], [sac[-1, 1]], s=42, marker='x', color=ax.lines[-1].get_color())
            if head is not None:
                ax.plot(head[::decim, 0], head[::decim, 1], linewidth=1.8, alpha=0.9, label='HEAD path')
            if lf is not None:
                ax.plot(lf[::decim, 0], lf[::decim, 1], linewidth=1.4, alpha=0.9, label='Left foot center')
            if rf is not None:
                ax.plot(rf[::decim, 0], rf[::decim, 1], linewidth=1.4, alpha=0.9, label='Right foot center')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title('Global top-view walk trajectory')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            fn3 = f'{prefix}_topview_trajectories.png'
            fig.savefig(fn3, dpi=180)
            plt.close(fig)
            plots.append(fn3)
        if 'SACR' in self.marker_defs and len(traj_markers) >= 2:
            head_xy, lf_xy, rf_xy = ([], [], [])
            for fr in traj_frames:
                pts_ref = {n: np.asarray(fr['markers'][n], dtype=float) for n in set(['SACR', 'HEAD', 'HIP_L', 'HIP_R', 'ASI_L', 'ASI_R', 'SHO_L', 'SHO_R']) if n in fr['markers']}
                origin, forward, lateral, up = self._presentation_frame(pts_ref)
                if 'HEAD' in fr['markers']:
                    head_xy.append(self._point_to_presentation_local(fr['markers']['HEAD'], origin, forward, lateral, up)[:2])
                if 'TOE_L' in fr['markers'] and 'HEE_L' in fr['markers']:
                    lf = 0.5 * (np.asarray(fr['markers']['TOE_L'], dtype=float) + np.asarray(fr['markers']['HEE_L'], dtype=float))
                    lf_xy.append(self._point_to_presentation_local(lf, origin, forward, lateral, up)[:2])
                if 'TOE_R' in fr['markers'] and 'HEE_R' in fr['markers']:
                    rf = 0.5 * (np.asarray(fr['markers']['TOE_R'], dtype=float) + np.asarray(fr['markers']['HEE_R'], dtype=float))
                    rf_xy.append(self._point_to_presentation_local(rf, origin, forward, lateral, up)[:2])
            fig = plt.figure(figsize=(6.4, 6.2))
            ax = fig.add_subplot(111)
            if head_xy:
                rel_head = np.asarray(head_xy, dtype=float)
                if rel_head.shape[0] > 5:
                    rel_head[:, 0] = self._median_filter(rel_head[:, 0], 7)
                    rel_head[:, 1] = self._median_filter(rel_head[:, 1], 7)
                ax.plot(rel_head[::decim, 0], rel_head[::decim, 1], linewidth=1.8, label='HEAD')
            if lf_xy:
                rel_lf = np.asarray(lf_xy, dtype=float)
                if rel_lf.shape[0] > 5:
                    rel_lf[:, 0] = self._median_filter(rel_lf[:, 0], 5)
                    rel_lf[:, 1] = self._median_filter(rel_lf[:, 1], 5)
                ax.scatter(rel_lf[::decim, 0], rel_lf[::decim, 1], s=12, alpha=0.3, label='Left foot center')
            if rf_xy:
                rel_rf = np.asarray(rf_xy, dtype=float)
                if rel_rf.shape[0] > 5:
                    rel_rf[:, 0] = self._median_filter(rel_rf[:, 0], 5)
                    rel_rf[:, 1] = self._median_filter(rel_rf[:, 1], 5)
                ax.scatter(rel_rf[::decim, 0], rel_rf[::decim, 1], s=12, alpha=0.3, label='Right foot center')
            ax.scatter([0.0], [0.0], s=42, color='black', alpha=0.8, label='SACR origin')
            ax.axhline(0.0, color='0.7', linewidth=0.8)
            ax.axvline(0.0, color='0.7', linewidth=0.8)
            ax.set_xlabel('Forward rel. to pelvis [m]')
            ax.set_ylabel('Lateral rel. to pelvis [m]')
            ax.set_title('Pelvis-centred top-view workspace (pelvis-aligned)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            fn4 = f'{prefix}_topview_relative.png'
            fig.savefig(fn4, dpi=180)
            plt.close(fig)
            plots.append(fn4)
        if len(names3d) >= 10 and snapshot_idxs:
            fig = plt.figure(figsize=(14.5, 4.0))
            frames_local = []
            all_pts = []
            for idx in snapshot_idxs:
                pts_local = local_pts_for_frame(idx)
                frames_local.append(pts_local)
                all_pts.extend(self._collect_plot_points(pts_local, names3d))
            lims3d = self._compute_equal_limits_3d(all_pts, pad=0.06)
            for k, (idx, pts) in enumerate(zip(snapshot_idxs, frames_local), start=1):
                ax = fig.add_subplot(1, len(snapshot_idxs), k, projection='3d')
                self._plot_edges_3d(ax, pts, perm=(0, 1, 2), marker_size=14, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
                ax.set_title(self._snapshot_label(idx))
                ax.set_xlabel('Forward')
                ax.set_ylabel('Lateral')
                ax.set_zlabel('Up')
                if lims3d:
                    ax.set_xlim(*lims3d[0])
                    ax.set_ylim(*lims3d[1])
                    ax.set_zlim(*lims3d[2])
                ax.view_init(elev=18, azim=-68)
            fig.suptitle('3D marker-skeleton snapshots (pelvis-centred)')
            fig.tight_layout()
            fn5 = f'{prefix}_skeleton_3d_snapshots.png'
            fig.savefig(fn5, dpi=180)
            plt.close(fig)
            plots.append(fn5)
            pts = local_pts_for_frame(pose_idx)
            fig = plt.figure(figsize=(11.5, 3.9))
            views = [((0, 2), 'Sagittal forward-up'), ((1, 2), 'Frontal lateral-up'), ((0, 1), 'Top forward-lateral')]
            labels = [('Forward [m]', 'Up [m]'), ('Lateral [m]', 'Up [m]'), ('Forward [m]', 'Lateral [m]')]
            all_pts = self._collect_plot_points(pts, names3d)
            for k, ((xi, yi), ttl) in enumerate(views, start=1):
                ax = fig.add_subplot(1, 3, k)
                self._plot_edges_2d(ax, pts, xi, yi, marker_size=26, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
                lims = self._compute_equal_limits(all_pts, dims=(xi, yi), pad=0.05)
                ax.set_title(ttl)
                ax.set_xlabel(labels[k - 1][0])
                ax.set_ylabel(labels[k - 1][1])
                ax.grid(True, alpha=0.3)
                if lims:
                    ax.set_xlim(*lims[0])
                    ax.set_ylim(*lims[1])
                ax.set_aspect('equal', adjustable='box')
            fig.suptitle(f"Pose-style marker skeleton ({self._snapshot_label(pose_idx).replace(chr(10), ' | ')})")
            fig.tight_layout()
            fn6 = f'{prefix}_pose_style_views.png'
            fig.savefig(fn6, dpi=180)
            plt.close(fig)
            plots.append(fn6)
            fig = plt.figure(figsize=(7.4, 4.0))
            ax = fig.add_subplot(111, projection='3d')
            self._plot_edges_3d(ax, pts, perm=(0, 1, 2), marker_size=18, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
            ax.set_title('OpenPose-style fall skeleton (pelvis-centred)')
            ax.set_xlabel('Forward')
            ax.set_ylabel('Lateral')
            ax.set_zlabel('Up')
            if lims3d:
                ax.set_xlim(*lims3d[0])
                ax.set_ylim(*lims3d[1])
                ax.set_zlim(*lims3d[2])
            ax.view_init(elev=18, azim=118)
            fig.tight_layout()
            fn6b = f'{prefix}_openpose_style_fall.png'
            fig.savefig(fn6b, dpi=180)
            plt.close(fig)
            plots.append(fn6b)
            fig = plt.figure(figsize=(12.5, 8.0))
            gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])
            ax_time = fig.add_subplot(gs[0, :])
            ax_time.plot(times, pelvis_h, label='Pelvis height [m]', linewidth=2.0)
            ax_time.plot(times, trunk / 100.0, label='Trunk lean /100 [deg]', linewidth=2.0)
            for idx in snapshot_idxs:
                ax_time.axvline(times[idx], color='0.82', linestyle='--', linewidth=0.9)
            ax_time.scatter([times[pose_idx]], [pelvis_h[pose_idx]], s=42, color='tab:red', zorder=3)
            ax_time.set_xlabel('Time [s]')
            ax_time.set_ylabel('Value')
            ax_time.set_title('Layer-1 overview: timeline + presentation geometry')
            ax_time.grid(True, alpha=0.3)
            ax_time.legend(loc='upper right')
            ax_top = fig.add_subplot(gs[1, 0])
            if head_xy:
                rel_head = np.asarray(head_xy, dtype=float)
                if rel_head.shape[0] > 5:
                    rel_head[:, 0] = self._median_filter(rel_head[:, 0], 7)
                    rel_head[:, 1] = self._median_filter(rel_head[:, 1], 7)
                ax_top.plot(rel_head[::decim, 0], rel_head[::decim, 1], linewidth=1.5, label='HEAD')
            if lf_xy:
                rel_lf = np.asarray(lf_xy, dtype=float)
                if rel_lf.shape[0] > 5:
                    rel_lf[:, 0] = self._median_filter(rel_lf[:, 0], 5)
                    rel_lf[:, 1] = self._median_filter(rel_lf[:, 1], 5)
                ax_top.scatter(rel_lf[::decim, 0], rel_lf[::decim, 1], s=8, alpha=0.28, label='Left foot')
            if rf_xy:
                rel_rf = np.asarray(rf_xy, dtype=float)
                if rel_rf.shape[0] > 5:
                    rel_rf[:, 0] = self._median_filter(rel_rf[:, 0], 5)
                    rel_rf[:, 1] = self._median_filter(rel_rf[:, 1], 5)
                ax_top.scatter(rel_rf[::decim, 0], rel_rf[::decim, 1], s=8, alpha=0.28, label='Right foot')
            ax_top.scatter([0.0], [0.0], s=24, color='black', alpha=0.8)
            ax_top.set_title('Pelvis-aligned walk workspace')
            ax_top.set_xlabel('Forward rel. to pelvis [m]')
            ax_top.set_ylabel('Lateral rel. to pelvis [m]')
            ax_top.grid(True, alpha=0.3)
            ax_top.set_aspect('equal', adjustable='box')
            ax_pose = fig.add_subplot(gs[1, 1])
            self._plot_edges_2d(ax_pose, pts, 0, 2, marker_size=28, edges=self.VISUAL_SKELETON_EDGES, names=names3d)
            lims = self._compute_equal_limits(self._collect_plot_points(pts, names3d), dims=(0, 2), pad=0.05)
            if lims:
                ax_pose.set_xlim(*lims[0])
                ax_pose.set_ylim(*lims[1])
            ax_pose.set_title('Peak-fall sagittal pose')
            ax_pose.set_xlabel('Forward [m]')
            ax_pose.set_ylabel('Up [m]')
            ax_pose.grid(True, alpha=0.3)
            ax_pose.set_aspect('equal', adjustable='box')
            ax_text = fig.add_subplot(gs[1, 2])
            ax_text.axis('off')
            ax_text.text(0.0, 1.0, '\n'.join(self._quality_summary_lines()), va='top', ha='left', fontsize=10, family='monospace')
            fig.tight_layout()
            fn7 = f'{prefix}_layer1_overview.png'
            fig.savefig(fn7, dpi=180)
            plt.close(fig)
            plots.append(fn7)
        return {'available': True, 'plots': plots}

    def _plot_edges_2d(self, ax, pts, xidx, yidx, marker_size=10, edges=None, names=None):
        edges = edges or self.VISUAL_SKELETON_EDGES
        for a, b in edges:
            if a in pts and b in pts:
                pa = pts[a]
                pb = pts[b]
                style = self._edge_style(a, b)
                ax.plot([pa[xidx], pb[xidx]], [pa[yidx], pb[yidx]], '-', **style)
        names = names or [n for n in self.VISUAL_MARKERS if n in pts]
        if names:
            arr = np.array([pts[n] for n in names], dtype=float)
            ax.scatter(arr[:, xidx], arr[:, yidx], s=marker_size, color='#1f77b4', edgecolors='white', linewidths=0.4, zorder=3)

    def _plot_edges_3d(self, ax, pts, perm=(0, 1, 2), marker_size=12, edges=None, names=None):
        i, j, k = perm
        edges = edges or self.VISUAL_SKELETON_EDGES
        for a, b in edges:
            if a in pts and b in pts:
                pa = pts[a]
                pb = pts[b]
                style = self._edge_style(a, b)
                ax.plot([pa[i], pb[i]], [pa[j], pb[j]], [pa[k], pb[k]], '-', color=style['color'], linewidth=style['linewidth'])
        names = names or [n for n in self.VISUAL_MARKERS if n in pts]
        if names:
            arr = np.array([pts[n] for n in names], dtype=float)
            ax.scatter(arr[:, i], arr[:, j], arr[:, k], s=marker_size, color='#1f77b4', depthshade=True)

    def export_pose_json(self, filename):
        import json
        rows = []
        for idx, fr in enumerate(self.frames):
            pts = {k: list(map(float, v)) for k, v in fr['markers'].items() if k in self.POSE_MARKERS}
            pts_opensim = {self.OPENSIM_TRC_NAMES.get(k, k): v for k, v in pts.items()}
            rows.append({'frame': idx, 'time': float(fr['time']), 'markers_3d': pts, 'markers_3d_opensim': pts_opensim})
        with open(filename, 'w') as f:
            json.dump({'pose_markers': self.POSE_MARKERS, 'opensim_names': self.OPENSIM_TRC_NAMES, 'frames': rows}, f)
        return {'filename': filename, 'frames': len(rows), 'markers': len(self.POSE_MARKERS)}

    def export_quality_csv(self, filename):
        import csv
        qc = self.quality_report()
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['marker', 'status', 'body', 'side_ok', 'family_ok'])
            for m, rec in qc.items():
                writer.writerow([m, rec.get('status', 'missing'), rec.get('body', ''), rec.get('side_ok', ''), rec.get('family_ok', '')])
        return {'filename': filename, 'markers': len(qc)}

    def export_bundle(self, prefix):
        marker_csv = self.export_marker_csv(f'{prefix}_markers.csv')
        seg_csv = self.export_segment_csv(f'{prefix}_segments.csv')
        joint_csv = self.export_joint_csv(f'{prefix}_joints.csv')
        trc = self.export_trc(f'{prefix}_markers.trc')
        opensim_trc = self.export_trc(f'{prefix}_markers_opensim.trc')
        mot = self.export_mot(f'{prefix}_joints.mot')
        quality_csv = self.export_quality_csv(f'{prefix}_marker_quality.csv')
        pose_json = self.export_pose_json(f'{prefix}_pose_markers.json')
        visuals = self.export_visuals(prefix)
        return {'marker_csv': marker_csv, 'segment_csv': seg_csv, 'joint_csv': joint_csv, 'trc': trc, 'opensim_trc': opensim_trc, 'mot': mot, 'quality_csv': quality_csv, 'pose_json': pose_json, 'visuals': visuals, 'summary': self.marker_summary()}

class FallValidator:

    def __init__(self):
        self.thresholds = {'head_impact_velocity': 4.5, 'pelvis_impact_velocity': 3.0, 'max_angular_velocity': 25.0, 'min_protective_time': 0.15, 'max_fall_duration': 4.0, 'impact_sequence_time': 0.5}

    def validate_fall(self, imu_data, kinematic_data, fall_type='backward_walking', perturb_start_time=None, event_summary=None):
        """
        FIX Z2: perturb_start_time is the timestamp when the perturbation began.
                fall_start is only searched AFTER this time, preventing false
                long-duration detection when the model fell during stand/walk.
        FIX Z3: jerk computed from 7-tap filtered accel, not raw.
        """
        results = {'overall_score': 0.0, 'checks': {}, 'warnings': [], 'recommendations': []}
        scores = []
        if 'head_velocity' in kinematic_data:
            head_v = np.linalg.norm(kinematic_data['head_velocity'])
            check = head_v < self.thresholds['head_impact_velocity']
            results['checks']['head_velocity_safe'] = check
            if not check:
                results['warnings'].append(f'Head impact {head_v:.2f} m/s exceeds threshold')
            scores.append(1.0 if check else max(0, 1.0 - (head_v - 4.5) / 4.5))
        ts, heights, velocities = (imu_data.get('timestamp', []), imu_data.get('pelvis_height', []), imu_data.get('pelvis_velocity', []))
        impacts = imu_data.get('impact_force', [])
        fall_duration = 0.0
        if event_summary and event_summary.get('available', False):
            fall_duration = float(event_summary.get('fall_duration_s', 0.0))
            check = 0.3 <= fall_duration <= 4.0
            results['checks']['duration_realistic'] = check
            if not check:
                if fall_duration == 0.0:
                    results['warnings'].append('No fall event detected after perturbation')
                elif fall_duration < 0.3:
                    results['warnings'].append(f'Fall {fall_duration:.2f}s unrealistically fast')
                else:
                    results['warnings'].append(f'Fall {fall_duration:.2f}s too long (>4s) - body not settling')
                    results['recommendations'].append('Check muscle weakening, rest-mode settling, and perturbation timing')
            scores.append(1.0 if check else 0.5)
        elif ts and heights:
            search_from = 0
            if perturb_start_time is not None:
                for j, t in enumerate(ts):
                    if t >= perturb_start_time:
                        search_from = j
                        break
            h_ref = float(max(heights[:max(search_from + 1, min(len(heights), search_from + 30))])) if heights else 0.0
            fall_start_idx = None
            for j in range(search_from, len(heights)):
                speed_xy = 0.0
                if j < len(velocities):
                    v = velocities[j]
                    if hasattr(v, '__len__') and len(v) >= 2:
                        speed_xy = float(np.linalg.norm(np.asarray(v)[:2]))
                    else:
                        speed_xy = abs(float(v))
                if heights[j] < max(0.4, 0.65 * h_ref) or speed_xy > 0.9:
                    fall_start_idx = j
                    break
            main_impact_idx = None
            if fall_start_idx is not None and impacts:
                peak_imp = float(np.max(impacts[fall_start_idx:])) if len(impacts) > fall_start_idx else 0.0
                thr = max(250.0, 0.35 * peak_imp)
                for j in range(fall_start_idx, len(impacts)):
                    if impacts[j] >= thr:
                        main_impact_idx = j
                        break
            fall_end_idx = None
            search_end_from = main_impact_idx if main_impact_idx is not None else fall_start_idx
            if search_end_from is not None:
                streak = 0
                for j in range(search_end_from, len(velocities)):
                    v = velocities[j]
                    if hasattr(v, '__len__') and len(v) >= 2:
                        speed_xy = float(np.linalg.norm(np.asarray(v)[:2]))
                    else:
                        speed_xy = abs(float(v))
                    low_h = heights[j] < 0.28
                    streak = streak + 1 if speed_xy < 0.12 and low_h else 0
                    if streak >= 8:
                        fall_end_idx = j - 7
                        break
            if fall_start_idx is not None and fall_end_idx is not None:
                fall_duration = ts[fall_end_idx] - ts[fall_start_idx]
            elif fall_start_idx is not None:
                fall_duration = ts[-1] - ts[fall_start_idx]
            check = 0.3 <= fall_duration <= 4.0
            results['checks']['duration_realistic'] = check
            if not check:
                if fall_duration == 0.0:
                    results['warnings'].append('No fall event detected after perturbation')
                elif fall_duration < 0.3:
                    results['warnings'].append(f'Fall {fall_duration:.2f}s unrealistically fast')
                else:
                    results['warnings'].append(f'Fall {fall_duration:.2f}s too long (>4s) - body not settling')
                    results['recommendations'].append('Check muscle weakening and perturbation timing')
            scores.append(1.0 if check else 0.5)
        raw_accels = imu_data.get('accel_raw', imu_data.get('accelerometer', []))
        if len(raw_accels) > 10 and SCIPY_AVAILABLE:
            acc_arr = np.array(raw_accels)
            if len(acc_arr) >= 7:
                k7 = np.hanning(7)
                k7 /= k7.sum()
                acc_mags = np.linalg.norm(acc_arr, axis=1)
                filt_m = np.convolve(acc_mags, k7, mode='same')
                filt_m[:3] = acc_mags[:3]
                filt_m[-3:] = acc_mags[-3:]
                scale = np.where(acc_mags > 1e-06, filt_m / acc_mags, 1.0)
                acc_filt = acc_arr * scale[:, None]
            else:
                acc_filt = acc_arr
            ts_arr = np.array(ts[:len(acc_filt)])
            dt = np.diff(ts_arr)
            dt = np.where(dt == 0, 1e-06, dt)
            jerk = np.diff(acc_filt, axis=0) / dt.reshape(-1, 1)
            max_jerk = float(np.max(np.linalg.norm(jerk, axis=1)))
            check = 5 < max_jerk < 2000
            results['checks']['jerk_realistic'] = check
            if not check:
                results['warnings'].append('Jerk too low - fall may be too slow' if max_jerk <= 5 else 'Jerk extremely high - check solver')
            scores.append(1.0 if check else 0.7)
        scores.append(self._check_protective_response(imu_data))
        results['checks']['protective_response'] = scores[-1] > 0.5
        conf_buf = imu_data.get('sensor_confidence', [])
        if conf_buf:
            avg_conf = float(np.mean(conf_buf))
            results['checks']['imu_quality'] = avg_conf > 0.6
            scores.append(avg_conf)
        impacts = imu_data.get('impact_force', [])
        if impacts:
            imp_arr = np.array(impacts)
            transitions, in_impact, settled = (0, False, False)
            last_end, zero_streak = (-9, 0)
            for j, f in enumerate(imp_arr):
                if not settled:
                    if not in_impact and f > 150.0:
                        if j - last_end >= 9:
                            transitions += 1
                            in_impact = True
                            if f >= 500.0:
                                settled = True
                    elif in_impact and f < 80.0:
                        last_end = j
                        in_impact = False
                else:
                    zero_streak = zero_streak + 1 if f < 5.0 else 0
                    if zero_streak >= 15 and f > 150.0:
                        transitions += 1
                        zero_streak = 0
            check = 1 <= transitions <= 8
            results['checks']['contact_pattern'] = check
            if not check:
                results['warnings'].append(f'Impact transitions={transitions} (expected 1-8)' if transitions > 0 else 'No significant ground contacts detected')
            scores.append(1.0 if check else 0.6)
        results['overall_score'] = float(np.mean(scores)) if scores else 0.0
        if impacts:
            pos = np.array(impacts)
            pos = pos[pos > 0]
            if len(pos) > 1:
                ps = np.sort(pos)
                n = len(ps)
                gini = float(2 * np.sum(np.arange(1, n + 1) * ps) / (n * ps.sum()) - (n + 1) / n)
                gini = max(0.0, min(1.0, gini))
                results['advanced_metrics'] = results.get('advanced_metrics', {})
                results['advanced_metrics']['gini_impact'] = round(gini, 3)
                results['advanced_metrics']['gini_status'] = 'REALISTIC' if 0.5 < gini < 0.95 else 'ATYPICAL'
        gyros = imu_data.get('gyroscope', [])
        if gyros and len(gyros) > 10:
            ga = np.array(gyros)
            gm = np.linalg.norm(ga, axis=1)
            results['advanced_metrics'] = results.get('advanced_metrics', {})
            results['advanced_metrics']['peak_angular_velocity_rad_s'] = round(float(np.max(gm)), 3)
            results['advanced_metrics']['resting_angular_velocity_rad_s'] = round(float(np.mean(gm[-30:])) if len(gm) >= 30 else 0.0, 3)
            results['advanced_metrics']['angular_momentum_status'] = 'OK' if float(np.max(gm)) > 0.5 and (float(np.mean(gm[-30:])) if len(gm) >= 30 else 1.0) < 0.5 else 'CHECK'
        if results['overall_score'] > 0.8:
            results['classification'] = 'HIGH_CONFIDENCE'
        elif results['overall_score'] > 0.6:
            results['classification'] = 'MODERATE_CONFIDENCE'
        else:
            results['classification'] = 'LOW_CONFIDENCE'
        return results

    def _check_protective_response(self, imu_data):
        impacts = imu_data.get('impact_force', [])
        if len(impacts) < 20:
            return 0.5
        imp_arr = np.array(impacts)
        if not SCIPY_AVAILABLE:
            small = np.where(imp_arr > 30)[0]
            large = np.where(imp_arr > 500)[0]
            if len(small) > 0 and len(large) > 0:
                return 0.8 if small[0] < large[0] else 0.4
            return 0.5
        peaks, _ = find_peaks(imp_arr, height=30.0, distance=3)
        if len(peaks) == 0:
            return 0.3
        if len(peaks) >= 2:
            return 0.9 if imp_arr[peaks[0]] < imp_arr[peaks[1]] else 0.6
        return 0.4

    def validate_biomechanical_ranges(self, imu_data_buffer, body_mass=70.0, dynamics_frames=None):
        g, BW = (9.81, body_mass * 9.81)
        results = {'body_mass_kg': body_mass, 'body_weight_n': BW}
        gyros = imu_data_buffer.get('gyroscope', [])
        if gyros:
            ga = np.array(gyros)
            peak_omg = float(np.max(np.linalg.norm(ga, axis=1)) if ga.ndim == 2 else np.max(np.abs(ga)))
            omg_t = self.thresholds.get('max_angular_velocity', 25.0)
            results.update({'peak_angular_velocity_ok': peak_omg < omg_t, 'peak_angular_velocity_rads': peak_omg, 'angular_velocity_threshold': omg_t})
        else:
            results.update({'peak_angular_velocity_ok': None, 'peak_angular_velocity_rads': 0.0})
        impacts = imu_data_buffer.get('impact_force', [])
        peak_f = None
        if dynamics_frames:
            dyn_vals = [max(float(fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0))), float(fr.get('nonfoot_ground_normal_n_filt', fr.get('nonfoot_ground_normal_n', 0.0)))) for fr in dynamics_frames]
            if dyn_vals:
                peak_f = float(np.max(dyn_vals))
        if peak_f is None and impacts:
            peak_f = float(np.max(impacts))
        if peak_f is not None:
            f_lo, f_hi = (1.5 * BW, 20.0 * BW)
            results.update({'impact_force_in_range': f_lo <= peak_f <= f_hi, 'peak_impact_force_n': peak_f, 'expected_range_n': (f_lo, f_hi)})
        else:
            results.update({'impact_force_in_range': None, 'peak_impact_force_n': 0.0, 'expected_range_n': (1.5 * BW, 20.0 * BW)})
        return results

    def validate_capture_point(self, xcom_history, support_center_history, timestamps, body_mass=70.0, leg_length=0.93):
        if not xcom_history:
            return {'error': 'no_xcom_data', 'hof_2005_compliant': False}
        BASE_HALF_WIDTH = 0.15
        margins, fall_pred_time = ([], None)
        for xcom_2d, sup_c, t in zip(xcom_history, support_center_history, timestamps):
            try:
                margin = BASE_HALF_WIDTH - float(np.linalg.norm(np.asarray(xcom_2d) - np.asarray(sup_c)))
            except Exception:
                margin = BASE_HALF_WIDTH
            margins.append(margin)
            if margin < 0.0 and fall_pred_time is None:
                fall_pred_time = float(t)
        return {'fall_prediction_time': fall_pred_time, 'min_margin_m': float(np.min(margins)) if margins else 0.0, 'max_margin_m': float(np.max(margins)) if margins else 0.0, 'num_samples': len(margins), 'hof_2005_compliant': fall_pred_time is not None}

    def generate_report(self, validation_results, output_file=None):
        lines = ['=' * 60, 'FALL SIMULATION VALIDATION REPORT', '=' * 60, f"Overall Confidence: {validation_results['overall_score']:.1%}", f"Classification:     {validation_results['classification']}", '', 'Detailed Checks:']
        for check, passed in validation_results['checks'].items():
            lines.append(f"  {check:35s}  {('PASS' if passed else 'FAIL')}")
        if validation_results['warnings']:
            lines += ['', 'Warnings:']
            for w in validation_results['warnings']:
                lines.append(f'  ! {w}')
        if validation_results.get('recommendations'):
            lines += ['', 'Recommendations:']
            for r in validation_results['recommendations']:
                lines.append(f'  -> {r}')
        adv = validation_results.get('advanced_metrics', {})
        if adv:
            lines += ['', 'Advanced Metrics:']
            for k, v in adv.items():
                lines.append(f'  {k:40s}  {v}')
        lines.append('=' * 60)
        report_text = '\n'.join(lines)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        return report_text

class SISFallValidator:
    """
    FIX U: Pelvis-mounted sensor has different characteristic ranges
    than wrist/chest sensors used in original SISFall dataset.
    Updated ranges based on:
      - van den Bogert et al. 1996: pelvis accelerations during falls
      - Casilari et al. 2017 (Sensors): multi-sensor SISFall analysis
      - Bourke & Lyons 2008: pelvis vs waist sensor comparison
    Pelvis backward fall peak: 15-130 m/s^2 (Casilari 2017 mean=51.2 SD=18.4;
    Bagala 2012: 25-130 m/s^2; 7-tap Hanning filtered, 30Hz sampling).
    """
    SISFALL_RANGES = {'backward': {'peak_accel_ms2': (15.0, 130.0), 'duration_s': (0.9, 2.5)}, 'forward': {'peak_accel_ms2': (12.0, 70.0), 'duration_s': (0.8, 2.0)}, 'lateral': {'peak_accel_ms2': (15.0, 80.0), 'duration_s': (0.7, 1.8)}, 'adl': {'peak_accel_ms2': (5.0, 20.0), 'duration_s': (0.0, 5.0)}}
    FALL_TYPE_TO_DIRECTION = {'backward_walking': 'backward', 'forward_stumble': 'forward', 'lateral_left': 'lateral', 'lateral_right': 'lateral'}

    def validate_sisfall_signature(self, imu_data_buffer, fall_type, body_mass=70.0, perturb_start_time=None, event_summary=None):
        direction = self.FALL_TYPE_TO_DIRECTION.get(fall_type, 'backward')
        ranges = self.SISFALL_RANGES[direction]
        accels = imu_data_buffer.get('accel_raw', imu_data_buffer.get('accelerometer', []))
        ts = imu_data_buffer.get('timestamp', [])
        heights = imu_data_buffer.get('pelvis_height', [])
        result = {'direction': direction, 'checks': {}, 'margins': {}, 'sisfall_compliant': False}
        if not accels:
            result['checks']['data_available'] = False
            return result
        acc_arr = np.array(accels)
        acc_mags = np.linalg.norm(acc_arr, axis=1)
        if len(acc_mags) >= 7:
            k7 = np.hanning(7)
            k7 = k7 / k7.sum()
            acc_mags_f = np.convolve(acc_mags, k7, mode='same')
            acc_mags_f[:3] = acc_mags[:3]
            acc_mags_f[-3:] = acc_mags[-3:]
        else:
            acc_mags_f = acc_mags.copy()
        peak_acc = float(np.max(acc_mags_f))
        peak_raw = float(np.max(acc_mags))
        lo, hi = ranges['peak_accel_ms2']
        result['checks']['peak_accel_in_range'] = lo <= peak_acc <= hi
        result['margins']['peak_accel_ms2'] = peak_acc
        result['margins']['peak_accel_raw_ms2'] = peak_raw
        result['margins']['peak_accel_range'] = (lo, hi)
        result['margins']['channel_used'] = 'accel_raw (FIX-Q/X4/X5: world-frame FD, vector-norm clip, v29)'
        result['margins']['filter_note'] = 'peak_accel_ms2 = FIX-Y5 7-tap Hanning LP (Casilari 2017 pelvis BW)'
        result['margins']['sensor_location'] = 'Lower-back L1-L2 proxy on Torso; comparison thresholds remain legacy pelvis-style'
        search_from_sf = 0
        if perturb_start_time is not None and ts:
            for j, t in enumerate(ts):
                if t >= perturb_start_time:
                    search_from_sf = j
                    break
        fall_start_idx, fall_end_idx = (None, None)
        for j, h in enumerate(heights):
            if j < search_from_sf:
                continue
            if h < 0.4 and fall_start_idx is None:
                fall_start_idx = j
            if fall_start_idx is not None and j > fall_start_idx:
                vel_list = imu_data_buffer.get('pelvis_velocity', [])
                if j < len(vel_list):
                    v = vel_list[j]
                    speed = float(np.linalg.norm(v)) if hasattr(v, '__len__') else abs(float(v))
                    if speed < 0.1:
                        fall_end_idx = j
                        break
        if event_summary and event_summary.get('available', False):
            duration = float(event_summary.get('fall_duration_s', 0.0))
        elif fall_start_idx is not None and ts:
            t_s = ts[fall_start_idx] if fall_start_idx < len(ts) else ts[-1]
            t_e = ts[fall_end_idx] if fall_end_idx is not None and fall_end_idx < len(ts) else ts[-1]
            duration = t_e - t_s
        else:
            duration = 0.0
        dlo, dhi = ranges['duration_s']
        result['checks']['duration_in_range'] = dlo <= duration <= dhi
        result['margins']['fall_duration_s'] = duration
        result['margins']['duration_range'] = (dlo, dhi)
        mf = float(np.sqrt(body_mass / 70.0))
        result['checks']['peak_accel_mass_scaled'] = lo * mf <= peak_acc <= hi * mf
        result['margins']['mass_scale_factor'] = mf
        result['margins']['mass_adjusted_range'] = (lo * mf, hi * mf)
        result['sisfall_compliant'] = result['checks']['peak_accel_in_range'] and result['checks']['duration_in_range']
        return result

class KFallValidator:
    """KFall-style low-back pre-impact validation.

    Uses publication-style checks when the actual KFall dataset files are not
    locally available. This keeps the validator aligned with 100 Hz low-back
    sensing and pre-impact timing expectations.
    """

    def __init__(self):
        self.target_hz = 100.0
        self.window_s = 0.5
        self.benchmark_lead_ms = 403.0

    def validate(self, imu_validator, perturb_start_time=None, event_summary=None):
        resampled = imu_validator._resample_buffers_100hz()
        if resampled is None:
            return {'available': False, 'checks': {'data_available': False}, 'kfall_compliant': False}
        t = np.asarray(resampled['timestamp'], dtype=float)
        acc = np.asarray(resampled['accel_raw'], dtype=float)
        gyro = np.asarray(resampled['gyroscope'], dtype=float)
        heights = np.asarray(resampled['pelvis_height'], dtype=float)
        impacts = np.asarray(resampled['impact_force'], dtype=float)
        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)
        t0 = float(perturb_start_time) if perturb_start_time is not None else float(t[0])
        baseline_start = max(float(t[0]), t0 - 1.0)
        baseline_mask = (t >= baseline_start) & (t < t0)
        if not np.any(baseline_mask):
            baseline_mask = t < t[0] + min(1.0, float(t[-1] - t[0]))
        acc_base = acc_mag[baseline_mask] if np.any(baseline_mask) else acc_mag[:max(10, min(len(acc_mag), 100))]
        gyro_base = gyro_mag[baseline_mask] if np.any(baseline_mask) else gyro_mag[:max(10, min(len(gyro_mag), 100))]
        h_base = heights[baseline_mask] if np.any(baseline_mask) else heights[:max(10, min(len(heights), 100))]
        dt = np.diff(t)
        dt = np.where(dt <= 0, 1e-06, dt)
        jerk_mag = np.zeros_like(acc_mag)
        if len(acc_mag) > 1:
            jerk_mag[1:] = np.abs(np.diff(acc_mag) / dt)
        jerk_base = jerk_mag[baseline_mask] if np.any(baseline_mask) else jerk_mag[:max(10, min(len(jerk_mag), 100))]
        acc_mu, acc_sd = (float(np.mean(acc_base)), float(np.std(acc_base)))
        gyro_mu, gyro_sd = (float(np.mean(gyro_base)), float(np.std(gyro_base)))
        jerk_mu, jerk_sd = (float(np.mean(jerk_base)), float(np.std(jerk_base)))
        h_ref = float(np.mean(h_base)) if len(h_base) else float(np.max(heights[:50])) if len(heights) else 0.0
        acc_thr = max(12.0, acc_mu + 2.5 * acc_sd)
        gyro_thr = max(1.2, gyro_mu + 2.5 * gyro_sd)
        jerk_thr = max(25.0, jerk_mu + 3.0 * jerk_sd)
        height_thr = 0.88 * h_ref if h_ref > 0 else 0.0
        search_indices = np.where((t >= t0) & (t <= t0 + 1.2))[0]
        onset_idx = None
        onset_reason = 'not_found'
        for i in search_indices:
            if i + 1 >= len(t):
                break
            acc_cond = np.all(acc_mag[i:i + 2] > acc_thr)
            gyro_cond = np.all(gyro_mag[i:i + 2] > gyro_thr)
            jerk_cond = np.all(jerk_mag[i:i + 2] > jerk_thr)
            if jerk_cond or acc_cond or gyro_cond:
                onset_idx = i
                onset_reason = 'jerk_threshold' if jerk_cond else 'adaptive_acc_threshold' if acc_cond else 'adaptive_gyro_threshold'
                break
        if onset_idx is None:
            for i in search_indices:
                if i + 4 >= len(t) or t[i] < t0 + 0.25:
                    continue
                h_cond = np.all(heights[i:i + 5] < height_thr) if h_ref > 0 else False
                if h_cond:
                    onset_idx = i
                    onset_reason = 'height_drop_threshold'
                    break
        impact_idx = -1
        peak_imp = 0.0
        if len(impacts):
            search_imp = np.where(t >= t0)[0]
            if len(search_imp):
                peak_imp = float(np.max(impacts[search_imp]))
                thr = max(250.0, 0.35 * peak_imp)
                for i in search_imp:
                    if impacts[i] >= thr:
                        impact_idx = int(i)
                        break
        if impact_idx >= 0:
            back_window = np.where((t >= max(t0 + 0.1, t[impact_idx] - 1.2)) & (t <= t[impact_idx]))[0]
            onset_idx = None
            onset_reason = 'not_found'
            for i in back_window:
                if i + 1 >= len(t):
                    break
                acc_cond = np.all(acc_mag[i:i + 2] > acc_thr)
                gyro_cond = np.all(gyro_mag[i:i + 2] > gyro_thr)
                jerk_cond = np.all(jerk_mag[i:i + 2] > jerk_thr)
                if jerk_cond or acc_cond or gyro_cond:
                    onset_idx = int(i)
                    onset_reason = 'jerk_threshold' if jerk_cond else 'adaptive_acc_threshold' if acc_cond else 'adaptive_gyro_threshold'
                    break
            if onset_idx is None:
                for i in back_window:
                    if i + 4 >= len(t):
                        break
                    h_cond = np.all(heights[i:i + 5] < height_thr) if h_ref > 0 else False
                    if h_cond:
                        onset_idx = int(i)
                        onset_reason = 'height_drop_threshold'
                        break
        impact_t = float(t[impact_idx]) if impact_idx >= 0 else float(t[-1])
        if event_summary and event_summary.get('available', False):
            lead_ms = float(event_summary.get('lead_time_ms', 0.0))
            preimpact_duration = max(0.0, float(event_summary.get('impact_time', impact_t)) - float(event_summary.get('onset_time', impact_t)))
        else:
            lead_ms = 0.0 if onset_idx is None or impact_idx < 0 else max(0.0, (impact_t - float(t[onset_idx])) * 1000.0)
            preimpact_duration = 0.0 if onset_idx is None else max(0.0, impact_t - float(t[onset_idx]))
        checks = {'sampling_rate_100hz': True, 'lead_time_realistic': 200.0 <= lead_ms <= 650.0, 'preimpact_window_present': 0.25 <= preimpact_duration <= 1.0, 'low_back_signal_shape': 10.0 <= float(np.max(acc_mag)) <= 130.0}
        score = float(np.mean([1.0 if v else 0.0 for v in checks.values()]))
        return {'available': True, 'checks': checks, 'lead_time_ms': lead_ms, 'preimpact_duration_s': preimpact_duration, 'peak_accel_ms2': float(np.max(acc_mag)) if len(acc_mag) else 0.0, 'peak_gyro_rads': float(np.max(gyro_mag)) if len(gyro_mag) else 0.0, 'benchmark_lead_ms': self.benchmark_lead_ms, 'score': score, 'kfall_compliant': bool(score >= 0.75), 'onset_reason': onset_reason, 'adaptive_acc_threshold_ms2': acc_thr, 'adaptive_gyro_threshold_rads': gyro_thr, 'adaptive_jerk_threshold_ms3': jerk_thr, 'adaptive_height_threshold_m': height_thr, 'baseline_acc_mean_ms2': acc_mu, 'baseline_acc_std_ms2': acc_sd, 'baseline_gyro_mean_rads': gyro_mu, 'baseline_gyro_std_rads': gyro_sd, 'baseline_jerk_mean_ms3': jerk_mu, 'baseline_jerk_std_ms3': jerk_sd}

class DynamicsContactAnalyzer:
    """
    Layer-2 contact and dynamics extraction.

    Focuses on physically interpretable outputs without altering controller
    torques or contact parameters. All quantities are logged from the current
    simulation state and exported for validation/plots.
    """

    def __init__(self, mj_model, mj_data, body_mass, leg_length=0.93):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.body_mass = float(body_mass)
        self.leg_length = float(leg_length)
        self.g = 9.81
        self.frames = []
        self.contact_rows = []
        self._ground_geoms = self._detect_ground_geoms()
        self._left_foot_bodies, self._right_foot_bodies = self._detect_foot_bodies()
        self._ema_alpha = 0.65
        self._ema = {'total_ground_vertical_n': 0.0, 'support_vertical_n': 0.0, 'left_foot_vertical_n': 0.0, 'right_foot_vertical_n': 0.0, 'nonfoot_ground_normal_n': 0.0, 'primary_impact_body_load_n': 0.0, 'tangential_ratio': 0.0}

    def _detect_ground_geoms(self):
        plane_type = getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', None)
        ground = set()
        for g in range(self.mj_model.ngeom):
            gname = (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g) or '').lower()
            if 'floor' in gname or 'ground' in gname or 'plane' in gname or (plane_type is not None and self.mj_model.geom_type[g] == plane_type):
                ground.add(g)
        return ground

    def _detect_foot_bodies(self):
        left, right = (set(), set())
        for bid in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or ''
            norm = _norm_name(name)
            if any((k in norm for k in ('foot', 'ankle', 'toe', 'heel'))):
                if _body_name_matches_side(name, 'left'):
                    left.add(bid)
                elif _body_name_matches_side(name, 'right'):
                    right.add(bid)
        return (left, right)

    def _body_name(self, bid):
        return mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or f'body_{bid}'

    def _geom_name(self, gid):
        return mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, gid) or f'geom_{gid}'

    def _classify_contact(self, body_id):
        if body_id in self._left_foot_bodies:
            return 'left_foot'
        if body_id in self._right_foot_bodies:
            return 'right_foot'
        return 'body'

    def capture_frame(self, sim_time=None, phase=''):
        if sim_time is None:
            sim_time = float(self.mj_data.time)
        total_ground_vertical = 0.0
        total_ground_normal = 0.0
        left_vertical = 0.0
        right_vertical = 0.0
        left_normal = 0.0
        right_normal = 0.0
        nonfoot_ground_normal = 0.0
        cop_x = 0.0
        cop_y = 0.0
        support_cop_x = 0.0
        support_cop_y = 0.0
        tangential_sum = 0.0
        contact_count = 0
        impact_by_body = defaultdict(float)
        left_contact = False
        right_contact = False
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = (c.geom1, c.geom2)
            is_ground = g1 in self._ground_geoms or g2 in self._ground_geoms
            wrench_local = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench_local)
            f_world, t_world = _contact_wrench_world(c, wrench_local)
            normal_n = float(max(0.0, wrench_local[0]))
            tangential_n = float(np.linalg.norm(wrench_local[1:3]))
            if not is_ground or normal_n <= 0.5:
                continue
            contact_count += 1
            total_ground_normal += normal_n
            total_ground_vertical += max(0.0, float(f_world[2]))
            tangential_sum += tangential_n
            cop_x += float(c.pos[0]) * max(0.0, float(f_world[2]))
            cop_y += float(c.pos[1]) * max(0.0, float(f_world[2]))
            non_ground_geom = g2 if g1 in self._ground_geoms else g1
            body_id = int(self.mj_model.geom_bodyid[non_ground_geom])
            body_name = self._body_name(body_id)
            cls = self._classify_contact(body_id)
            if cls == 'left_foot':
                fz = max(0.0, float(f_world[2]))
                left_vertical += fz
                left_normal += normal_n
                support_cop_x += float(c.pos[0]) * fz
                support_cop_y += float(c.pos[1]) * fz
                left_contact = True
            elif cls == 'right_foot':
                fz = max(0.0, float(f_world[2]))
                right_vertical += fz
                right_normal += normal_n
                support_cop_x += float(c.pos[0]) * fz
                support_cop_y += float(c.pos[1]) * fz
                right_contact = True
            else:
                nonfoot_ground_normal += normal_n
                impact_by_body[body_name] += normal_n
            self.contact_rows.append({'time': float(sim_time), 'phase': str(phase), 'contact_index': int(i), 'body_name': body_name, 'geom_name': self._geom_name(non_ground_geom), 'class': cls, 'normal_n': normal_n, 'tangential_n': tangential_n, 'world_fx': float(f_world[0]), 'world_fy': float(f_world[1]), 'world_fz': float(f_world[2]), 'px': float(c.pos[0]), 'py': float(c.pos[1]), 'pz': float(c.pos[2])})
        pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        pelvis_pos = self.mj_data.xpos[pelvis_id].copy() if pelvis_id >= 0 else np.zeros(3)
        pelvis_xy = pelvis_pos[:2].copy()
        pelvis_R = self.mj_data.xmat[pelvis_id].reshape(3, 3).copy() if pelvis_id >= 0 else np.eye(3)
        cop = None
        if total_ground_vertical > 1.0:
            cop = np.array([cop_x / total_ground_vertical, cop_y / total_ground_vertical], dtype=float)
        support_cop = None
        support_vertical = left_vertical + right_vertical
        if support_vertical > 1.0:
            support_cop = np.array([support_cop_x / support_vertical, support_cop_y / support_vertical], dtype=float)
        contact_load_threshold = 0.12 * self.body_mass * self.g
        left_contact = bool(left_vertical > contact_load_threshold)
        right_contact = bool(right_vertical > contact_load_threshold)
        support_cop_rel = support_cop - pelvis_xy if support_cop is not None else None
        support_cop_body = None
        if support_cop is not None:
            diff3 = np.array([support_cop[0] - pelvis_pos[0], support_cop[1] - pelvis_pos[1], 0.0 - pelvis_pos[2]], dtype=float)
            support_cop_body = (pelvis_R.T @ diff3)[:2].copy()
        if total_ground_normal > 0.05 * self.body_mass * self.g:
            tangential_ratio = tangential_sum / max(total_ground_normal, 1e-09)
        else:
            tangential_ratio = 0.0
        primary_body = max(impact_by_body.items(), key=lambda kv: kv[1])[0] if impact_by_body else ''
        primary_body_load = max(impact_by_body.values()) if impact_by_body else 0.0
        raw_vals = {'total_ground_vertical_n': float(total_ground_vertical), 'support_vertical_n': float(support_vertical), 'left_foot_vertical_n': float(left_vertical), 'right_foot_vertical_n': float(right_vertical), 'nonfoot_ground_normal_n': float(nonfoot_ground_normal), 'primary_impact_body_load_n': float(primary_body_load), 'tangential_ratio': float(tangential_ratio)}
        filt = {}
        a = float(self._ema_alpha)
        for k, v in raw_vals.items():
            prev = float(self._ema.get(k, v))
            val = a * prev + (1.0 - a) * float(v)
            self._ema[k] = val
            filt[k + '_filt'] = float(val)
        frame = {'time': float(sim_time), 'phase': str(phase), 'contact_count': int(contact_count), 'total_ground_vertical_n': float(total_ground_vertical), 'total_ground_normal_n': float(total_ground_normal), 'support_vertical_n': float(support_vertical), 'left_foot_vertical_n': float(left_vertical), 'right_foot_vertical_n': float(right_vertical), 'left_foot_normal_n': float(left_normal), 'right_foot_normal_n': float(right_normal), 'nonfoot_ground_normal_n': float(nonfoot_ground_normal), 'pelvis_xy': pelvis_xy.copy(), 'cop_xy': cop.copy() if cop is not None else None, 'support_cop_xy': support_cop.copy() if support_cop is not None else None, 'support_cop_rel_xy': support_cop_rel.copy() if support_cop_rel is not None else None, 'support_cop_body_xy': support_cop_body.copy() if support_cop_body is not None else None, 'double_support': bool(left_contact and right_contact), 'left_support': bool(left_contact), 'right_support': bool(right_contact), 'tangential_ratio': float(tangential_ratio), 'primary_impact_body': primary_body, 'primary_impact_body_load_n': float(primary_body_load), 'impact_by_body': dict(impact_by_body), **filt}
        self.frames.append(frame)
        return frame

    def latest(self):
        return self.frames[-1] if self.frames else None

    def export_frame_csv(self, filename):
        import csv
        if not self.frames:
            return {'filename': filename, 'frames': 0}
        keys = ['time', 'phase', 'contact_count', 'total_ground_vertical_n', 'total_ground_normal_n', 'support_vertical_n', 'left_foot_vertical_n', 'right_foot_vertical_n', 'left_foot_normal_n', 'right_foot_normal_n', 'nonfoot_ground_normal_n', 'tangential_ratio', 'double_support', 'left_support', 'right_support', 'primary_impact_body', 'primary_impact_body_load_n', 'total_ground_vertical_n_filt', 'support_vertical_n_filt', 'left_foot_vertical_n_filt', 'right_foot_vertical_n_filt', 'nonfoot_ground_normal_n_filt', 'primary_impact_body_load_n_filt', 'tangential_ratio_filt']
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys + ['cop_x', 'cop_y', 'support_cop_x', 'support_cop_y', 'support_cop_rel_x', 'support_cop_rel_y', 'support_cop_body_x', 'support_cop_body_y', 'pelvis_x', 'pelvis_y'])
            for fr in self.frames:
                cop = fr['cop_xy'] if fr['cop_xy'] is not None else [np.nan, np.nan]
                scop = fr['support_cop_xy'] if fr.get('support_cop_xy') is not None else [np.nan, np.nan]
                scop_rel = fr['support_cop_rel_xy'] if fr.get('support_cop_rel_xy') is not None else [np.nan, np.nan]
                scop_body = fr['support_cop_body_xy'] if fr.get('support_cop_body_xy') is not None else [np.nan, np.nan]
                pel = fr.get('pelvis_xy', [np.nan, np.nan])
                writer.writerow([fr[k] for k in keys] + [float(cop[0]), float(cop[1]), float(scop[0]), float(scop[1]), float(scop_rel[0]), float(scop_rel[1]), float(scop_body[0]), float(scop_body[1]), float(pel[0]), float(pel[1])])
        return {'filename': filename, 'frames': len(self.frames)}

    def export_contact_csv(self, filename):
        import csv
        if not self.contact_rows:
            return {'filename': filename, 'rows': 0}
        keys = list(self.contact_rows[0].keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.contact_rows)
        return {'filename': filename, 'rows': len(self.contact_rows)}

    def _median_filter(self, arr, k=5):
        arr = np.asarray(arr, dtype=float)
        if k <= 1 or arr.size == 0:
            return arr.copy()
        h = k // 2
        out = np.empty_like(arr)
        for i in range(arr.size):
            lo = max(0, i - h)
            hi = min(arr.size, i + h + 1)
            out[i] = np.nanmedian(arr[lo:hi])
        return out

    def export_visuals(self, prefix):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            return {'available': False, 'plots': [], 'error': str(e)}
        if not self.frames:
            return {'available': False, 'plots': []}
        t = np.array([fr['time'] for fr in self.frames], dtype=float)
        bw = self.body_mass * self.g
        total_v = np.array([fr.get('support_vertical_n_filt', fr['total_ground_vertical_n_filt']) for fr in self.frames], dtype=float) / max(bw, 1.0)
        left_v = np.array([fr['left_foot_vertical_n_filt'] for fr in self.frames], dtype=float) / max(bw, 1.0)
        right_v = np.array([fr['right_foot_vertical_n_filt'] for fr in self.frames], dtype=float) / max(bw, 1.0)
        impact = np.array([fr['primary_impact_body_load_n_filt'] for fr in self.frames], dtype=float) / max(bw, 1.0)
        tan = np.array([fr['tangential_ratio_filt'] for fr in self.frames], dtype=float)
        total_v = self._median_filter(total_v, 5)
        left_v = self._median_filter(left_v, 5)
        right_v = self._median_filter(right_v, 5)
        impact = self._median_filter(impact, 5)
        tan = self._median_filter(tan, 5)
        fig = plt.figure(figsize=(10.5, 6.2))
        ax1 = fig.add_subplot(211)
        ax1.plot(t, total_v, label='Support vertical GRF/BW (median)', linewidth=2.0)
        ax1.plot(t, left_v, label='Left foot GRF/BW (median)', linewidth=1.7)
        ax1.plot(t, right_v, label='Right foot GRF/BW (median)', linewidth=1.7)
        ax1.plot(t, impact, label='Primary impact/BW (median)', linewidth=1.7)
        ax1.set_ylabel('Load / BW')
        ymax = max(3.0, float(np.nanpercentile(np.r_[total_v, left_v, right_v, impact], 99.0)) * 1.15)
        ax1.set_ylim(0.0, ymax)
        ax1.set_title('Layer-2 dynamics summary')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax2 = fig.add_subplot(212)
        ax2.plot(t, tan, label='Tangential/normal ratio (median)', linewidth=2.0)
        ax2.set_ylim(0.0, max(1.5, float(np.nanpercentile(tan, 99.0)) * 1.15))
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        fn = f'{prefix}_dynamics_summary.png'
        fig.tight_layout()
        fig.savefig(fn, dpi=180)
        plt.close(fig)
        cops = np.array([fr['support_cop_body_xy'] if fr.get('support_cop_body_xy') is not None else fr['support_cop_rel_xy'] if fr.get('support_cop_rel_xy') is not None else [np.nan, np.nan] for fr in self.frames], dtype=float)
        fig = plt.figure(figsize=(6.2, 6.0))
        ax = fig.add_subplot(111)
        valid = np.isfinite(cops[:, 0]) & np.isfinite(cops[:, 1])
        if np.any(valid):
            phases = np.array([str(fr.get('phase', '')) for fr in self.frames])
            for ph, color in [('stand', '#1f77b4'), ('walk', '#2ca02c'), ('perturb', '#ff7f0e'), ('react', '#d62728'), ('fall', '#9467bd')]:
                mask = valid & (phases == ph)
                if np.any(mask):
                    ax.plot(cops[mask, 0], cops[mask, 1], '.', color=color, alpha=0.65, label=ph, markersize=4)
        ax.axhline(0.0, color='0.7', linewidth=0.8)
        ax.axvline(0.0, color='0.7', linewidth=0.8)
        ax.set_title('Support CoP in pelvis body frame (phase-coloured)')
        ax.set_xlabel('Forward rel. pelvis [m]')
        ax.set_ylabel('Lateral rel. pelvis [m]')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        fn2 = f'{prefix}_cop_path.png'
        fig.tight_layout()
        fig.savefig(fn2, dpi=180)
        plt.close(fig)
        return {'available': True, 'plots': [fn, fn2]}

    def export_bundle(self, prefix):
        frame_csv = self.export_frame_csv(f'{prefix}_dynamics_frames.csv')
        contact_csv = self.export_contact_csv(f'{prefix}_contacts.csv')
        visuals = self.export_visuals(prefix)
        return {'frame_csv': frame_csv, 'contact_csv': contact_csv, 'visuals': visuals}

class PaperAlignmentExporter:
    """
    High-level bridge for the paper-oriented deliverables that are still missing
    from the base MuJoCo/Meta-Motivo rollout.

    It does three things without requiring OpenSim/AnyBody to be installed:
      1) exports foot-wise GRF/CoP/torque in an OpenSim-style .mot file
      2) exports an ExternalLoads XML template plus marker-registration metadata
      3) exports a multiview 2D pose dataset derived from the simulated 3D markers

    This closes the "export-compatible" gap in a much more paper-faithful way.
    """

    def __init__(self, marker_exporter, dynamics_analyzer, subject_meta=None):
        self.marker_exporter = marker_exporter
        self.dynamics_analyzer = dynamics_analyzer
        self.subject_meta = dict(subject_meta or {})

    def _master_times(self):
        if self.dynamics_analyzer.frames:
            return np.array([float(fr['time']) for fr in self.dynamics_analyzer.frames], dtype=float)
        if self.marker_exporter.frames:
            return np.array([float(fr['time']) for fr in self.marker_exporter.frames], dtype=float)
        return np.array([], dtype=float)

    def _time_key(self, t):
        return round(float(t), 6)

    def _aggregate_external_loads(self):
        master_t = self._master_times()
        if master_t.size == 0:
            return []
        buckets = {self._time_key(t): {'left_foot': {'F': np.zeros(3), 'cop_num': np.zeros(3), 'fz_sum': 0.0, 'contacts': []}, 'right_foot': {'F': np.zeros(3), 'cop_num': np.zeros(3), 'fz_sum': 0.0, 'contacts': []}} for t in master_t}
        for row in self.dynamics_analyzer.contact_rows:
            cls = str(row.get('class', ''))
            if cls not in ('left_foot', 'right_foot'):
                continue
            tk = self._time_key(row['time'])
            if tk not in buckets:
                continue
            force = np.array([float(row['world_fx']), float(row['world_fy']), float(row['world_fz'])], dtype=float)
            pos = np.array([float(row['px']), float(row['py']), float(row['pz'])], dtype=float)
            rec = buckets[tk][cls]
            rec['F'] += force
            fz = max(0.0, float(force[2]))
            rec['cop_num'] += pos * fz
            rec['fz_sum'] += fz
            rec['contacts'].append((pos, force))
        out = []
        for t in master_t:
            tk = self._time_key(t)
            row_out = {'time': float(t)}
            for cls, side in (('left_foot', 'l'), ('right_foot', 'r')):
                rec = buckets[tk][cls]
                F = rec['F']
                if rec['fz_sum'] > 1e-09:
                    cop = rec['cop_num'] / rec['fz_sum']
                else:
                    cop = np.array([0.0, 0.0, 0.0], dtype=float)
                torque = np.zeros(3)
                for pos, force in rec['contacts']:
                    torque += np.cross(pos - cop, force)
                row_out[f'F_{side}'] = F.copy()
                row_out[f'cop_{side}'] = cop.copy()
                row_out[f'T_{side}'] = torque.copy()
            out.append(row_out)
        return out

    def export_opensim_grf_mot(self, filename):
        rows = self._aggregate_external_loads()
        if not rows:
            return {'filename': filename, 'frames': 0}
        cols = ['time', 'ground_force_l_vx', 'ground_force_l_vy', 'ground_force_l_vz', 'ground_force_l_px', 'ground_force_l_py', 'ground_force_l_pz', 'ground_torque_l_x', 'ground_torque_l_y', 'ground_torque_l_z', 'ground_force_r_vx', 'ground_force_r_vy', 'ground_force_r_vz', 'ground_force_r_px', 'ground_force_r_py', 'ground_force_r_pz', 'ground_torque_r_x', 'ground_torque_r_y', 'ground_torque_r_z']
        with open(filename, 'w') as f:
            f.write(f'name {Path(filename).name}\n')
            f.write(f'datacolumns {len(cols)}\n')
            f.write(f'datarows {len(rows)}\n')
            f.write(f"range {rows[0]['time']:.6f} {rows[-1]['time']:.6f}\n")
            f.write('endheader\n')
            f.write('\t'.join(cols) + '\n')
            for row in rows:
                vals = [row['time'], *row['F_l'], *row['cop_l'], *row['T_l'], *row['F_r'], *row['cop_r'], *row['T_r']]
                f.write('\t'.join((f'{float(v):.6f}' for v in vals)) + '\n')
        return {'filename': filename, 'frames': len(rows), 'columns': len(cols)}

    def export_opensim_external_loads_xml(self, filename, grf_mot_file):
        import xml.etree.ElementTree as ET
        root = ET.Element('OpenSimDocument', Version='40000')
        ext = ET.SubElement(root, 'ExternalLoads', name='external_loads')
        ET.SubElement(ext, 'objects')
        objs = ext.find('objects')
        for side, body in (('l', 'calcn_l'), ('r', 'calcn_r')):
            ef = ET.SubElement(objs, 'ExternalForce', name=f'grf_{side}')
            ET.SubElement(ef, 'applied_to_body').text = body
            ET.SubElement(ef, 'force_expressed_in_body').text = 'ground'
            ET.SubElement(ef, 'point_expressed_in_body').text = 'ground'
            ET.SubElement(ef, 'force_identifier').text = f'ground_force_{side}_v'
            ET.SubElement(ef, 'point_identifier').text = f'ground_force_{side}_p'
            ET.SubElement(ef, 'torque_identifier').text = f'ground_torque_{side}_'
        ET.SubElement(ext, 'groups')
        ET.SubElement(ext, 'datafile').text = Path(grf_mot_file).name
        ET.SubElement(ext, 'external_loads_model_kinematics_file').text = ''
        ET.SubElement(ext, 'lowpass_cutoff_frequency_for_load_kinematics').text = '-1'
        tree = ET.ElementTree(root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        return {'filename': filename, 'datafile': Path(grf_mot_file).name, 'note': 'applied_to_body defaults to calcn_l/calcn_r and may need adaptation to the target OpenSim model'}

    def export_marker_registration_json(self, filename):
        import json
        recs = []
        for marker_name, rec in self.marker_exporter.marker_defs.items():
            recs.append({'avatar_marker': marker_name, 'opensim_marker': self.marker_exporter.OPENSIM_TRC_NAMES.get(marker_name, marker_name), 'resolved_body': rec['body_name'], 'resolved_body_id': int(rec['body_id']), 'offset_local_m': [float(x) for x in np.asarray(rec['offset_local'], dtype=float)], 'offset_mode': rec.get('offset_mode', 'center'), 'family': rec.get('family'), 'side': rec.get('side')})
        payload = {'subject_meta': self.subject_meta, 'num_markers': len(recs), 'marker_registration': recs, 'trc_names': self.marker_exporter.OPENSIM_TRC_NAMES, 'note': 'This is the avatar-to-musculoskeletal bridge metadata. It is intended to support IK marker registration in OpenSim/AnyBody workflows.'}
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
        return {'filename': filename, 'markers': len(recs)}

    def _camera_specs(self):
        all_pts = []
        for fr in self.marker_exporter.frames:
            for name in self.marker_exporter.POSE_MARKERS:
                if name in fr['markers']:
                    all_pts.append(np.asarray(fr['markers'][name], dtype=float))
        if not all_pts:
            return {}
        arr = np.vstack(all_pts)
        mins = np.min(arr, axis=0)
        maxs = np.max(arr, axis=0)
        center = 0.5 * (mins + maxs)
        span = np.maximum(maxs - mins, np.array([0.4, 0.4, 1.0]))
        radius = float(max(span[0], span[1], span[2]))
        target = center + np.array([0.0, 0.0, 0.55 * span[2]])
        return {'frontal': {'pos': center + np.array([0.0, -2.8 * radius, 0.7 * radius]), 'target': target}, 'sagittal': {'pos': center + np.array([2.8 * radius, 0.0, 0.7 * radius]), 'target': target}, 'oblique': {'pos': center + np.array([2.2 * radius, -2.2 * radius, 0.8 * radius]), 'target': target}}

    def _look_at(self, pos, target, up=np.array([0.0, 0.0, 1.0])):
        pos = np.asarray(pos, dtype=float)
        target = np.asarray(target, dtype=float)
        up = np.asarray(up, dtype=float)
        z = target - pos
        z /= max(np.linalg.norm(z), 1e-09)
        x = np.cross(z, up)
        if np.linalg.norm(x) < 1e-09:
            x = np.array([1.0, 0.0, 0.0], dtype=float)
        x /= max(np.linalg.norm(x), 1e-09)
        y = np.cross(x, z)
        y /= max(np.linalg.norm(y), 1e-09)
        R = np.vstack([x, y, z])
        return R

    def _project(self, pts_world, cam, image_wh=(1280, 720)):
        w, h = image_wh
        fx = fy = 0.95 * min(w, h)
        cx, cy = (0.5 * w, 0.5 * h)
        R = self._look_at(cam['pos'], cam['target'])
        t = np.asarray(cam['pos'], dtype=float)
        pts2d = []
        vis = []
        for p in pts_world:
            p = np.asarray(p, dtype=float)
            pc = R @ (p - t)
            if pc[2] <= 1e-06:
                pts2d.append([np.nan, np.nan])
                vis.append(0)
                continue
            u = fx * (pc[0] / pc[2]) + cx
            v = cy - fy * (pc[1] / pc[2])
            in_frame = int(-0.1 * w <= u <= 1.1 * w and -0.1 * h <= v <= 1.1 * h)
            pts2d.append([float(u), float(v)])
            vis.append(in_frame)
        return (np.array(pts2d, dtype=float), np.array(vis, dtype=int))

    def export_pose_dataset_json(self, filename, views=('frontal', 'sagittal', 'oblique')):
        import json
        cameras = self._camera_specs()
        if not cameras or not self.marker_exporter.frames:
            return {'filename': filename, 'frames': 0}
        views = [v for v in views if v in cameras]
        records = []
        marker_names = [m for m in self.marker_exporter.POSE_MARKERS if m in self.marker_exporter.marker_defs]
        for idx, fr in enumerate(self.marker_exporter.frames):
            pts_world = [np.asarray(fr['markers'][m], dtype=float) for m in marker_names]
            pts3d = [[float(x) for x in p] for p in pts_world]
            view_rows = []
            for v in views:
                pts2d, vis = self._project(pts_world, cameras[v])
                valid = np.isfinite(pts2d[:, 0]) & np.isfinite(pts2d[:, 1]) & (vis > 0)
                if np.any(valid):
                    uv = pts2d[valid]
                    x0, y0 = np.min(uv, axis=0)
                    x1, y1 = np.max(uv, axis=0)
                    bbox = [float(x0), float(y0), float(max(0.0, x1 - x0)), float(max(0.0, y1 - y0))]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                view_rows.append({'camera': v, 'bbox_xywh': bbox, 'keypoints_xyv': [[float(pts2d[k, 0]) if np.isfinite(pts2d[k, 0]) else None, float(pts2d[k, 1]) if np.isfinite(pts2d[k, 1]) else None, int(vis[k])] for k in range(len(marker_names))]})
            records.append({'frame_index': int(idx), 'time': float(fr['time']), 'phase': str(fr.get('phase', '')), 'marker_names': marker_names, 'keypoints_xyz_m': pts3d, 'views': view_rows})
        payload = {'subject_meta': self.subject_meta, 'marker_names': marker_names, 'skeleton_edges': self.marker_exporter.VISUAL_SKELETON_EDGES, 'cameras': {k: {'position_world_m': [float(x) for x in v['pos']], 'target_world_m': [float(x) for x in v['target']], 'image_size': [1280, 720]} for k, v in cameras.items() if k in views}, 'frames': records, 'note': 'Synthetic multiview 2D/3D pose annotations derived from the simulated anatomical markers. These are render-free annotations; photo-real RGB generation requires an external renderer.'}
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
        return {'filename': filename, 'frames': len(records), 'views': len(views)}

    def export_pose_preview(self, filename, views=('frontal', 'sagittal', 'oblique'), num_frames=4):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            return {'filename': filename, 'available': False, 'error': str(e)}
        cameras = self._camera_specs()
        if not cameras or not self.marker_exporter.frames:
            return {'filename': filename, 'available': False}
        views = [v for v in views if v in cameras]
        idxs = np.linspace(0, len(self.marker_exporter.frames) - 1, num=max(1, num_frames), dtype=int)
        marker_names = [m for m in self.marker_exporter.VISUAL_MARKERS if m in self.marker_exporter.marker_defs]
        fig = plt.figure(figsize=(4.2 * len(views), 3.4 * len(idxs)))
        for r, idx in enumerate(idxs, start=1):
            fr = self.marker_exporter.frames[int(idx)]
            pts_world = {m: np.asarray(fr['markers'][m], dtype=float) for m in marker_names}
            for c, view in enumerate(views, start=1):
                ax = fig.add_subplot(len(idxs), len(views), (r - 1) * len(views) + c)
                cam = cameras[view]
                proj = {}
                pts_list = []
                for m, p in pts_world.items():
                    uv, vis = self._project([p], cam)
                    if vis[0] > 0 and np.isfinite(uv[0, 0]) and np.isfinite(uv[0, 1]):
                        proj[m] = uv[0].copy()
                        pts_list.append(uv[0].copy())
                for a, b in self.marker_exporter.VISUAL_SKELETON_EDGES:
                    if a in proj and b in proj:
                        pa, pb = (proj[a], proj[b])
                        style = self.marker_exporter._edge_style(a, b)
                        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], '-', color=style['color'], linewidth=style['linewidth'])
                if pts_list:
                    arr = np.vstack(pts_list)
                    ax.scatter(arr[:, 0], arr[:, 1], s=18, color='#1f77b4', edgecolors='white', linewidths=0.4, zorder=3)
                    x0, y0 = np.min(arr, axis=0)
                    x1, y1 = np.max(arr, axis=0)
                    pad = 60.0
                    ax.set_xlim(x0 - pad, x1 + pad)
                    ax.set_ylim(y1 + pad, y0 - pad)
                ax.set_title(f"{view} | t={fr['time']:.2f}s")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal', adjustable='box')
        fig.suptitle('Paper-alignment synthetic pose preview')
        fig.tight_layout()
        fig.savefig(filename, dpi=180)
        plt.close(fig)
        return {'filename': filename, 'available': True, 'frames': len(idxs), 'views': len(views)}

    def export_bundle(self, prefix, views=('frontal', 'sagittal', 'oblique')):
        grf = self.export_opensim_grf_mot(f'{prefix}_opensim_grf.mot')
        ext = self.export_opensim_external_loads_xml(f'{prefix}_ExternalLoads.xml', grf['filename']) if grf.get('frames', 0) else {'filename': f'{prefix}_ExternalLoads.xml', 'frames': 0}
        reg = self.export_marker_registration_json(f'{prefix}_marker_registration.json')
        pose_json = self.export_pose_dataset_json(f'{prefix}_synthetic_pose_dataset.json', views=views)
        pose_preview = self.export_pose_preview(f'{prefix}_synthetic_pose_preview.png', views=views, num_frames=PAPER_ALIGNMENT_EXPORT.get('pose_preview_frames', 4))
        return {'opensim_grf': grf, 'external_loads_xml': ext, 'marker_registration': reg, 'pose_dataset_json': pose_json, 'pose_preview': pose_preview}

class PhysicsDashboard:

    def __init__(self, mj_model, mj_data, body_mass, leg_length, upright_h):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.body_mass = body_mass
        self.leg_length = leg_length
        self.upright_h = upright_h
        self.g = 9.81
        self.ip_omega = float(np.sqrt(self.g / max(leg_length, 0.1)))
        self._pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'Pelvis')
        self._torso_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'Torso')
        self._foot_l_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'FootL')
        self._foot_r_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'FootR')
        self._hand_l_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'HandL')
        self._hand_r_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'HandR')
        self._lower_limb_kw = ['foot', 'ankle', 'toe', 'heel', 'lower', 'shin', 'tibia']
        self._left_kw = ['l', 'left']
        self._right_kw = ['r', 'right']
        plane_type = getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', None)
        self._ground_geoms = set()
        for g in range(mj_model.ngeom):
            gname = (mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g) or '').lower()
            is_ground = 'floor' in gname or 'ground' in gname or 'plane' in gname or (plane_type is not None and mj_model.geom_type[g] == plane_type)
            if is_ground:
                self._ground_geoms.add(g)
        self._left_lower_bodies = set()
        self._right_lower_bodies = set()
        for i in range(mj_model.nbody):
            n = mujoco.mj_id2name(mj_model, MJOBJ_BODY, i) or ''
            nlow = n.lower()
            if any((k in nlow for k in self._lower_limb_kw)):
                if _body_name_matches_side(n, 'left'):
                    self._left_lower_bodies.add(i)
                elif _body_name_matches_side(n, 'right'):
                    self._right_lower_bodies.add(i)
        self._shoulder_dof_ids = []
        for i in range(mj_model.nu):
            n = (mujoco.mj_id2name(mj_model, MJOBJ_ACTUATOR, i) or '').lower()
            if 'shoulder' in n:
                self._shoulder_dof_ids.append(i)
        self._header_printed = False
        self._foot_contact_history = deque(maxlen=60)
        self._cadence_hz = 0.0

    def _print_header(self):
        print('\n' + '=' * 170)
        print(f"  {'step':>5} {'phase':>8} {'h[m]':>6} {'vx':>6} {'trunkdeg':>7} {'armSwdeg':>7} {'dbl_sup':>7} {'GRF/BW':>7} {'XCoM_m':>7} {'Fr':>6} {'omg[dps]':>8} {'Imp/BW':>7} {'slip':>6} {'ncon':>5} {'settle':>6} {'src':>4} {'mpcC':>7} {'E_fall[J]':>10} {'JRMS[deg]':>8} {'strength':>9} {'IMU_pk':>8} {'fall?':>5}")
        print('-' * 205)
        self._header_printed = True

    def _pelvis_world_vel(self):
        vel6 = np.zeros(6)
        try:
            mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, self._pelvis_id, vel6, 0)
            return vel6[3:].copy()
        except Exception:
            R = self.mj_data.xmat[self._pelvis_id].reshape(3, 3)
            return R @ self.mj_data.cvel[self._pelvis_id][3:]

    def _trunk_lean_deg(self):
        """
        FIX X2: Trunk lean = angle between pelvis->Head world vector and world-z.
        0deg = perfectly upright. 90deg = horizontal (fallen).
        Replaces v12 method which used Torso local-z (gave ~90deg when upright
        because Torso local-z points FORWARD in humanenv, not UP).
        """
        head_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Head')
        if head_id < 0 or self._pelvis_id < 0:
            return 0.0
        pelvis_pos = self.mj_data.xpos[self._pelvis_id]
        head_pos = self.mj_data.xpos[head_id]
        vec = head_pos - pelvis_pos
        norm = float(np.linalg.norm(vec))
        if norm < 1e-06:
            return 0.0
        cos_theta = float(np.clip(np.dot(vec / norm, [0.0, 0.0, 1.0]), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    def _arm_swing_deg(self):
        """
        FIX X3: Arm swing from ANGULAR VELOCITY of shoulder actuators (qvel).
        Heading-invariant. Returns RMS of shoulder angular velocities in deg/s
        divided by a scaling factor to give approximate peak angle in deg.
        Norm: 8-20deg shoulder flexion/extension during normal walking.
        """
        if not self._shoulder_dof_ids:
            return 0.0
        shoulder_speeds = []
        for act_id in self._shoulder_dof_ids:
            jnt_id = self.mj_model.actuator_trnid[act_id, 0]
            if 0 <= jnt_id < self.mj_model.njnt:
                dof_id = self.mj_model.jnt_dofadr[jnt_id]
                if 0 <= dof_id < len(self.mj_data.qvel):
                    shoulder_speeds.append(abs(float(self.mj_data.qvel[dof_id])))
        if not shoulder_speeds:
            return 0.0
        rms_speed = float(np.sqrt(np.mean(np.array(shoulder_speeds) ** 2)))
        approx_angle = float(np.degrees(rms_speed) / (2 * np.pi * 1.0))
        return float(np.clip(approx_angle, 0.0, 90.0))

    def _double_support(self):
        """
        FIX X7: Double-support via actual ground contacts on lower-limb bodies.
        v12 used foot body z-position which didn't match humanenv's body names.
        Now checks contact geom body IDs against left/right lower-limb body sets.
        Falls back to z-position check if no contact body IDs matched.
        """
        left_contact = False
        right_contact = False
        for i in range(self.mj_data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, force)
            if force[0] < 0.5:
                continue
            g1 = self.mj_data.contact[i].geom1
            g2 = self.mj_data.contact[i].geom2
            if self._ground_geoms and g1 not in self._ground_geoms and (g2 not in self._ground_geoms):
                continue
            b1 = self.mj_model.geom_bodyid[g1]
            b2 = self.mj_model.geom_bodyid[g2]
            for b in (b1, b2):
                if b in self._left_lower_bodies:
                    left_contact = True
                if b in self._right_lower_bodies:
                    right_contact = True
        if not self._left_lower_bodies and (not self._right_lower_bodies):
            thresh = 0.08
            fl_id = self._foot_l_id
            fr_id = self._foot_r_id
            if fl_id >= 0 and fr_id >= 0:
                left_contact = float(self.mj_data.xpos[fl_id][2]) < thresh
                right_contact = float(self.mj_data.xpos[fr_id][2]) < thresh
        return left_contact and right_contact

    def _grf_bw(self):
        """World-vertical ground reaction force normalised to body weight."""
        total_vertical = 0.0
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1 = c.geom1
            g2 = c.geom2
            if self._ground_geoms and g1 not in self._ground_geoms and (g2 not in self._ground_geoms):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            f_world, _ = _contact_wrench_world(c, wrench)
            total_vertical += max(0.0, float(f_world[2]))
        bw = self.body_mass * self.g
        return float(total_vertical / max(bw, 1.0))

    def _pelvis_angular_speed_dps(self):
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, self._pelvis_id, vel6, 0)
        return float(np.degrees(np.linalg.norm(vel6[:3])))

    def _froude_number(self, pelvis_vel_world):
        vxy = float(np.linalg.norm(pelvis_vel_world[:2]))
        return float(vxy * vxy / max(self.g * self.leg_length, 1e-09))

    def report(self, step, phase, leg_strength, xcom_margin=None, fall_predicted=False, imu_peak=0.0, sensor_impact=0.0, control_source='-', mjpc_cost=np.nan, dynamics=None):
        if not self._header_printed:
            self._print_header()
        pelvis_pos = self.mj_data.xpos[self._pelvis_id]
        pelvis_vel_world = self._pelvis_world_vel()
        h = float(pelvis_pos[2])
        delta_h = self.upright_h - h
        v_trans = float(np.linalg.norm(pelvis_vel_world))
        E_pot = self.body_mass * self.g * max(0.0, delta_h)
        E_kin = 0.5 * self.body_mass * v_trans ** 2
        E_fall = E_pot + E_kin
        jrms_rad = float(np.sqrt(np.mean(self.mj_data.qpos[7:] ** 2)))
        jrms_deg = np.degrees(jrms_rad)
        trunk_deg = self._trunk_lean_deg()
        arm_swing = self._arm_swing_deg()
        dbl_sup = dynamics.get('double_support') if dynamics is not None else self._double_support()
        if dynamics is not None:
            if phase in ('stand', 'walk', 'perturb', 'react'):
                grf_bw = float(dynamics.get('support_vertical_n_filt', dynamics.get('support_vertical_n', 0.0)) / max(self.body_mass * self.g, 1.0))
            else:
                grf_bw = float(dynamics.get('total_ground_vertical_n_filt', dynamics.get('total_ground_vertical_n', 0.0)) / max(self.body_mass * self.g, 1.0))
        else:
            grf_bw = self._grf_bw()
        xcom_str = f'{xcom_margin:+7.3f}' if xcom_margin is not None else '   N/A '
        bar_len = int(leg_strength * 10)
        bar = chr(9608) * bar_len + chr(9617) * (10 - bar_len)
        fall_flag = 'FALL!' if fall_predicted else '  ok '
        slip_vxy = float(np.linalg.norm(pelvis_vel_world[:2]))
        froude = self._froude_number(pelvis_vel_world)
        omg_dps = self._pelvis_angular_speed_dps()
        impact_n = float(dynamics.get('primary_impact_body_load_n_filt', dynamics.get('primary_impact_body_load_n', sensor_impact))) if dynamics is not None else float(sensor_impact)
        impact_bw = float(impact_n / max(self.body_mass * self.g, 1.0))
        ncon = int(dynamics.get('contact_count', self.mj_data.ncon)) if dynamics is not None else int(self.mj_data.ncon)
        settle_flag = 'YES' if h < 0.18 and ncon > 0 and (slip_vxy < 0.1) else ' no'
        src = 'MPC' if str(control_source).lower().startswith('mjpc') else 'GDN'
        mpc_cost_str = f'{mjpc_cost:7.1f}' if np.isfinite(mjpc_cost) else '   N/A '
        print(f"  {step:>5d} {phase:>8s} {h:>6.3f} {float(pelvis_vel_world[0]):>6.2f} {trunk_deg:>7.2f} {arm_swing:>7.2f} {('YES' if dbl_sup else ' no'):>7} {grf_bw:>7.2f} {xcom_str:>7} {froude:>6.2f} {omg_dps:>8.1f} {impact_bw:>7.2f} {slip_vxy:>6.2f} {ncon:>5d} {settle_flag:>6} {src:>4} {mpc_cost_str:>7} {E_fall:>10.1f} {jrms_deg:>8.2f}  {bar}{leg_strength:>5.0%}  {imu_peak:>8.2f}  {fall_flag}")
        return {'pelvis_h': h, 'E_fall': E_fall, 'jrms_deg': float(jrms_deg), 'trunk_lean_deg': trunk_deg, 'arm_swing_deg': arm_swing, 'double_support': dbl_sup, 'grf_bw': grf_bw, 'imu_peak': imu_peak, 'leg_strength': leg_strength, 'slip_vxy': slip_vxy, 'froude': froude, 'pelvis_angular_speed_dps': omg_dps, 'impact_bw': impact_bw, 'contact_count': ncon, 'settle_flag': settle_flag.strip() == 'YES', 'control_source': control_source, 'mjpc_cost': float(mjpc_cost) if np.isfinite(mjpc_cost) else np.nan}

    def _compute_cop(self):
        total_fz, cop_x, cop_y = (0.0, 0.0, 0.0)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            f_world, _ = _contact_wrench_world(c, wrench)
            fz = max(0.0, float(f_world[2]))
            if fz > 1.0:
                cx = self.mj_data.contact[i].pos[0]
                cy = self.mj_data.contact[i].pos[1]
                total_fz += fz
                cop_x += cx * fz
                cop_y += cy * fz
        if total_fz > 1.0:
            return (cop_x / total_fz, cop_y / total_fz)
        return (0.0, 0.0)

    def print_phase_summary(self, phase, metrics_slice):
        if not metrics_slice:
            return
        heights = [m['pelvis_h'] for m in metrics_slice]
        e_falls = [m.get('E_fall', 0) for m in metrics_slice]
        jrms = [m.get('jrms_deg', 0) for m in metrics_slice]
        imu_pks = [m.get('imu_peak', 0) for m in metrics_slice]
        trunk_ang = [m.get('trunk_lean_deg', 0) for m in metrics_slice]
        arm_sw = [m.get('arm_swing_deg', 0) for m in metrics_slice]
        grfs = [m.get('grf_bw', 0) for m in metrics_slice]
        froudes = [m.get('froude', 0) for m in metrics_slice]
        omg_dps = [m.get('pelvis_angular_speed_dps', 0) for m in metrics_slice]
        imp_bw = [m.get('impact_bw', 0) for m in metrics_slice]
        dbl_frac = float(sum((1 for m in metrics_slice if m.get('double_support', False)))) / max(len(metrics_slice), 1)
        print(f"\n  -- Phase '{phase}' summary --")
        print(f'     pelvis h    : min={min(heights):.3f}m  max={max(heights):.3f}m  mean={np.mean(heights):.3f}m')
        print(f'     E_fall      : min={min(e_falls):.1f}J  max={max(e_falls):.1f}J  mean={np.mean(e_falls):.1f}J')
        print(f'     JRMS        : min={min(jrms):.2f}deg  max={max(jrms):.2f}deg  mean={np.mean(jrms):.2f}deg')
        print(f'     trunk lean  : min={min(trunk_ang):.2f}deg  max={max(trunk_ang):.2f}deg  mean={np.mean(trunk_ang):.2f}deg  [norm: <15deg upright, >20deg falling]')
        print(f'     arm swing   : min={min(arm_sw):.2f}deg  max={max(arm_sw):.2f}deg  mean={np.mean(arm_sw):.2f}deg  [norm: 8-20deg for walking]')
        print(f'     GRF/BW      : min={min(grfs):.2f}  max={max(grfs):.2f}  mean={np.mean(grfs):.2f}  [norm: 1.0-1.3 walking, >3 impact]')
        print(f'     Froude no.  : min={min(froudes):.2f}  max={max(froudes):.2f}  mean={np.mean(froudes):.2f}  [walk norm roughly 0.10-0.40]')
        print(f'     pelvis omg  : min={min(omg_dps):.1f}  max={max(omg_dps):.1f}  mean={np.mean(omg_dps):.1f} deg/s')
        print(f'     impact/BW   : min={min(imp_bw):.2f}  max={max(imp_bw):.2f}  mean={np.mean(imp_bw):.2f}')
        print(f'     dbl_support : {dbl_frac:.1%} of frames  [norm: 20-25% walking, >35% elderly]')
        if any((p > 0 for p in imu_pks)):
            print(f'     IMU_peak    : min={min(imu_pks):.2f}  max={max(imu_pks):.2f}  mean={np.mean(imu_pks):.2f} m/s^2')

class FallTypeLibrary:
    """
    Comprehensive fall scenarios based on Ferrari et al.
    """
    FALL_TYPES = {'backward_walking': {'description': 'Backward fall while walking', 'reward_sequence': ['walk', 'lie_down_up'], 'force_direction': [-1, 0, 0.3], 'application_point': 'Pelvis', 'perturbation_timing': 'mid_stance'}, 'forward_stumble': {'description': 'Forward fall from stumbling', 'reward_sequence': ['walk', 'lie_down_forward'], 'force_direction': [0.5, 0, -0.3], 'application_point': 'Foot', 'perturbation_timing': 'heel_strike'}, 'lateral_left': {'description': 'Left side fall', 'reward_sequence': ['stand', 'lie_down_side_left'], 'force_direction': [0, -1, 0.2], 'application_point': 'Torso', 'perturbation_timing': 'instant'}, 'lateral_right': {'description': 'Right side fall', 'reward_sequence': ['stand', 'lie_down_side_right'], 'force_direction': [0, 1, 0.2], 'application_point': 'Torso', 'perturbation_timing': 'instant'}, 'slip_induced': {'description': 'Slip-induced fall on low friction surface', 'reward_sequence': ['walk_slippery', 'lie_down'], 'friction_modification': 0.1, 'perturbation_timing': 'random'}, 'collapse': {'description': 'Sudden collapse (muscle weakness)', 'reward_sequence': ['stand', 'collapse'], 'muscle_weakening': 'instant', 'perturbation_timing': 'none'}, 'sitting_loss_balance': {'description': 'Fall from sitting position', 'reward_sequence': ['sit', 'lie_down'], 'object_required': 'chair', 'perturbation_timing': 'delayed'}}

    @staticmethod
    def get_lie_down_reward(fall_type):
        """
        Get an orientation-appropriate reward for the fall type.
        Maps fall types to proper orientations; returns LieDownReward() or
        a custom reward function for prone / lateral falls.
        """
        orient_map = {'backward_walking': 'up', 'forward_stumble': 'down', 'lateral_left': 'left', 'lateral_right': 'right', 'slip_induced': 'up', 'collapse': 'up', 'sitting_loss_balance': 'up'}
        orient = orient_map.get(fall_type, 'up')
        if orient == 'up':
            return LieDownReward()
        elif orient == 'down':

            def prone_reward(mdl, dat):
                pelvis_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Pelvis')
                head_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Head')
                pelvis_pos = dat.xpos[pelvis_id]
                head_pos = dat.xpos[head_id]
                fwd_disp = head_pos[0] - pelvis_pos[0]
                height = (pelvis_pos[2] + head_pos[2]) / 2
                return -height + max(0, fwd_disp) * 0.5
            return prone_reward
        elif orient == 'left':

            def lateral_left_reward(mdl, dat):
                pelvis_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Pelvis')
                pelvis_pos = dat.xpos[pelvis_id]
                return -pelvis_pos[2] + 0.3 * pelvis_pos[1]
            return lateral_left_reward
        elif orient == 'right':

            def lateral_right_reward(mdl, dat):
                pelvis_id = mujoco.mj_name2id(mdl, MJOBJ_BODY, 'Pelvis')
                pelvis_pos = dat.xpos[pelvis_id]
                return -pelvis_pos[2] - 0.3 * pelvis_pos[1]
            return lateral_right_reward
        else:
            return LieDownReward()

class EnhancedBiofidelicController(BiofidelicFallController):
    """
    Extended controller with improved stability and realism.
    v2: protective reflexes use policy monitoring, not manual arm override.
    """

    def __init__(self, env, mj_model, mj_data, anthropometry=None):
        super().__init__(env, mj_model, mj_data)
        self.anthro = anthropometry or {}
        self.age_style = get_age_style(self.anthro.get('age_years', 35), self.anthro.get('height_m', 1.7), self.anthro.get('sex', 'male'), body_mass_kg=float(self.anthro.get('body_mass_kg', SIM_RESOLVED_WEIGHT)))
        self.current_phase = 'stand'
        self.protective_reflexes = True
        self.balance_control = True
        self.vestibular_delay = float(self.anthro.get('reaction_delay', 0.15))
        self.ankle_strategy_threshold = 0.02
        self.hip_strategy_threshold = 0.08
        self.walk_guidance_enabled = True
        self.walk_target_speed = float(self.age_style.get('target_walk_speed', 1.0))
        self.walk_ref_y = None
        self.walk_ref_yaw = None
        self.walk_ref_origin_xy = None
        self.walk_ref_fwd_xy = None
        self.walk_ref_lat_xy = None
        lat_scale = float(self.age_style.get('guidance_lateral_scale', 1.0))
        yaw_scale = float(self.age_style.get('guidance_yaw_scale', 1.0))
        self.walk_kp_speed = 7.5
        self.walk_kp_lateral = 12.0 * lat_scale
        self.walk_kd_lateral = 13.0 * lat_scale
        self.walk_kp_yaw = 10.0 * yaw_scale
        self.walk_kd_yaw = 2.0 * np.sqrt(max(yaw_scale, 1e-06))
        self.walk_force_limit_x = 18.0
        self.walk_force_limit_y = 16.0 * lat_scale
        self.walk_torque_limit_z = 14.0 * yaw_scale
        self.walk_guidance_alpha = float(self.age_style.get('guidance_alpha', 0.88))
        self.walk_guidance_force_xy = np.zeros(2, dtype=float)
        self.walk_guidance_tz = 0.0
        self.walk_control_expected_ds = float(self.age_style.get('expected_double_support', 0.3))
        self._pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        self._torso_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Torso')
        self._head_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Head')
        self._upright_pelvis_height_ref = float(self.mj_data.xpos[self._pelvis_id][2]) if self._pelvis_id >= 0 else None
        self._posture_height_vel_lp = 0.0
        self.walk_leg_gain_boost = 1.0
        self.walk_speed_ema = 0.0
        self.walk_signed_speed_ema = 0.0
        self.walk_forward_locked = False
        self.walk_forward_lock_count = 0
        self._last_protective_state = None

    def get_protective_action(self, obs, z_fall, fall_direction):
        """
        Get action from policy; monitor (but do NOT override) protective reflexes.
        The Meta Motivo policy learns protective behaviours when properly rewarded.
        """
        action = self.get_action(obs, z_fall)
        if self.protective_reflexes:
            self._monitor_protective_response(fall_direction)
        return action

    def _monitor_protective_response(self, fall_direction):
        """Monitor and log protective responses for validation."""
        pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        pelvis_vel = self.mj_data.cvel[pelvis_id]
        pelvis_pos = self.mj_data.xpos[pelvis_id]
        if pelvis_vel[2] < -1.5 and pelvis_pos[2] < 0.8:
            arm_positions = self._get_arm_positions()
            self._last_protective_state = {'pelvis_velocity': pelvis_vel.copy(), 'arm_extension': arm_positions, 'timestamp': self.mj_data.time, 'fall_direction': fall_direction}

    def _get_arm_positions(self):
        arm_data = {}
        arm_keywords = ['shoulder', 'elbow', 'wrist']
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and any((k in name.lower() for k in arm_keywords)):
                arm_data[name] = self.mj_data.qpos[self.mj_model.jnt_qposadr[i]]
        return arm_data

    def start_walk_phase(self):
        pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        self.walk_leg_gain_boost = 1.0
        self.walk_speed_ema = 0.0
        self.walk_signed_speed_ema = 0.0
        self.walk_forward_locked = False
        self.walk_forward_lock_count = 0
        if pelvis_id >= 0:
            pos = self.mj_data.xpos[pelvis_id].copy()
            self.walk_ref_y = float(pos[1])
            self.walk_ref_origin_xy = np.asarray(pos[:2], dtype=float).copy()
            diag_heading = np.asarray(LAST_Z_WALK_DIAGNOSTICS.get('heading_xy', [1.0, 0.0]), dtype=float)
            diag_norm = float(np.linalg.norm(diag_heading))
            if diag_norm > 1e-06:
                ref_fwd = diag_heading / diag_norm
            else:
                R = self.mj_data.xmat[pelvis_id].reshape(3, 3)
                fwd = np.asarray(R[:, 0], dtype=float)
                yaw_tmp = float(np.arctan2(fwd[1], fwd[0]))
                ref_fwd = np.array([np.cos(yaw_tmp), np.sin(yaw_tmp)], dtype=float)
            self.walk_ref_yaw = float(np.arctan2(ref_fwd[1], ref_fwd[0]))
            ref_lat = np.array([-ref_fwd[1], ref_fwd[0]], dtype=float)
            self.walk_ref_fwd_xy = ref_fwd
            self.walk_ref_lat_xy = ref_lat

    def _compute_sagittal_trunk_lean_deg(self, ref_fwd):
        if self._pelvis_id < 0 or self._head_id < 0:
            return 0.0
        pelvis = self.mj_data.xpos[self._pelvis_id].copy()
        head = self.mj_data.xpos[self._head_id].copy()
        vec = np.asarray(head - pelvis, dtype=float)
        up_comp = float(vec[2])
        fwd_comp = float(np.dot(vec[:2], np.asarray(ref_fwd[:2], dtype=float)))
        return float(np.degrees(np.arctan2(fwd_comp, max(abs(up_comp), 1e-06))))

    def _apply_age_posture_bias(self, ref_fwd, ref_lat):
        if self.current_phase not in ('stand', 'walk') or self.rest_mode:
            return
        if self._torso_id < 0:
            return
        if self.current_phase == 'stand':
            target_deg = float(self.age_style.get('stand_stoop_target_deg', self.age_style.get('stoop_target_deg', 7.0)))
            base_kp, base_kd = (0.46, 0.22)
            torque_limit = 9.0
            force_limit = 6.0
            pelvis_torque_gain = 0.3
        else:
            target_deg = float(self.age_style.get('walk_stoop_target_deg', self.age_style.get('stoop_target_deg', 7.0)))
            base_kp, base_kd = (0.22, 0.12)
            torque_limit = 4.5
            force_limit = 2.5
            pelvis_torque_gain = 0.12
        current_deg = self._compute_sagittal_trunk_lean_deg(ref_fwd)
        stoop_err = float(target_deg - current_deg)
        if self._pelvis_id >= 0:
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, self._pelvis_id, vel6, 0)
            ang = vel6[:3].copy()
        else:
            ang = np.zeros(3)
        lat_axis = np.array([ref_lat[0], ref_lat[1], 0.0], dtype=float)
        pitch_rate = float(np.dot(ang, lat_axis))
        age_scale = float(np.clip((target_deg - 4.0) / 10.0, 0.0, 1.2))
        kp = base_kp + 0.12 * age_scale
        kd = base_kd + 0.06 * age_scale
        pitch_torque = float(np.clip(kp * stoop_err - kd * pitch_rate, -torque_limit, torque_limit))
        forward_force = float(np.clip(0.22 * stoop_err, -force_limit, force_limit))
        self.mj_data.xfrc_applied[self._torso_id, 0] += forward_force * float(ref_fwd[0])
        self.mj_data.xfrc_applied[self._torso_id, 1] += forward_force * float(ref_fwd[1])
        self.mj_data.xfrc_applied[self._torso_id, 3] += pitch_torque * float(ref_lat[0])
        self.mj_data.xfrc_applied[self._torso_id, 4] += pitch_torque * float(ref_lat[1])
        if self._pelvis_id >= 0:
            pelvis_torque = pelvis_torque_gain * pitch_torque
            self.mj_data.xfrc_applied[self._pelvis_id, 3] += pelvis_torque * float(ref_lat[0])
            self.mj_data.xfrc_applied[self._pelvis_id, 4] += pelvis_torque * float(ref_lat[1])

    def apply_age_posture_bias_only(self):
        if self._pelvis_id >= 0:
            R = self.mj_data.xmat[self._pelvis_id].reshape(3, 3)
            fwd = np.asarray(R[:, 0], dtype=float)
            yaw = float(np.arctan2(fwd[1], fwd[0]))
            ref_fwd = np.asarray(self.walk_ref_fwd_xy, dtype=float) if self.walk_ref_fwd_xy is not None else np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        else:
            ref_fwd = np.array([1.0, 0.0], dtype=float)
        ref_lat = np.array([-ref_fwd[1], ref_fwd[0]], dtype=float)
        self._apply_age_posture_bias(ref_fwd, ref_lat)

    def apply_walk_guidance(self):
        """
        Small pelvis-level guidance during the WALK phase only.

        Important fix: guidance is now expressed in the *reference heading frame*
        captured at walk onset, not the avatar's drifting instantaneous heading.
        This suppresses the large circular-walk artefact seen in the previous run.
        """
        if not self.walk_guidance_enabled or self.current_phase != 'walk' or self.rest_mode:
            return
        body_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        if body_id < 0:
            return
        pos = self.mj_data.xpos[body_id].copy()
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, body_id, vel6, 0)
        vel = vel6[3:].copy()
        ang = vel6[:3].copy()
        R = self.mj_data.xmat[body_id].reshape(3, 3)
        fwd_body = np.asarray(R[:, 0], dtype=float)
        yaw = float(np.arctan2(fwd_body[1], fwd_body[0]))
        if self.walk_ref_y is None or self.walk_ref_yaw is None or self.walk_ref_origin_xy is None:
            self.start_walk_phase()
        ref_fwd = np.asarray(self.walk_ref_fwd_xy, dtype=float) if self.walk_ref_fwd_xy is not None else np.array([1.0, 0.0], dtype=float)
        ref_lat = np.asarray(self.walk_ref_lat_xy, dtype=float) if self.walk_ref_lat_xy is not None else np.array([0.0, 1.0], dtype=float)
        pos_xy = np.asarray(pos[:2], dtype=float)
        vel_xy = np.asarray(vel[:2], dtype=float)
        signed_speed_probe = float(np.dot(vel_xy, ref_fwd))
        self.walk_signed_speed_ema = 0.88 * float(self.walk_signed_speed_ema) + 0.12 * signed_speed_probe
        if not self.walk_forward_locked:
            self.walk_forward_lock_count += 1
            if self.walk_forward_lock_count >= 12:
                if self.walk_signed_speed_ema < -0.05:
                    ref_fwd = -ref_fwd
                    ref_lat = -ref_lat
                    self.walk_ref_fwd_xy = ref_fwd.copy()
                    self.walk_ref_lat_xy = ref_lat.copy()
                    self.walk_ref_yaw = float(np.arctan2(ref_fwd[1], ref_fwd[0]))
                self.walk_forward_locked = True
        rel_xy = pos_xy - np.asarray(self.walk_ref_origin_xy, dtype=float)
        forward_speed = float(np.dot(vel_xy, ref_fwd))
        lateral_speed = float(np.dot(vel_xy, ref_lat))
        lateral_err = float(np.dot(rel_xy, ref_lat))
        self.walk_speed_ema = 0.88 * float(self.walk_speed_ema) + 0.12 * max(forward_speed, 0.0)
        speed_err = self.walk_target_speed - forward_speed
        braking_scale = float(self.age_style.get('braking_scale', 1.0))
        strict_gain = float(self.age_style.get('speed_supervision_gain', 0.0))
        speed_shortfall = self.walk_target_speed - float(self.walk_speed_ema)
        self.walk_leg_gain_boost = float(np.clip(1.0 + strict_gain * speed_shortfall, 0.96, 1.2))
        forward_boost = float(np.clip(1.0 + 1.4 * strict_gain * max(speed_shortfall, 0.0), 1.0, 1.35))
        if speed_err < 0.0:
            fx_ref = self.walk_kp_speed * braking_scale * speed_err
        else:
            fx_ref = self.walk_kp_speed * forward_boost * speed_err
        fy_ref = -self.walk_kp_lateral * lateral_err - self.walk_kd_lateral * lateral_speed
        yaw_err = float(np.arctan2(np.sin(yaw - self.walk_ref_yaw), np.cos(yaw - self.walk_ref_yaw)))
        heading_scale = float(np.clip(np.cos(yaw_err), 0.0, 1.0))
        fx_ref *= 0.35 + 0.65 * heading_scale
        tz = -(self.walk_kp_yaw * (1.0 + 1.8 * (1.0 - heading_scale))) * yaw_err - self.walk_kd_yaw * float(ang[2])
        brake_limit_x = float(self.walk_force_limit_x * self.age_style.get('braking_scale', 1.0))
        fx_ref = float(np.clip(fx_ref, -brake_limit_x, self.walk_force_limit_x))
        fy_ref = float(np.clip(fy_ref, -self.walk_force_limit_y, self.walk_force_limit_y))
        tz = float(np.clip(tz, -self.walk_torque_limit_z, self.walk_torque_limit_z))
        f_world_xy = fx_ref * ref_fwd + fy_ref * ref_lat
        a = float(self.walk_guidance_alpha)
        self.walk_guidance_force_xy = a * self.walk_guidance_force_xy + (1.0 - a) * np.asarray(f_world_xy, dtype=float)
        self.walk_guidance_tz = a * float(self.walk_guidance_tz) + (1.0 - a) * float(tz)
        self.mj_data.xfrc_applied[body_id, 0] = float(self.walk_guidance_force_xy[0])
        self.mj_data.xfrc_applied[body_id, 1] = float(self.walk_guidance_force_xy[1])
        self.mj_data.xfrc_applied[body_id, 5] = float(self.walk_guidance_tz)
        self._apply_age_posture_bias(ref_fwd, ref_lat)

    def get_protective_reward_bonus(self):
        """
        Reward bonus for protective behaviours (call from reward function).
        """
        bonus = 0.0
        contacts = self._analyze_contact_sequence()
        if contacts.get('hands_first', False):
            bonus += 0.5
        if self._lie_orient == 'up':
            neck_flexion = self._get_neck_flexion()
            if neck_flexion > 0.3:
                bonus += 0.3
        return bonus

    def _analyze_contact_sequence(self):
        result = {'hands_first': False, 'contacts': []}
        body_names = ['HandL', 'HandR', 'Torso', 'Head', 'Pelvis']
        body_ids = {}
        for bname in body_names:
            bid = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, bname)
            if bid >= 0:
                body_ids[bname] = bid
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            for bname, bid in body_ids.items():
                if contact.geom1 == bid or contact.geom2 == bid:
                    result['contacts'].append((bname, self.mj_data.time))
        if result['contacts']:
            result['contacts'].sort(key=lambda x: x[1])
            if result['contacts'][0][0] in ('HandL', 'HandR'):
                result['hands_first'] = True
        return result

    def _get_neck_flexion(self):
        """Measure neck flexion angle (chin tuck) - placeholder."""
        return 0.5

    def _estimate_support_polygon_center(self):
        weighted_xy = np.zeros(2)
        total_fz = 0.0
        foot_positions = []
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = (c.geom1, c.geom2)
            names = ((mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g2) or '').lower())
            is_ground = any(('floor' in n or 'ground' in n or 'plane' in n for n in names)) or any((self.mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2)))
            if not is_ground:
                continue
            non_ground_geom = g2 if 'floor' in names[0] or 'ground' in names[0] or 'plane' in names[0] or (g1 in self._ground_geoms) else g1
            body_id = int(self.mj_model.geom_bodyid[non_ground_geom])
            bname = (mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, body_id) or '').lower()
            if not any((k in bname for k in ('foot', 'toe', 'heel', 'ankle'))):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            f_world, _ = _contact_wrench_world(c, wrench)
            fz = max(0.0, float(f_world[2]))
            if fz > 0.5:
                weighted_xy += np.asarray(c.pos[:2], dtype=float) * fz
                total_fz += fz
        if total_fz > 1.0:
            return weighted_xy / total_fz
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i)
            if name and any((k in name.lower() for k in ('foot', 'heel', 'toe', 'ankle'))):
                foot_positions.append(self.mj_data.xpos[i][:2])
        if foot_positions:
            return np.mean(foot_positions, axis=0)
        return np.zeros(2)

    def compute_xcom(self):
        """
        XCoM using body/world linear velocity from mj_objectVelocity rather than
        raw cvel spatial quantities.
        """
        pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        com_pos = self.mj_data.xpos[pelvis_id]
        com_vel = _body_world_velocity(self.mj_model, self.mj_data, pelvis_id)
        g = 9.81
        leg_length = max(0.1, float(self.anthro.get('leg_length', 0.53 * 1.75)))
        omega = np.sqrt(g / leg_length)
        xcom_2d = com_pos[:2] + np.asarray(com_vel[:2], dtype=float) / omega
        pts = []
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = (c.geom1, c.geom2)
            names = ((mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g2) or '').lower())
            is_ground = any(('floor' in n or 'ground' in n or 'plane' in n for n in names)) or any((self.mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2)))
            if not is_ground:
                continue
            non_ground_geom = g2 if 'floor' in names[0] or 'ground' in names[0] or 'plane' in names[0] or (g1 in self._ground_geoms) else g1
            body_id = int(self.mj_model.geom_bodyid[non_ground_geom])
            bname = (mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, body_id) or '').lower()
            if not any((k in bname for k in ('foot', 'toe', 'heel', 'ankle'))):
                continue
            pts.append(np.asarray(c.pos[:2], dtype=float))
        if not pts:
            pts = []
            for i in range(self.mj_model.nbody):
                name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i)
                if name and any((k in name.lower() for k in ('foot', 'toe', 'heel', 'ankle'))):
                    pts.append(np.asarray(self.mj_data.xpos[i][:2], dtype=float))
        support_center = self._estimate_support_polygon_center()
        if pts:
            arr = np.vstack(pts)
            min_xy = np.min(arr, axis=0) - np.array([0.06, 0.04])
            max_xy = np.max(arr, axis=0) + np.array([0.06, 0.04])
            dx_out = max(min_xy[0] - xcom_2d[0], 0.0, xcom_2d[0] - max_xy[0])
            dy_out = max(min_xy[1] - xcom_2d[1], 0.0, xcom_2d[1] - max_xy[1])
            if dx_out == 0.0 and dy_out == 0.0:
                margin = min(xcom_2d[0] - min_xy[0], max_xy[0] - xcom_2d[0], xcom_2d[1] - min_xy[1], max_xy[1] - xcom_2d[1])
            else:
                margin = -float(np.hypot(dx_out, dy_out))
        else:
            BASE_HALF_WIDTH = 0.18
            dist_xcom_to_center = float(np.linalg.norm(xcom_2d - support_center))
            margin = BASE_HALF_WIDTH - dist_xcom_to_center
        return (xcom_2d, com_pos[:2].copy(), float(margin))

    def _ankle_strategy_action(self, obs, z):
        return self.get_action(obs, z)

    def _hip_strategy_action(self, obs, z):
        return self.get_action(obs, z)

    def apply_ankle_hip_strategy(self, obs, z):
        """
        Balance strategies upgraded to use XCoM margin (Hof et al. 2005):
        - XCoM margin > 0.08 m : stable - ankle strategy
        - XCoM margin > 0.02 m : borderline - hip strategy
        - XCoM margin = 0.02 m : capture-point outside base - step / fall
        Falls back to CoM displacement if XCoM computation is unavailable.
        """
        try:
            xcom_2d, com_2d, margin = self.compute_xcom()
            if margin > 0.08:
                return self._ankle_strategy_action(obs, z)
            elif margin > 0.02:
                return self._hip_strategy_action(obs, z)
            else:
                return self.get_action(obs, z)
        except Exception:
            pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
            com_pos = self.mj_data.xpos[pelvis_id]
            support_center = self._estimate_support_polygon_center()
            com_disp = np.linalg.norm(com_pos[:2] - support_center)
            if com_disp < self.ankle_strategy_threshold:
                return self._ankle_strategy_action(obs, z)
            elif com_disp < self.hip_strategy_threshold:
                return self._hip_strategy_action(obs, z)
            else:
                return self.get_action(obs, z)

def run_enhanced_simulation(fall_type='backward_walking', age=75, height=1.65, weight=None, save_imu=True, sex='male'):
    """
    Run a complete fall simulation with all enhancements:
    - Anthropometric customization (age, height, default model mass)
    - IMU data logging and CSV export
    - FallTypeLibrary scenario selection
    - EnhancedBiofidelicController with protective reflexes
    """
    from datetime import datetime
    resolved_weight = float(weight) if weight is not None else resolve_subject_weight_kg(height, age, sex, explicit_weight=None)[0]
    print('\n' + '=' * 70)
    print(f'  Enhanced Simulation: {fall_type} | age={age} height={height}m sex={sex} weight={resolved_weight:.1f}kg')
    print('=' * 70)
    env_enhanced, _ = make_humenv(task='move-ego-0-0')
    obs_enh, _ = env_enhanced.reset()
    mj_model_enh = env_enhanced.unwrapped.model
    mj_data_enh = env_enhanced.unwrapped.data
    myosuite_mount_enh = resolve_imu_mount_configuration(mj_model_enh, requested_xml_path=MYOSUITE_INTEGRATION['model_xml'] if MYOSUITE_INTEGRATION.get('enable_reference_xml', True) else '')
    anthro = AnthropometricModel(mj_model_enh, age=age, height=height, weight=resolved_weight, sex=sex)
    age_params = anthro.apply_age_effects()
    print(f"  Anthropometry: strength={age_params['strength_factor']:.2f} (raw={age_params.get('strength_factor_raw', age_params['strength_factor']):.2f}), reaction_delay={age_params['reaction_delay']:.3f}s, balance_impairment={age_params['balance_impairment']:.3f}")
    imu = IMUValidator(mj_model_enh, mj_data_enh, sensor_body=myosuite_mount_enh.get('sensor_body', IMU_HARDWARE_SPEC['proxy_body']), sensor_site=myosuite_mount_enh.get('sensor_site'), sensor_offset_local=myosuite_mount_enh.get('sensor_offset_local'), age=age, height=height, target_output_hz=IMU_HARDWARE_SPEC['sampling_hz'], mount_label=myosuite_mount_enh.get('mount_label', IMU_HARDWARE_SPEC['mount_label']))
    imu.print_configuration_report(age=age, height=height, sex=sex)
    _ver = imu.get_sampling_report()
    print(f"    [IMU reality] effective physical bandwidth  {_ver['effective_bandwidth_hz']:.2f} Hz")
    print(f"    [IMU reality] true hardware-equivalent 100 Hz = {_ver['true_hardware_equivalent_100hz']}")
    controller_enh = EnhancedBiofidelicController(env_enhanced, mj_model_enh, mj_data_enh, anthropometry=age_params)
    global PHASES, TOTAL_STEPS, FORCE_CONFIG, WEAKENING_CONFIG
    PHASES = phase_timing(age, sex)
    TOTAL_STEPS = sum(PHASES.values())
    model_body_mass_enh = float(np.sum(mj_model_enh.body_mass))
    FORCE_CONFIG.update(perturbation_force_config(model_body_mass_enh, age, sex, bw_fraction=1.15))
    WEAKENING_CONFIG = weakening_config(age, sex, model_body_mass_enh)
    print(f"  Weakening: min_factor={WEAKENING_CONFIG['min_factor']:.3f}, decay_time={WEAKENING_CONFIG['decay_time']} steps, fatigue_k={WEAKENING_CONFIG.get('fatigue_coefficient', 1.0):.2f}, sarcopenia_loss={WEAKENING_CONFIG.get('sarcopenia_force_loss', 0.0):.1%}")
    fall_config = FallTypeLibrary.FALL_TYPES.get(fall_type, FallTypeLibrary.FALL_TYPES['backward_walking'])
    print(f"  Fall scenario: {fall_config['description']}")
    try:
        z_w = z_walk
        z_f = z_fall
        z_s = z_stand
        print('  Reusing pre-computed task embeddings.')
    except NameError:
        print('  Computing task embeddings from scratch...')
        z_s = infer_z_stand()
        z_w = infer_z_walk_stable()
        z_f = infer_z_backward_fall()
    enh_boundaries = np.cumsum([0] + list(PHASES.values()))
    with mujoco.viewer.launch_passive(mj_model_enh, mj_data_enh) as viewer_enh:
        viewer_enh.cam.distance = 4.5
        viewer_enh.cam.elevation = -10
        viewer_enh.cam.azimuth = 90
        for step in range(TOTAL_STEPS):
            if step < enh_boundaries[1]:
                phase = 'stand'
                if step == 0:
                    controller_enh.set_target_z(z_s, blend_steps=10)
            elif step < enh_boundaries[2]:
                phase = 'walk'
                if step == enh_boundaries[1]:
                    controller_enh.set_target_z(z_w, blend_steps=30)
                controller_enh.start_walk_phase()
            elif step < enh_boundaries[3]:
                phase = 'perturb'
                if step == enh_boundaries[2]:
                    controller_enh.set_target_z(z_f, blend_steps=40)
                ramp_progress = (step - enh_boundaries[2]) / PHASES['perturb']
                force_dir = np.array(fall_config.get('force_direction', FORCE_CONFIG['direction']), dtype=float)
                force_dir /= np.linalg.norm(force_dir)
                scaled_magnitude = FORCE_CONFIG['magnitude'] * (1.0 - age_params.get('balance_impairment', 0.0))
                controller_enh.apply_external_force(scaled_magnitude, force_dir, ramp_progress)
                weaken_progress = (step - enh_boundaries[2]) / (PHASES['perturb'] + PHASES['react'])
                controller_enh.update_muscle_weakening(weaken_progress)
            elif step < enh_boundaries[4]:
                phase = 'react'
                force_dir = np.array(fall_config.get('force_direction', FORCE_CONFIG['direction']), dtype=float)
                force_dir /= np.linalg.norm(force_dir)
                scaled_magnitude = FORCE_CONFIG['magnitude'] * (1.0 - age_params.get('balance_impairment', 0.0))
                controller_enh.apply_external_force(scaled_magnitude, force_dir, 1.0)
                weaken_progress = (step - enh_boundaries[2]) / (PHASES['perturb'] + PHASES['react'])
                controller_enh.update_muscle_weakening(weaken_progress)
            else:
                phase = 'fall'
                if step == enh_boundaries[4]:
                    controller_enh.clear_forces()
                    controller_enh.finalize_z_transition()
                controller_enh.update_muscle_weakening(1.0)
            z_current = controller_enh.update_z_interpolation()
            controller_enh.current_phase = phase
            action = controller_enh.get_protective_action(obs_enh, z_current, fall_type)
            if phase == 'walk':
                controller_enh.apply_walk_guidance()
            obs_enh, _, terminated, truncated, _ = env_enhanced.step(action)
            sim_time_now = float(mj_data_enh.time)
            _last_imu_peak = imu.log_frame(sim_time_now)
            viewer_enh.sync()
            if terminated and step < enh_boundaries[1]:
                obs_enh, _ = env_enhanced.reset()
                controller_enh.restore_strength()
    if save_imu:
        filename = f"fall_{fall_type}_age{age}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        imu.export_to_csv(filename, metadata={'age': age, 'height': height, 'sex': sex, 'weight': float(np.sum(mj_model_enh.body_mass)), 'body_mass_kg': float(np.sum(mj_model_enh.body_mass)), 'fall_type': fall_type})
    controller_enh.restore_strength()
    controller_enh.clear_forces()
    env_enhanced.close()
    print(f'\n  Enhanced simulation complete: {fall_type}')

def compute_preperturb_walk_metrics(controller, mj_model, mj_data, body_mass, age_params):
    pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'Pelvis')
    if pelvis_id < 0:
        return {'ready': False, 'ready_mode': 'none', 'speed': 0.0, 'froude': 0.0, 'xcom_margin': -1.0, 'double_support': True, 'grf_bw': 0.0, 'lateral_err': 0.0, 'yaw_err_deg': 0.0, 'reference_speed_ratio': 0.0, 'policy_limited': False}
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vel6, 0)
    vel = vel6[3:].copy()
    pos = mj_data.xpos[pelvis_id].copy()
    R = mj_data.xmat[pelvis_id].reshape(3, 3)
    fwd_body = np.asarray(R[:, 0], dtype=float)
    yaw = float(np.arctan2(fwd_body[1], fwd_body[0]))
    if getattr(controller, 'walk_ref_fwd_xy', None) is not None:
        ref_fwd = np.asarray(controller.walk_ref_fwd_xy, dtype=float)
        ref_lat = np.asarray(controller.walk_ref_lat_xy, dtype=float)
        origin_xy = np.asarray(controller.walk_ref_origin_xy, dtype=float)
    else:
        ref_fwd = np.asarray(fwd_body[:2], dtype=float)
        ref_fwd /= max(np.linalg.norm(ref_fwd), 1e-09)
        ref_lat = np.array([-ref_fwd[1], ref_fwd[0]], dtype=float)
        origin_xy = np.asarray(pos[:2], dtype=float)
    vel_xy = np.asarray(vel[:2], dtype=float)
    forward_speed = float(np.dot(vel_xy, ref_fwd))
    abs_speed = abs(forward_speed)
    lateral_err = float(np.dot(np.asarray(pos[:2], dtype=float) - origin_xy, ref_lat))
    yaw_ref = float(getattr(controller, 'walk_ref_yaw', yaw))
    yaw_err_deg = float(np.degrees(np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref))))
    froude = float(abs_speed ** 2 / max(9.81 * float(age_params.get('leg_length', 0.9)), 1e-09))
    try:
        _, _, xcom_margin = controller.compute_xcom()
    except Exception:
        xcom_margin = -1.0
    left_vertical = 0.0
    right_vertical = 0.0
    total_ground_vertical = 0.0
    for i in range(int(mj_data.ncon)):
        c = mj_data.contact[i]
        g1, g2 = (int(c.geom1), int(c.geom2))
        names = ((mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g2) or '').lower())
        ground = any(('floor' in n or 'ground' in n or 'plane' in n for n in names)) or any((mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2)))
        if not ground:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, wrench)
        if wrench[0] <= 1.0:
            continue
        f_world, _ = _contact_wrench_world(c, wrench)
        total_ground_vertical += max(0.0, float(f_world[2]))
        ng = g2 if g1 in controller._ground_geoms else g1
        bid = int(mj_model.geom_bodyid[ng])
        bname = (mujoco.mj_id2name(mj_model, MJOBJ_BODY, bid) or '').lower()
        is_footlike = any((k in bname for k in ('foot', 'ankle', 'toe', 'heel')))
        if is_footlike and _body_name_matches_side(bname, 'left'):
            left_vertical += max(0.0, float(f_world[2]))
        if is_footlike and _body_name_matches_side(bname, 'right'):
            right_vertical += max(0.0, float(f_world[2]))
    foot_load_threshold = 0.12 * body_mass * 9.81
    left_support = bool(left_vertical > foot_load_threshold)
    right_support = bool(right_vertical > foot_load_threshold)
    double_support = bool(left_support and right_support)
    grf_bw = float(total_ground_vertical / max(body_mass * 9.81, 1.0))
    lit_target = float(getattr(controller, 'walk_lit_target_speed', controller.age_style.get('target_walk_speed', max(controller.walk_target_speed, 0.2))))
    policy_target = float(max(controller.walk_target_speed, 0.18))
    policy_limited = bool(getattr(controller, 'walk_policy_limited', False))
    reference_speed_ratio = float(abs_speed / max(lit_target, 1e-06))
    stable_base = xcom_margin > -0.22 and grf_bw < 2.6
    reference_ready = stable_base and abs_speed >= 0.7 * lit_target
    policy_limited_ready = stable_base and policy_limited and (abs_speed >= 0.8 * policy_target)
    ready = bool(reference_ready or policy_limited_ready)
    ready_mode = 'reference' if reference_ready else 'policy_limited' if policy_limited_ready else 'none'
    return {'ready': ready, 'ready_mode': ready_mode, 'speed': float(forward_speed), 'froude': float(froude), 'xcom_margin': float(xcom_margin), 'double_support': bool(double_support), 'grf_bw': float(grf_bw), 'lateral_err': float(lateral_err), 'yaw_err_deg': float(yaw_err_deg), 'reference_speed_ratio': float(reference_speed_ratio), 'policy_limited': bool(policy_limited)}

def detect_fall_events(imu_data_buffer, dynamics_frames, marker_frames=None, perturb_start_time=None):
    ts = np.asarray(imu_data_buffer.get('timestamp', []), dtype=float)
    heights = np.asarray(imu_data_buffer.get('pelvis_height', []), dtype=float)
    vels = np.asarray(imu_data_buffer.get('pelvis_velocity', []), dtype=float)
    if ts.size == 0 or heights.size == 0:
        return {'available': False}
    if vels.ndim == 1:
        speed = np.abs(vels)
    else:
        speed = np.linalg.norm(vels[:, :2], axis=1)
    trunk = None
    if marker_frames:
        trunk = np.asarray([fr.get('trunk_lean_deg', 0.0) for fr in marker_frames], dtype=float)
    imp_t = np.asarray([fr.get('time', np.nan) for fr in dynamics_frames], dtype=float) if dynamics_frames else np.array([], dtype=float)
    imp_bw = np.asarray([fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0)) / max(1.0, 70.0 * 9.81) for fr in dynamics_frames], dtype=float) if dynamics_frames else np.array([], dtype=float)
    if dynamics_frames:
        body_mass_guess = float(max(1.0, np.median([fr.get('support_vertical_n', 0.0) for fr in dynamics_frames[:min(len(dynamics_frames), 30)] if fr.get('support_vertical_n', 0.0) > 0.0]) / 9.81)) if any((fr.get('support_vertical_n', 0.0) > 0.0 for fr in dynamics_frames[:min(len(dynamics_frames), 30)])) else 70.0
        imp_bw = np.asarray([fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0)) / max(1.0, body_mass_guess * 9.81) for fr in dynamics_frames], dtype=float)
    t0 = float(perturb_start_time) if perturb_start_time is not None else float(ts[0])
    i0 = int(np.searchsorted(ts, t0, side='left'))
    pre_h = float(np.median(heights[max(0, i0 - 30):max(i0, i0 + 1)])) if i0 > 0 else float(np.max(heights[:min(len(heights), 30)]))
    onset_idx = None
    for i in range(i0, len(ts)):
        trunk_cond = bool(trunk is not None and i < len(trunk) and (trunk[i] > 22.0))
        h_cond = heights[i] < max(0.78 * pre_h, pre_h - 0.1)
        v_cond = speed[i] > 0.55
        if trunk_cond and h_cond or (h_cond and v_cond):
            onset_idx = i
            break
    if onset_idx is None:
        onset_idx = i0
    impact_idx = None
    if imp_t.size:
        j0 = int(np.searchsorted(imp_t, ts[onset_idx], side='left'))
        for j in range(j0, len(imp_t)):
            if imp_bw[j] > 0.25:
                impact_idx = j
                break
        if impact_idx is None:
            impact_idx = int(np.argmax(imp_bw[j0:]) + j0) if j0 < len(imp_bw) else None
    settle_idx = None
    start_settle = onset_idx
    if impact_idx is not None:
        start_settle = max(start_settle, int(np.searchsorted(ts, imp_t[impact_idx], side='left')))
    streak = 0
    for i in range(start_settle, len(ts)):
        low_h = heights[i] < 0.2
        low_v = speed[i] < 0.08
        if low_h and low_v:
            streak += 1
        else:
            streak = 0
        if streak >= 35:
            settle_idx = i - 34
            break
    impact_time = float(imp_t[impact_idx]) if impact_idx is not None else float(ts[min(len(ts) - 1, onset_idx)])
    onset_time = float(ts[onset_idx])
    settle_time = float(ts[settle_idx]) if settle_idx is not None else float(ts[-1])
    return {'available': True, 'onset_time': onset_time, 'impact_time': impact_time, 'settle_time': settle_time, 'fall_duration_s': max(0.0, settle_time - onset_time), 'lead_time_ms': max(0.0, (impact_time - onset_time) * 1000.0), 'onset_idx': int(onset_idx), 'impact_idx': int(impact_idx) if impact_idx is not None else None, 'settle_idx': int(settle_idx) if settle_idx is not None else None}

def compute_preperturb_walk_metrics(controller, mj_model, mj_data, body_mass, age_params):
    pelvis_id = mujoco.mj_name2id(mj_model, MJOBJ_BODY, 'Pelvis')
    if pelvis_id < 0:
        return {'ready': False, 'ready_mode': 'none', 'speed': 0.0, 'froude': 0.0, 'xcom_margin': -1.0, 'double_support': True, 'grf_bw': 0.0, 'lateral_err': 0.0, 'yaw_err_deg': 0.0, 'reference_speed_ratio': 0.0, 'policy_limited': False}
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vel6, 0)
    vel = vel6[3:].copy()
    pos = mj_data.xpos[pelvis_id].copy()
    R = mj_data.xmat[pelvis_id].reshape(3, 3)
    fwd_body = np.asarray(R[:, 0], dtype=float)
    yaw = float(np.arctan2(fwd_body[1], fwd_body[0]))
    if getattr(controller, 'walk_ref_fwd_xy', None) is not None:
        ref_fwd = np.asarray(controller.walk_ref_fwd_xy, dtype=float)
        ref_lat = np.asarray(controller.walk_ref_lat_xy, dtype=float)
        origin_xy = np.asarray(controller.walk_ref_origin_xy, dtype=float)
    else:
        ref_fwd = np.asarray(fwd_body[:2], dtype=float)
        ref_fwd /= max(np.linalg.norm(ref_fwd), 1e-09)
        ref_lat = np.array([-ref_fwd[1], ref_fwd[0]], dtype=float)
        origin_xy = np.asarray(pos[:2], dtype=float)
    vel_xy = np.asarray(vel[:2], dtype=float)
    forward_speed = float(np.dot(vel_xy, ref_fwd))
    abs_speed = abs(forward_speed)
    lateral_err = float(np.dot(np.asarray(pos[:2], dtype=float) - origin_xy, ref_lat))
    yaw_ref = float(getattr(controller, 'walk_ref_yaw', yaw))
    yaw_err_deg = float(np.degrees(np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref))))
    froude = float(abs_speed ** 2 / max(9.81 * float(age_params.get('leg_length', 0.9)), 1e-09))
    try:
        _, _, xcom_margin = controller.compute_xcom()
    except Exception:
        xcom_margin = -1.0
    left_vertical = 0.0
    right_vertical = 0.0
    total_ground_vertical = 0.0
    for i in range(int(mj_data.ncon)):
        c = mj_data.contact[i]
        g1, g2 = (int(c.geom1), int(c.geom2))
        names = ((mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g1) or '').lower(), (mujoco.mj_id2name(mj_model, MJOBJ_GEOM, g2) or '').lower())
        ground = any(('floor' in n or 'ground' in n or 'plane' in n for n in names)) or any((mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999) for g in (g1, g2)))
        if not ground:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, wrench)
        if wrench[0] <= 1.0:
            continue
        f_world, _ = _contact_wrench_world(c, wrench)
        total_ground_vertical += max(0.0, float(f_world[2]))
        ng = g2 if g1 in controller._ground_geoms else g1
        bid = int(mj_model.geom_bodyid[ng])
        bname = (mujoco.mj_id2name(mj_model, MJOBJ_BODY, bid) or '').lower()
        is_footlike = any((k in bname for k in ('foot', 'ankle', 'toe', 'heel')))
        if is_footlike and _body_name_matches_side(bname, 'left'):
            left_vertical += max(0.0, float(f_world[2]))
        if is_footlike and _body_name_matches_side(bname, 'right'):
            right_vertical += max(0.0, float(f_world[2]))
    foot_load_threshold = 0.12 * body_mass * 9.81
    left_support = bool(left_vertical > foot_load_threshold)
    right_support = bool(right_vertical > foot_load_threshold)
    double_support = bool(left_support and right_support)
    grf_bw = float(total_ground_vertical / max(body_mass * 9.81, 1.0))
    lit_target = float(getattr(controller, 'walk_lit_target_speed', controller.age_style.get('target_walk_speed', max(controller.walk_target_speed, 0.2))))
    policy_target = float(max(controller.walk_target_speed, 0.18))
    policy_limited = bool(getattr(controller, 'walk_policy_limited', False))
    reference_speed_ratio = float(abs_speed / max(lit_target, 1e-06))
    stable_base = xcom_margin > -0.22 and grf_bw < 2.6
    reference_ready = stable_base and abs_speed >= 0.7 * lit_target
    policy_limited_ready = stable_base and policy_limited and (abs_speed >= 0.8 * policy_target)
    ready = bool(reference_ready or policy_limited_ready)
    ready_mode = 'reference' if reference_ready else 'policy_limited' if policy_limited_ready else 'none'
    return {'ready': ready, 'ready_mode': ready_mode, 'speed': float(forward_speed), 'froude': float(froude), 'xcom_margin': float(xcom_margin), 'double_support': bool(double_support), 'grf_bw': float(grf_bw), 'lateral_err': float(lateral_err), 'yaw_err_deg': float(yaw_err_deg), 'reference_speed_ratio': float(reference_speed_ratio), 'policy_limited': bool(policy_limited)}

def detect_fall_events(imu_data_buffer, dynamics_frames, marker_frames=None, perturb_start_time=None):
    ts = np.asarray(imu_data_buffer.get('timestamp', []), dtype=float)
    heights = np.asarray(imu_data_buffer.get('pelvis_height', []), dtype=float)
    vels = np.asarray(imu_data_buffer.get('pelvis_velocity', []), dtype=float)
    if ts.size == 0 or heights.size == 0:
        return {'available': False}
    if vels.ndim == 1:
        speed = np.abs(vels)
    else:
        speed = np.linalg.norm(vels[:, :2], axis=1)
    trunk = None
    if marker_frames:
        trunk = np.asarray([fr.get('trunk_lean_deg', 0.0) for fr in marker_frames], dtype=float)
    imp_t = np.asarray([fr.get('time', np.nan) for fr in dynamics_frames], dtype=float) if dynamics_frames else np.array([], dtype=float)
    imp_bw = np.asarray([fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0)) / max(1.0, 70.0 * 9.81) for fr in dynamics_frames], dtype=float) if dynamics_frames else np.array([], dtype=float)
    if dynamics_frames:
        body_mass_guess = float(max(1.0, np.median([fr.get('support_vertical_n', 0.0) for fr in dynamics_frames[:min(len(dynamics_frames), 30)] if fr.get('support_vertical_n', 0.0) > 0.0]) / 9.81)) if any((fr.get('support_vertical_n', 0.0) > 0.0 for fr in dynamics_frames[:min(len(dynamics_frames), 30)])) else 70.0
        imp_bw = np.asarray([fr.get('primary_impact_body_load_n_filt', fr.get('primary_impact_body_load_n', 0.0)) / max(1.0, body_mass_guess * 9.81) for fr in dynamics_frames], dtype=float)
    t0 = float(perturb_start_time) if perturb_start_time is not None else float(ts[0])
    i0 = int(np.searchsorted(ts, t0, side='left'))
    pre_h = float(np.median(heights[max(0, i0 - 30):max(i0, i0 + 1)])) if i0 > 0 else float(np.max(heights[:min(len(heights), 30)]))
    onset_idx = None
    for i in range(i0, len(ts)):
        trunk_cond = bool(trunk is not None and i < len(trunk) and (trunk[i] > 22.0))
        h_cond = heights[i] < max(0.78 * pre_h, pre_h - 0.1)
        v_cond = speed[i] > 0.55
        if trunk_cond and h_cond or (h_cond and v_cond):
            onset_idx = i
            break
    if onset_idx is None:
        onset_idx = i0
    impact_idx = None
    if imp_t.size:
        j0 = int(np.searchsorted(imp_t, ts[onset_idx], side='left'))
        for j in range(j0, len(imp_t)):
            if imp_bw[j] > 0.25:
                impact_idx = j
                break
        if impact_idx is None:
            impact_idx = int(np.argmax(imp_bw[j0:]) + j0) if j0 < len(imp_bw) else None
    settle_idx = None
    start_settle = onset_idx
    if impact_idx is not None:
        start_settle = max(start_settle, int(np.searchsorted(ts, imp_t[impact_idx], side='left')))
    streak = 0
    for i in range(start_settle, len(ts)):
        low_h = heights[i] < 0.2
        low_v = speed[i] < 0.08
        if low_h and low_v:
            streak += 1
        else:
            streak = 0
        if streak >= 35:
            settle_idx = i - 34
            break
    impact_time = float(imp_t[impact_idx]) if impact_idx is not None else float(ts[min(len(ts) - 1, onset_idx)])
    onset_time = float(ts[onset_idx])
    settle_time = float(ts[settle_idx]) if settle_idx is not None else float(ts[-1])
    return {'available': True, 'onset_time': onset_time, 'impact_time': impact_time, 'settle_time': settle_time, 'fall_duration_s': max(0.0, settle_time - onset_time), 'lead_time_ms': max(0.0, (impact_time - onset_time) * 1000.0), 'onset_idx': int(onset_idx), 'impact_idx': int(impact_idx) if impact_idx is not None else None, 'settle_idx': int(settle_idx) if settle_idx is not None else None}
from datetime import datetime

# Safe default placeholders for import-time use
SIM_AGE = 75
SIM_HEIGHT = 1.65
SIM_SEX = 'male'
SIM_WEIGHT = None
SIM_RESOLVED_WEIGHT = 70.8
SIM_TARGET_BMI = 26.0
SIM_WEIGHT_SOURCE = 'safe_import_default'
PHASES = {}
TOTAL_STEPS = 0
ACTIVE_FALL_TYPE = 'backward_walking'
