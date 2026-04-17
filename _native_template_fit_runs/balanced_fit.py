
# -*- coding: utf-8 -*-
"""
Native-Template Walk Prototype v82-templatephasewarp

Purpose:
- isolate the locomotion problem from the fall stack
- keep subject anthropometry and physiology
- use a subject-conditioned gait phase machine + tracking controller
- use the Meta Motivo policy only as a residual walk generator

This file is intentionally focused on STAND + WALK only.
"""

import sys
import re
from collections import deque

import numpy as np
import torch
import mujoco
import mujoco.viewer

import humenv
from humenv import make_humenv
from humenv.rewards import LocomotionReward
from metamotivo.fb_cpr.huggingface import FBcprModel

from biofidelic_profile_rebuilt import (
    get_age_style_v2,
    apply_age_effects_v2,
    get_segment_mass_fractions,
    print_subject_profile,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if hasattr(mujoco, 'mjtObj'):
    MJOBJ_BODY = mujoco.mjtObj.mjOBJ_BODY
    MJOBJ_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR
    MJOBJ_GEOM = mujoco.mjtObj.mjOBJ_GEOM
else:
    MJOBJ_BODY = mujoco.mjOBJ_BODY
    MJOBJ_ACTUATOR = mujoco.mjOBJ_ACTUATOR
    MJOBJ_GEOM = mujoco.mjOBJ_GEOM


def _norm_name(s):
    return ''.join(ch.lower() for ch in str(s or '') if ch.isalnum())


def _tokenize_name(s):
    s = str(s or '').replace('-', '_')
    return [tok.lower() for tok in re.split(r'[^A-Za-z0-9]+', s) if tok]


def _body_name_matches_side(name, side):
    norm = _norm_name(name)
    toks = _tokenize_name(name)
    left_hit = any(t in ('l', 'left') for t in toks) or norm.endswith('l') or '_l' in str(name).lower()
    right_hit = any(t in ('r', 'right') for t in toks) or norm.endswith('r') or '_r' in str(name).lower()
    if side == 'left':
        return left_hit and not right_hit
    if side == 'right':
        return right_hit and not left_hit
    return True


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
    return f_world, t_world


def _prompt_float(label, default, unit='', lo=None, hi=None):
    while True:
        raw = input(f"    {label} [{default}{' ' + unit if unit else ''}]: ").strip()
        if raw == '':
            return default
        try:
            val = float(raw)
            if lo is not None and val < lo:
                print(f"      x  Must be >= {lo}. Try again.")
                continue
            if hi is not None and val > hi:
                print(f"      x  Must be <= {hi}. Try again.")
                continue
            return val
        except ValueError:
            print("      x  Please enter a number. Try again.")


def _prompt_str(label, default, choices):
    while True:
        raw = input(f"    {label} [{default}] ({'/'.join(choices)}): ").strip().lower()
        if raw == '':
            return default
        if raw in choices:
            return raw
        print(f"      x  Choose one of: {choices}. Try again.")


def _prompt_optional_float(label, unit='', lo=None, hi=None):
    while True:
        raw = input(f"    {label} [auto{' ' + unit if unit else ''}]: ").strip()
        if raw == '':
            return None
        try:
            val = float(raw)
            if lo is not None and val < lo:
                print(f"      x  Must be >= {lo}. Try again.")
                continue
            if hi is not None and val > hi:
                print(f"      x  Must be <= {hi}. Try again.")
                continue
            return val
        except ValueError:
            print("      x  Please enter a number or press Enter for auto. Try again.")


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
        target_bmi = weight_kg / max(float(height_m) * float(height_m), 1e-9)
        source = 'user_input'
    return float(weight_kg), float(target_bmi), source


class AnthropometricModel:
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
        self.weight = self.default_body_mass if weight is None else float(weight)
        self.height_scale = height / 1.75
        self.mass_scale = self.weight / max(self.default_body_mass, 1e-9)

        self.apply_scaling()
        self.age_params = self.apply_age_effects()

    def _get_fraction(self, body_name):
        if body_name is None:
            return None
        bname = body_name.lower()
        alias_map = [
            ('upperarm', 'upperarm'), ('shoulder', 'upperarm'),
            ('forearm', 'forearm'), ('elbow', 'forearm'),
            ('hand', 'hand'), ('wrist', 'hand'),
            ('abdomen', 'abdomen'), ('thorax', 'thorax'),
            ('pelvis', 'pelvis'), ('trunk', 'trunk'), ('torso', 'torso'), ('spine', 'spine'),
            ('thigh', 'thigh'), ('hip', 'thigh'),
            ('shank', 'shank'), ('knee', 'shank'),
            ('foot', 'foot'), ('ankle', 'foot'), ('toe', 'foot'), ('heel', 'foot'),
            ('head', 'head'), ('neck', 'head'),
        ]
        for token, key in alias_map:
            if token in bname:
                return self.WINTER_MASS_FRACTIONS[key]
        return None

    def _body_scale_vector(self, body_name):
        lname = (body_name or '').lower()
        hs = float(np.clip(self.height / 1.75, 0.90, 1.10))
        if 'pelvis' in lname:
            return np.array([1.0, 1.0 + 0.25 * (hs - 1.0), 1.0], dtype=float)
        if any(k in lname for k in ('torso', 'trunk', 'spine', 'abdomen', 'thorax')):
            return np.array([1.0, 1.0 + 0.35 * (hs - 1.0), 1.0 + 0.55 * (hs - 1.0)], dtype=float)
        if 'head' in lname or 'neck' in lname:
            return np.array([1.0, 1.0, 1.0 + 0.20 * (hs - 1.0)], dtype=float)
        return np.array([hs, hs, hs], dtype=float)

    def apply_scaling(self):
        matched_bodies = 0
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i)
            scale_vec = self._body_scale_vector(name)
            self.mj_model.body_pos[i] = self.original_body_pos[i] * scale_vec
            if self.original_body_ipos is not None:
                self.mj_model.body_ipos[i] = self.original_body_ipos[i] * scale_vec
            frac = self._get_fraction(name)
            orig_m = self.original_body_mass[i]
            if frac is not None and orig_m > 1e-9:
                self.mj_model.body_mass[i] = self.weight * frac
                matched_bodies += 1
            else:
                self.mj_model.body_mass[i] = orig_m * self.mass_scale
            mass_ratio = self.mj_model.body_mass[i] / max(orig_m, 1e-9)
            self.mj_model.body_inertia[i] = self.original_inertia[i] * mass_ratio * (self.height_scale ** 2.0)

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

        total_mass = float(np.sum(self.mj_model.body_mass))
        if total_mass > 1e-9:
            fix = float(self.weight / total_mass)
            self.mj_model.body_mass[:] *= fix
            self.mj_model.body_inertia[:] *= fix
        print(f"      [Anthropometry] custom mass scaling: mass-matched={matched_bodies}/{self.mj_model.nbody} | kinematic-anchor-rebuild={self.mj_model.nbody}/{self.mj_model.nbody}")

    def apply_age_effects(self):
        return apply_age_effects_v2(
            mj_model=self.mj_model,
            age=self.age,
            sex=self.sex,
            height=self.height,
            body_mass_kg=float(np.sum(self.mj_model.body_mass)),
            original_gear=self.original_gear,
        )


def infer_z_stand(model):
    reward_fn = LocomotionReward(move_speed=0.0, move_angle=0, stand_height=1.4)
    env_tmp, _ = make_humenv(task="move-ego-0-0")
    observations, rewards = [], []
    for trial in range(24):
        torch.manual_seed(SEED + trial + 1000)
        z = model.sample_z(1)
        obs, _ = env_tmp.reset()
        for _ in range(70):
            obs_t = torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act = model.act(obs_t, z).squeeze(0).numpy()
            obs, _, term, trunc, _ = env_tmp.step(act)
            r = reward_fn.compute(env_tmp.unwrapped.model, env_tmp.unwrapped.data)
            observations.append(obs['proprio'].copy())
            rewards.append(r)
            if term or trunc:
                break
    env_tmp.close()
    obs_tensor = torch.tensor(np.asarray(observations), dtype=torch.float32)
    rew_tensor = torch.tensor(np.asarray(rewards), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        z_stand = model.reward_inference(obs_tensor, rew_tensor).mean(dim=0, keepdim=True)
    return z_stand


def infer_z_walk(model, target_speed):
    print(f"  Inferring z_walk (walk-only broad search | target_speed={target_speed:.2f} m/s)...")
    reward_fn = LocomotionReward(move_speed=float(np.clip(target_speed, 0.6, 1.6)), move_angle=0, stand_height=1.4)
    env_tmp, _ = make_humenv(task="move-ego-0-2")
    pelvis_id = mujoco.mj_name2id(env_tmp.unwrapped.model, MJOBJ_BODY, 'Pelvis')
    candidates = []
    for sample_idx in range(40):
        torch.manual_seed(SEED + 4000 + sample_idx)
        candidates.append((f"sample={sample_idx+1}", model.sample_z(1)))

    best_z = None
    best_score = -np.inf
    best_diag = None
    for idx, (label, z) in enumerate(candidates):
        obs, _ = env_tmp.reset()
        rewards, xs, ys, vxy = [], [], [], []
        for _ in range(70):
            obs_t = torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act = model.act(obs_t, z).squeeze(0).numpy()
            obs, _, term, trunc, _ = env_tmp.step(act)
            rewards.append(reward_fn.compute(env_tmp.unwrapped.model, env_tmp.unwrapped.data))
            if pelvis_id >= 0:
                p = env_tmp.unwrapped.data.xpos[pelvis_id].copy()
                v = _body_world_velocity(env_tmp.unwrapped.model, env_tmp.unwrapped.data, pelvis_id)
                xs.append(float(p[0])); ys.append(float(p[1])); vxy.append(float(np.linalg.norm(v[:2])))
            if term or trunc:
                break
        mean_r = float(np.mean(rewards)) if rewards else 0.0
        mean_v = float(np.mean(vxy)) if vxy else 0.0
        if len(xs) > 2:
            dxy = np.array([xs[-1]-xs[0], ys[-1]-ys[0]], dtype=float)
            net = float(np.linalg.norm(dxy))
            heading_xy = dxy / max(net, 1e-9)
            steps = np.column_stack([np.diff(xs), np.diff(ys)])
            path = float(np.sum(np.linalg.norm(steps, axis=1))) if steps.size else 0.0
            straight = net / max(path, 1e-6) if path > 1e-6 else 0.0
        else:
            heading_xy = np.array([1.0, 0.0], dtype=float)
            straight = 0.0
        score = 0.55 * mean_r + 0.25 * straight + 0.20 * np.clip(1.0 - abs(mean_v - target_speed)/max(target_speed,0.25), 0.0, 1.0)
        print(f"      z_candidate {idx+1}/{len(candidates)} ({label}) reward={mean_r:.3f} vxy={mean_v:.3f} straight={straight:.3f} score={score:.3f}")
        if score > best_score:
            best_score = score
            best_z = z.clone()
            best_diag = {'mean_vxy': mean_v, 'straight': straight, 'reward': mean_r, 'heading_xy': heading_xy.tolist()}
    env_tmp.close()

    # Extract a native actuator-space gait template from the best latent. This
    # keeps actuator sign conventions and the model's own stable gait family,
    # instead of inventing joint commands from scratch.
    template_env, _ = make_humenv(task="move-ego-0-2")
    obs_tmpl, _ = template_env.reset()
    template_actions = []
    for _ in range(96):
        obs_tensor = torch.tensor(obs_tmpl['proprio'], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act = model.act(obs_tensor, best_z).squeeze(0).numpy()
        template_actions.append(act.copy())
        obs_tmpl, _, term, trunc, _ = template_env.step(act)
        if term or trunc:
            break
    template_env.close()
    template_arr = np.asarray(template_actions, dtype=float)
    if template_arr.shape[0] >= 64:
        template_arr = template_arr[16:80]
    best_diag['template_len'] = int(template_arr.shape[0])
    print(f"      Best z_walk selected (score={best_score:.3f} | reward={best_diag['reward']:.3f} | vxy={best_diag['mean_vxy']:.3f} | template={template_arr.shape[0]} frames)")
    return best_z, best_diag, template_arr


class SubjectGaitFSMController:
    def __init__(self, model, mj_model, mj_data, age_params, age_style):
        self.model = model
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.age_params = dict(age_params)
        self.age_style = dict(age_style)
        self.current_phase = 'stand'
        self.z_stand = None
        self.z_walk = None
        self.pelvis_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Pelvis')
        self.torso_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Torso')
        self.head_id = mujoco.mj_name2id(self.mj_model, MJOBJ_BODY, 'Head')
        self.original_gear = mj_model.actuator_gear[:, 0].copy()

        self.leg_actuators = []
        self.arm_actuators = []
        self.torso_actuators = []
        self.left_leg_actuators = []
        self.right_leg_actuators = []
        self.left_arm_actuators = []
        self.right_arm_actuators = []
        self._resolve_actuators()

        self.ref_fwd = np.array([1.0, 0.0], dtype=float)
        self.ref_lat = np.array([0.0, 1.0], dtype=float)
        self.ref_origin_xy = np.zeros(2, dtype=float)
        self.ref_yaw = 0.0

        self.phase = 0.0
        self.last_phase_time = float(self.mj_data.time)
        self.cadence_hz = 1.8
        self.step_width = 0.10
        self.ds_frac = float(self.age_style.get('expected_double_support', 0.24))
        self.lateral_amp = 0.04
        self.target_speed = float(self.age_style.get('target_walk_speed', 1.0))
        self.forward_speed_ema = 0.0
        self.action_hist = deque(maxlen=3)
        self.reanchor_interval_s = 0.80
        self.last_reanchor_time = float(self.mj_data.time)
        self.diag_heading_xy = np.array([1.0, 0.0], dtype=float)
        self.body_forward_axis_local = np.array([1.0, 0.0, 0.0], dtype=float)
        self.forward_speed_ema_signed = 0.0
        self.forward_lock_count = 0
        self.forward_locked = False
        self.travel_dir_ema = np.array([1.0, 0.0], dtype=float)
        self.yaw_calibration_active = False
        self.yaw_calibration_steps = 0
        self.yaw_calibration_needed = 40
        self.body_axis_calibrated = False
        self.axis_candidates = [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([-1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, -1.0, 0.0], dtype=float),
        ]
        self.axis_score_sum = np.zeros(4, dtype=float)
        self.axis_score_count = np.zeros(4, dtype=float)
        self.last_phase_sector = 0

        self.walk_metrics = []
        self.gait_template = None
        self.gait_template_len = 0
        self.template_phase_gain = 1.0

    def _resolve_actuators(self):
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_ACTUATOR, i) or ''
            lname = name.lower()
            if any(k in lname for k in ('hip', 'knee', 'ankle', 'foot', 'leg')):
                self.leg_actuators.append(i)
                if _body_name_matches_side(name, 'left'):
                    self.left_leg_actuators.append(i)
                elif _body_name_matches_side(name, 'right'):
                    self.right_leg_actuators.append(i)
            elif any(k in lname for k in ('shoulder', 'elbow', 'wrist', 'hand', 'arm')):
                self.arm_actuators.append(i)
                if _body_name_matches_side(name, 'left'):
                    self.left_arm_actuators.append(i)
                elif _body_name_matches_side(name, 'right'):
                    self.right_arm_actuators.append(i)
            elif any(k in lname for k in ('torso', 'spine', 'abdomen', 'chest')):
                self.torso_actuators.append(i)

    def set_latents(self, z_stand, z_walk, z_walk_diag=None, gait_template=None):
        self.z_stand = z_stand
        self.z_walk = z_walk
        if z_walk_diag and 'heading_xy' in z_walk_diag:
            h = np.asarray(z_walk_diag['heading_xy'], dtype=float)
            hn = float(np.linalg.norm(h))
            if hn > 1e-6:
                self.diag_heading_xy = h / hn
        if gait_template is not None:
            arr = np.asarray(gait_template, dtype=float)
            self.gait_template = arr.copy()
            self.gait_template_len = int(arr.shape[0])

    def start_walk_phase(self, natural_speed):
        self.current_phase = 'walk'
        if self.pelvis_id >= 0:
            pos = self.mj_data.xpos[self.pelvis_id].copy()
            self.ref_origin_xy = np.asarray(pos[:2], dtype=float)
            h = np.asarray(self.diag_heading_xy, dtype=float)
            hn = float(np.linalg.norm(h))
            if hn > 1e-6:
                self.ref_fwd = h / hn
                self.ref_yaw = float(np.arctan2(self.ref_fwd[1], self.ref_fwd[0]))
            else:
                R = self.mj_data.xmat[self.pelvis_id].reshape(3, 3)
                fwd = np.asarray(R[:, 0], dtype=float)
                yaw = float(np.arctan2(fwd[1], fwd[0]))
                self.ref_fwd = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
                self.ref_yaw = yaw

            # Initialize online body-heading calibration. We start with the
            # coarse axis from the instantaneous pose, but we do not trust it
            # for yaw control until enough walking frames have been observed.
            R = self.mj_data.xmat[self.pelvis_id].reshape(3, 3)
            best_axis = self.axis_candidates[0]
            best_dot = -1e9
            for ax in self.axis_candidates:
                world_ax = R @ ax
                dotv = float(np.dot(world_ax[:2], self.ref_fwd))
                if dotv > best_dot:
                    best_dot = dotv
                    best_axis = ax
            self.body_forward_axis_local = best_axis.copy()
            self.body_axis_calibrated = False
            self.yaw_calibration_active = True
            self.yaw_calibration_steps = 0
            self.axis_score_sum[:] = 0.0
            self.axis_score_count[:] = 0.0
            self.ref_lat = np.array([-self.ref_fwd[1], self.ref_fwd[0]], dtype=float)
        leg_length = float(max(0.55, self.age_params.get('leg_length', 0.93)))
        balance_imp = float(np.clip(self.age_params.get('balance_impairment', 0.0), 0.0, 0.8))
        desired_speed = float(self.age_style.get('target_walk_speed', natural_speed))
        # stability-first: stay near what the prior can actually realize
        self.target_speed = float(np.clip(desired_speed, 1.00 * natural_speed, 1.24 * natural_speed))
        step_length = float(np.clip(0.34 * leg_length + 0.24 * self.target_speed, 0.28 * leg_length, 0.90 * leg_length))
        cadence_spm = float(np.clip(60.0 * self.target_speed / max(step_length, 0.12), 85.0, 135.0))
        self.cadence_hz = cadence_spm / 60.0
        self.ds_frac = float(np.clip(self.age_style.get('expected_double_support', 0.24), 0.16, 0.40))
        self.step_width = float(np.clip(0.09 + 0.04 * balance_imp, 0.08, 0.15))
        self.lateral_amp = 0.24 * self.step_width
        self.phase = 0.0
        self.last_phase_time = float(self.mj_data.time)
        self.last_reanchor_time = float(self.mj_data.time)
        self.forward_speed_ema = 0.0
        self.forward_speed_ema_signed = 0.0
        self.forward_lock_count = 0
        self.forward_locked = False
        self.travel_dir_ema = self.ref_fwd.copy()
        self.last_phase_sector = 0
        self.action_hist.clear()
        self.walk_metrics.clear()
        if self.gait_template_len > 0 and natural_speed > 1e-6:
            self.template_phase_gain = float(np.clip(self.target_speed / natural_speed, 0.90, 1.18))
        else:
            self.template_phase_gain = 1.0

    def _update_phase(self):
        now = float(self.mj_data.time)
        dt = max(1e-4, now - float(self.last_phase_time))
        self.last_phase_time = now
        self.phase = float((self.phase + 2.0 * np.pi * self.cadence_hz * dt) % (2.0 * np.pi))

    def _foot_support_bw(self):
        left_bw = 0.0
        right_bw = 0.0
        total_bw = 0.0
        bw = max(float(np.sum(self.mj_model.body_mass)) * 9.81, 1.0)
        for i in range(int(self.mj_data.ncon)):
            c = self.mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            names = ((mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g1) or '').lower(),
                     (mujoco.mj_id2name(self.mj_model, MJOBJ_GEOM, g2) or '').lower())
            ground = any(('floor' in n or 'ground' in n or 'plane' in n) for n in names) or any(
                self.mj_model.geom_type[g] == getattr(getattr(mujoco, 'mjtGeom', object), 'mjGEOM_PLANE', -999)
                for g in (g1, g2)
            )
            if not ground:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, wrench)
            if wrench[0] <= 1.0:
                continue
            f_world, _ = _contact_wrench_world(c, wrench)
            fz = max(0.0, float(f_world[2]))
            total_bw += fz / bw
            ng = g2 if ('floor' in names[0] or 'ground' in names[0] or 'plane' in names[0]) else g1
            bid = int(self.mj_model.geom_bodyid[ng])
            bname = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, bid) or ''
            if any(k in bname.lower() for k in ('foot', 'ankle', 'toe', 'heel')):
                if _body_name_matches_side(bname, 'left'):
                    left_bw += fz / bw
                elif _body_name_matches_side(bname, 'right'):
                    right_bw += fz / bw
        return left_bw, right_bw, total_bw

    def _compute_xcom_margin(self):
        if self.pelvis_id < 0:
            return -1.0
        com_pos = self.mj_data.xpos[self.pelvis_id]
        com_vel = _body_world_velocity(self.mj_model, self.mj_data, self.pelvis_id)
        g = 9.81
        leg_length = max(0.1, float(self.age_params.get('leg_length', 0.93)))
        omega = np.sqrt(g / leg_length)
        xcom = com_pos[:2] + np.asarray(com_vel[:2]) / omega
        # crude support center from foot/ankle bodies
        pts = []
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, MJOBJ_BODY, i) or ''
            if any(k in name.lower() for k in ('foot', 'ankle', 'toe', 'heel')):
                pts.append(np.asarray(self.mj_data.xpos[i][:2], dtype=float))
        if not pts:
            return -1.0
        arr = np.vstack(pts)
        min_xy = np.min(arr, axis=0) - np.array([0.06, 0.04])
        max_xy = np.max(arr, axis=0) + np.array([0.06, 0.04])
        dx_out = max(min_xy[0] - xcom[0], 0.0, xcom[0] - max_xy[0])
        dy_out = max(min_xy[1] - xcom[1], 0.0, xcom[1] - max_xy[1])
        if dx_out == 0.0 and dy_out == 0.0:
            margin = min(xcom[0] - min_xy[0], max_xy[0] - xcom[0], xcom[1] - min_xy[1], max_xy[1] - xcom[1])
        else:
            margin = -float(np.hypot(dx_out, dy_out))
        return float(margin)

    def _apply_stoop_bias(self):
        if self.torso_id < 0 or self.pelvis_id < 0 or self.head_id < 0:
            return
        target_deg = float(self.age_style.get('walk_stoop_target_deg', 6.0))
        pelvis = self.mj_data.xpos[self.pelvis_id].copy()
        head = self.mj_data.xpos[self.head_id].copy()
        vec = np.asarray(head - pelvis, dtype=float)
        up_comp = float(vec[2])
        fwd_comp = float(np.dot(vec[:2], self.ref_fwd))
        current_deg = float(np.degrees(np.arctan2(fwd_comp, max(abs(up_comp), 1e-6))))
        stoop_err = float(target_deg - current_deg)
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, self.pelvis_id, vel6, 0)
        ang = vel6[:3].copy()
        lat_axis = np.array([self.ref_lat[0], self.ref_lat[1], 0.0], dtype=float)
        pitch_rate = float(np.dot(ang, lat_axis))
        kp, kd = 0.24, 0.13
        pitch_torque = float(np.clip(kp * stoop_err - kd * pitch_rate, -4.0, 4.0))
        self.mj_data.xfrc_applied[self.torso_id, 3] += pitch_torque * float(self.ref_lat[0])
        self.mj_data.xfrc_applied[self.torso_id, 4] += pitch_torque * float(self.ref_lat[1])



    def _phase_warp(self, phase_u):
        # Warp the native template so double-support windows are compressed
        # toward the subject target while single-support occupies more of the
        # cycle. This is structural template editing, not gain tweaking.
        native_ds = 0.38
        target_ds = float(np.clip(self.ds_frac, 0.16, 0.38))
        ds_n = 0.5 * native_ds
        ds_t = 0.5 * target_ds
        half = (phase_u * 2.0) % 1.0
        in_second = phase_u >= 0.5
        if half < ds_t:
            warped = half * ds_n / max(ds_t, 1e-6)
        elif half > 1.0 - ds_t:
            warped = 1.0 - (1.0 - half) * ds_n / max(ds_t, 1e-6)
        else:
            warped = ds_n + (half - ds_t) * (1.0 - 2.0 * ds_n) / max(1.0 - 2.0 * ds_t, 1e-6)
        return 0.5 * warped + (0.5 if in_second else 0.0)

    def _template_action(self):
        if self.gait_template is None or self.gait_template_len <= 1:
            return np.zeros(self.mj_model.nu, dtype=float)
        phase_u = float((self.phase / (2.0 * np.pi)) * self.template_phase_gain) % 1.0
        phase_u = self._phase_warp(phase_u)
        idxf = phase_u * self.gait_template_len
        i0 = int(np.floor(idxf)) % self.gait_template_len
        i1 = (i0 + 1) % self.gait_template_len
        w = float(idxf - np.floor(idxf))
        return (1.0 - w) * self.gait_template[i0] + w * self.gait_template[i1]

    def get_action(self, obs):
        z = self.z_stand if self.current_phase == 'stand' else self.z_walk
        obs_t = torch.tensor(obs['proprio'], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.model.act(obs_t, z).squeeze(0).numpy()

        if self.current_phase == 'stand':
            return action

        self._update_phase()

        # residual phase modulation
        left_stance = float(0.5 * (1.0 + np.sin(self.phase)))
        right_stance = float(0.5 * (1.0 - np.sin(self.phase)))
        # Stronger stance/swing asymmetry to encourage real single-support
        # stepping instead of a cautious shuffle with both feet mostly loaded.
        left_leg_gain = 0.88 + 0.22 * left_stance
        right_leg_gain = 0.88 + 0.22 * right_stance
        left_arm_gain = float(self.age_style.get('arm_gain', 1.0)) * (0.90 + 0.16 * right_stance)
        right_arm_gain = float(self.age_style.get('arm_gain', 1.0)) * (0.90 + 0.16 * left_stance)

        if self.left_leg_actuators:
            action[self.left_leg_actuators] *= left_leg_gain
        if self.right_leg_actuators:
            action[self.right_leg_actuators] *= right_leg_gain
        if self.left_arm_actuators:
            action[self.left_arm_actuators] *= left_arm_gain
        if self.right_arm_actuators:
            action[self.right_arm_actuators] *= right_arm_gain

        # local gait corridor guidance
        pos = self.mj_data.xpos[self.pelvis_id].copy()
        vel = _body_world_velocity(self.mj_model, self.mj_data, self.pelvis_id)
        pos_xy = np.asarray(pos[:2], dtype=float)
        vel_xy = np.asarray(vel[:2], dtype=float)

        rel_xy = pos_xy - np.asarray(self.ref_origin_xy, dtype=float)
        forward_speed = float(np.dot(vel_xy, self.ref_fwd))
        lateral_speed = float(np.dot(vel_xy, self.ref_lat))

        # Early heading lock: if the chosen forward axis is backwards relative
        # to the actual latent walk, flip it once instead of fighting the policy.
        self.forward_speed_ema_signed = 0.88 * float(self.forward_speed_ema_signed) + 0.12 * forward_speed
        if not self.forward_locked:
            self.forward_lock_count += 1
            if self.forward_lock_count >= 12:
                if self.forward_speed_ema_signed < -0.03:
                    self.ref_fwd = -self.ref_fwd
                    self.ref_lat = -self.ref_lat
                    self.ref_yaw = float(np.arctan2(self.ref_fwd[1], self.ref_fwd[0]))
                    forward_speed = -forward_speed
                    lateral_speed = -lateral_speed
                self.forward_locked = True

        # Track recent actual travel direction so heading control stabilizes the
        # walk instead of clinging to one fixed initial yaw forever.
        speed_xy = float(np.linalg.norm(vel_xy))
        if speed_xy > 0.20:
            inst_dir = vel_xy / max(speed_xy, 1e-9)
            self.travel_dir_ema = 0.92 * self.travel_dir_ema + 0.08 * inst_dir
            tn = float(np.linalg.norm(self.travel_dir_ema))
            if tn > 1e-6:
                self.travel_dir_ema = self.travel_dir_ema / tn

        # Re-anchor corridor every half gait cycle so lateral error stays local
        # to the current step instead of growing for the whole walk.
        phase_sector = int(self.phase // np.pi)
        if phase_sector != self.last_phase_sector:
            forward_progress = float(np.dot(pos_xy - np.asarray(self.ref_origin_xy, dtype=float), self.ref_fwd))
            self.ref_origin_xy = np.asarray(self.ref_origin_xy, dtype=float) + forward_progress * self.ref_fwd
            self.last_phase_sector = phase_sector
        desired_lateral = float(self.lateral_amp * np.sin(self.phase))
        lateral_pos = float(np.dot(rel_xy, self.ref_lat))
        lateral_err = lateral_pos - desired_lateral
        desired_speed = float(self.target_speed * (0.96 + 0.04 * np.cos(2.0 * self.phase)))
        speed_err = desired_speed - forward_speed
        self.forward_speed_ema = 0.90 * float(self.forward_speed_ema) + 0.10 * max(forward_speed, 0.0)
        leg_boost = float(np.clip(1.0 + 0.16 * max(desired_speed - self.forward_speed_ema, 0.0), 0.97, 1.20))
        if self.leg_actuators:
            action[self.leg_actuators] *= leg_boost

        # Gentle corridor-heading adaptation toward actual travel direction.
        self.ref_fwd = 0.93 * self.ref_fwd + 0.07 * self.travel_dir_ema
        fn = float(np.linalg.norm(self.ref_fwd))
        if fn > 1e-6:
            self.ref_fwd = self.ref_fwd / fn
        self.ref_lat = np.array([-self.ref_fwd[1], self.ref_fwd[0]], dtype=float)
        self.ref_yaw = float(np.arctan2(self.ref_fwd[1], self.ref_fwd[0]))

        R = self.mj_data.xmat[self.pelvis_id].reshape(3, 3)
        if self.yaw_calibration_active and speed_xy > 0.25:
            for i, ax in enumerate(self.axis_candidates):
                world_ax = np.asarray(R @ ax, dtype=float)
                ax_xy = world_ax[:2]
                an = float(np.linalg.norm(ax_xy))
                if an > 1e-9:
                    ax_xy = ax_xy / an
                    err = float(np.arctan2(ax_xy[0] * self.travel_dir_ema[1] - ax_xy[1] * self.travel_dir_ema[0],
                                           np.dot(ax_xy, self.travel_dir_ema)))
                    self.axis_score_sum[i] += abs(err)
                    self.axis_score_count[i] += 1.0
            self.yaw_calibration_steps += 1
            enough = self.yaw_calibration_steps >= self.yaw_calibration_needed and np.max(self.axis_score_count) >= 8
            if enough:
                mean_err = np.where(self.axis_score_count > 0, self.axis_score_sum / np.maximum(self.axis_score_count, 1.0), 1e9)
                best_i = int(np.argmin(mean_err))
                self.body_forward_axis_local = self.axis_candidates[best_i].copy()
                self.body_axis_calibrated = True
                self.yaw_calibration_active = False

        fwd_body = np.asarray(R @ self.body_forward_axis_local, dtype=float)
        yaw = float(np.arctan2(fwd_body[1], fwd_body[0]))
        if self.body_axis_calibrated:
            yaw_err = float(np.arctan2(np.sin(yaw - self.ref_yaw), np.cos(yaw - self.ref_yaw)))
        else:
            yaw_err = 0.0
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, MJOBJ_BODY, self.pelvis_id, vel6, 0)
        ang = vel6[:3].copy()

        fx = float(np.clip(5.4 * speed_err, -9.0, 9.0))
        fy = float(np.clip(-5.4 * lateral_err - 6.2 * lateral_speed, -6.0, 6.0))
        if self.body_axis_calibrated:
            tz = float(np.clip(-4.6 * yaw_err - 1.3 * float(ang[2]), -5.0, 5.0))
        else:
            tz = float(np.clip(-0.9 * float(ang[2]), -2.0, 2.0))

        if self.body_axis_calibrated and abs(np.degrees(yaw_err)) > 24.0:
            fx *= 0.60
            fy *= 0.50
            tz *= 0.50

        f_world = fx * self.ref_fwd + fy * self.ref_lat
        self.mj_data.xfrc_applied[self.pelvis_id, 0] = float(f_world[0])
        self.mj_data.xfrc_applied[self.pelvis_id, 1] = float(f_world[1])
        self.mj_data.xfrc_applied[self.pelvis_id, 5] = float(tz)
        self._apply_stoop_bias()

        left_bw, right_bw, total_bw = self._foot_support_bw()
        ds = bool(left_bw > 0.12 and right_bw > 0.12)
        xcom_margin = self._compute_xcom_margin()
        self.walk_metrics.append({
            'time': float(self.mj_data.time),
            'forward_speed': forward_speed,
            'lateral_err': lateral_err,
            'yaw_err_deg': float(np.degrees(yaw_err)),
            'grf_bw': total_bw,
            'double_support': ds,
            'xcom_margin': xcom_margin,
            'yaw_calibrated': bool(self.body_axis_calibrated),
        })

        self.action_hist.append(action.copy())
        if len(self.action_hist) >= 2:
            recent = np.asarray(list(self.action_hist)[-2:], dtype=float)
            action = 0.75 * recent[-1] + 0.25 * recent[-2]
        return action

    def gait_report(self):
        if not self.walk_metrics:
            return {}
        spd = np.asarray([m['forward_speed'] for m in self.walk_metrics], dtype=float)
        lat = np.asarray([m['lateral_err'] for m in self.walk_metrics], dtype=float)
        yaw = np.asarray([m['yaw_err_deg'] for m in self.walk_metrics], dtype=float)
        grf = np.asarray([m['grf_bw'] for m in self.walk_metrics], dtype=float)
        ds = np.asarray([1.0 if m['double_support'] else 0.0 for m in self.walk_metrics], dtype=float)
        xcom = np.asarray([m['xcom_margin'] for m in self.walk_metrics], dtype=float)
        return {
            'mean_speed': float(np.mean(np.abs(spd))),
            'mean_lateral_err': float(np.mean(np.abs(lat))),
            'mean_yaw_err_deg': float(np.mean(np.abs(yaw))),
            'peak_grf_bw': float(np.max(grf)),
            'mean_ds': float(np.mean(ds)),
            'xcom_negative_frac': float(np.mean(xcom < 0.0)),
        }


print("=" * 70)
print("  Native-Template Walk Prototype v82-templatephasewarp")
print("  Seed: 42 | Focus: STAND + WALK only")
print("=" * 70)
print("\n  Enter person parameters (press Enter to use default):\n")

SIM_AGE = int(_prompt_float("Age", 75, "years", lo=1, hi=120))
SIM_HEIGHT = _prompt_float("Height", 1.65, "m", lo=0.5, hi=2.5)
SIM_SEX = _prompt_str("Sex", "male", ["male", "female"])
SIM_WEIGHT = _prompt_optional_float("Weight", "kg", lo=25.0, hi=250.0)
SIM_RESOLVED_WEIGHT, SIM_TARGET_BMI, SIM_WEIGHT_SOURCE = resolve_subject_weight_kg(
    SIM_HEIGHT, SIM_AGE, SIM_SEX, explicit_weight=SIM_WEIGHT
)
SIM_WEIGHT_DISPLAY = (
    f"{SIM_WEIGHT:.1f}kg (user)" if SIM_WEIGHT is not None
    else f"{SIM_RESOLVED_WEIGHT:.1f}kg (auto BMI {SIM_TARGET_BMI:.1f})"
)

print(f"\n  >> Using: age={SIM_AGE}yr  height={SIM_HEIGHT}m  weight={SIM_WEIGHT_DISPLAY}  sex={SIM_SEX}\n")
print_subject_profile(SIM_AGE, SIM_SEX, SIM_HEIGHT, SIM_RESOLVED_WEIGHT)

print("\n[1/3] Loading Meta Motivo model...")
model = FBcprModel.from_pretrained("facebook/metamotivo-M-1")
model.eval()
device = next(model.parameters()).device
print(f"      Model loaded on {device}")

print("\n[2/3] Computing task embeddings...")
age_style = get_age_style_v2(SIM_AGE, SIM_HEIGHT, SIM_SEX, SIM_RESOLVED_WEIGHT)
z_stand = infer_z_stand(model)
z_walk, z_diag, gait_template = infer_z_walk(model, age_style['target_walk_speed'])

print("\n[3/3] Initializing walk-only environment...")
env, _ = make_humenv(task="move-ego-0-0")
obs, _ = env.reset()
mj_model = env.unwrapped.model
mj_data = env.unwrapped.data

anthro = AnthropometricModel(mj_model, age=SIM_AGE, height=SIM_HEIGHT, weight=SIM_RESOLVED_WEIGHT, sex=SIM_SEX)
age_params = anthro.age_params

controller = SubjectGaitFSMController(model, mj_model, mj_data, age_params, age_style)
controller.set_latents(z_stand, z_walk, z_diag)

STAND_STEPS = 180
WALK_STEPS = 420
TOTAL_STEPS = STAND_STEPS + WALK_STEPS

print("\n" + "=" * 70)
print("  RUNNING NATIVE-TEMPLATE WALK PROTOTYPE")
print("=" * 70)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.cam.distance = 4.5
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 90

    for step in range(TOTAL_STEPS):
        mj_data.xfrc_applied[:] = 0.0
        if step == STAND_STEPS:
            controller.start_walk_phase(float(z_diag['mean_vxy']))
            env.unwrapped.set_task("move-ego-0-2")
            print(f"  [Step {step}] Walking phase started | natural_speed={z_diag['mean_vxy']:.2f} m/s | controller_target={controller.target_speed:.2f} m/s")

        controller.current_phase = 'stand' if step < STAND_STEPS else 'walk'
        action = controller.get_action(obs)
        obs, _, term, trunc, _ = env.step(action)
        mujoco.mj_forward(mj_model, mj_data)

        if step % 30 == 0:
            pelvis_h = float(mj_data.xpos[controller.pelvis_id][2]) if controller.pelvis_id >= 0 else 0.0
            vel = _body_world_velocity(mj_model, mj_data, controller.pelvis_id) if controller.pelvis_id >= 0 else np.zeros(3)
            fwd_speed = float(np.dot(np.asarray(vel[:2], dtype=float), controller.ref_fwd)) if step >= STAND_STEPS else 0.0
            left_bw, right_bw, total_bw = controller._foot_support_bw()
            ds = bool(left_bw > 0.12 and right_bw > 0.12)
            xcom_margin = controller._compute_xcom_margin() if step >= STAND_STEPS else 0.0
            print(f"    {step:4d}  phase={'stand' if step < STAND_STEPS else 'walk ':5s}  h={pelvis_h:.3f}  fwd={fwd_speed:+.3f}  DS={'YES' if ds else ' no'}  GRF/BW={total_bw:.2f}  XCoM={xcom_margin:+.3f}")

        viewer.sync()
        if term or trunc:
            break

env.close()

rep = controller.gait_report()
print("\n" + "=" * 70)
print("  NATIVE-TEMPLATE WALK REPORT")
print("=" * 70)
print(f"  Requested subject speed : {age_style['target_walk_speed']:.2f} m/s")
print(f"  Natural z_walk speed    : {z_diag['mean_vxy']:.2f} m/s")
print(f"  Controller walk target  : {controller.target_speed:.2f} m/s")
print(f"  Mean realized speed     : {rep.get('mean_speed', 0.0):.2f} m/s")
print(f"  Mean double support     : {rep.get('mean_ds', 0.0):.1%}")
print(f"  Mean |lateral err|      : {rep.get('mean_lateral_err', 0.0):.3f} m")
print(f"  Mean |yaw err|          : {rep.get('mean_yaw_err_deg', 0.0):.1f} deg")
print(f"  Peak GRF/BW             : {rep.get('peak_grf_bw', 0.0):.2f}")
print(f"  XCoM negative fraction  : {rep.get('xcom_negative_frac', 0.0):.1%}")
print("=" * 70)
