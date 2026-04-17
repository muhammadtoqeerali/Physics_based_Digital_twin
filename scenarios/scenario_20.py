# -*- coding: utf-8 -*-
"""
scenario_20.py  -  COMPLETELY SELF-CONTAINED (no fall_core, no backward_fall_walking_best)
===========================================================================================
Scenario 20: Forward fall when trying to sit down.

Biomechanics
------------
  The subject stands still for ~5 s, then attempts to lower onto a chair.
  During the controlled lowering phase the CoM passes anterior to the base of
  support.  Quadriceps eccentric control fails; the subject tips forward at
  ~45-70° trunk lean and cannot recover (Cress et al. 2000; Schultz 1992).

Phase sequence  (all timings at 30 Hz native):
  STAND    150 steps  5.0 s  - stabilise upright with z_stand embedding
  SIT_DOWN  90 steps  3.0 s  - guided downward force lowers pelvis to ~0.50 m;
                               leg gears decay to 40%; blend toward z_fall
  COLLAPSE  30 steps  1.0 s  - forward topple force (0.65 BW ramped);
                               all muscles weaken toward min_gear
  FALL     400 steps 13.3 s  - free forward fall; prone landing; settle

IMPORTANT - safe imports only:
  numpy, torch, mujoco, mujoco.viewer, humenv, metamotivo
  biofidelic_profile  (has no module-level execution)
  Does NOT import backward_fall_walking_best or fall_core or fall_scenario_library.
"""
# --- stdlib -------------------------------------------------------------------
import sys, os, csv, math
from datetime import datetime
from collections import deque

# --- numerical / ML -----------------------------------------------------------
import numpy as np
import torch

# --- MuJoCo -------------------------------------------------------------------
import mujoco
import mujoco.viewer

# --- simulation environment + model -------------------------------------------
from humenv import make_humenv
from humenv.rewards import LocomotionReward
from metamotivo.fb_cpr.huggingface import FBcprModel

# --- project-local SAFE imports (no module-level execution) -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from biofidelic_profile import (
    muscle_strength_factor,
    reaction_delay_seconds,
    balance_impairment,
    weakening_config,
    print_subject_profile,
    get_age_style_v2,
)

# --- MuJoCo API constants -----------------------------------------------------
MJOBJ_BODY     = mujoco.mjtObj.mjOBJ_BODY
MJOBJ_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR
MJOBJ_GEOM     = mujoco.mjtObj.mjOBJ_GEOM

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
_EMBED_CACHE_FILE = os.path.join(_ROOT, "scenario20_embed_cache.pt")

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _resolve_weight(height, age, sex, explicit=None):
    if explicit is not None:
        w = float(explicit)
        return w, w / max(height**2, 1e-9), "user"
    bmi = {"young": 23.0, "mid": 24.5, "old": 26.0}[
        "young" if age < 40 else "old" if age >= 65 else "mid"
    ]
    if sex == "female":
        bmi -= 1.0
    w = bmi * height**2
    return float(w), float(bmi), "auto"


def _body_id(mj_model, *names):
    for n in names:
        bid = mujoco.mj_name2id(mj_model, MJOBJ_BODY, n)
        if bid >= 0:
            return bid
    return -1


def _body_vel_world(mj_model, mj_data, body_id):
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, body_id, vel6, 0)
    return vel6[3:].copy()


def _infer_anatomical_ref_xy(mj_model, mj_data, pelvis_id):
    """Estimate avatar anatomical forward in world XY from lower-limb anatomy."""
    pair_sets = [
        (('L_Toe','ToeL','LeftToe'), ('L_Foot','FootL','LeftFoot','L_Ankle','AnkleL','LeftAnkle')),
        (('R_Toe','ToeR','RightToe'), ('R_Foot','FootR','RightFoot','R_Ankle','AnkleR','RightAnkle')),
        (('L_Foot','FootL','LeftFoot'), ('L_Ankle','AnkleL','LeftAnkle')),
        (('R_Foot','FootR','RightFoot'), ('R_Ankle','AnkleR','RightAnkle')),
    ]
    vecs = []
    for distal_names, prox_names in pair_sets:
        distal = _body_id(mj_model, *distal_names)
        prox   = _body_id(mj_model, *prox_names)
        if distal >= 0 and prox >= 0:
            v = np.array(mj_data.xpos[distal][:2] - mj_data.xpos[prox][:2], dtype=float)
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                vecs.append(v / n)
    if vecs:
        ref = np.mean(np.vstack(vecs), axis=0)
        n = float(np.linalg.norm(ref))
        if n > 1e-6:
            return ref / n
    if pelvis_id >= 0:
        pelvis_xy = np.array(mj_data.xpos[pelvis_id][:2], dtype=float)
        refs = []
        for nm in ('L_Toe','R_Toe','ToeL','ToeR','LeftToe','RightToe',
                   'L_Foot','R_Foot','FootL','FootR','LeftFoot','RightFoot'):
            bid = _body_id(mj_model, nm)
            if bid >= 0:
                v = np.array(mj_data.xpos[bid][:2], dtype=float) - pelvis_xy
                n = float(np.linalg.norm(v))
                if n > 1e-6:
                    refs.append(v / n)
        if refs:
            ref = np.mean(np.vstack(refs), axis=0)
            n = float(np.linalg.norm(ref))
            if n > 1e-6:
                return ref / n
    return np.array([1.0, 0.0], dtype=float)


def _mean_landmark_pos(mj_model, mj_data, names):
    pts = []
    for nm in names:
        bid = _body_id(mj_model, nm)
        if bid >= 0:
            pts.append(np.array(mj_data.xpos[bid], dtype=float))
    if not pts:
        return None
    return np.mean(np.vstack(pts), axis=0)


def _infer_anatomical_frame(mj_model, mj_data, pelvis_id, head_id):
    """Infer forward/lateral/up from anatomical landmarks, not body-local axes.

    Forward is obtained from a right-handed anatomical frame:
      forward ~= cross(lateral(right-left), up(head-pelvis))
    The sign is aligned to the lower-limb/toe forward reference in world XY.
    """
    ref_xy = _infer_anatomical_ref_xy(mj_model, mj_data, pelvis_id)

    if pelvis_id >= 0 and head_id >= 0:
        up = np.array(mj_data.xpos[head_id] - mj_data.xpos[pelvis_id], dtype=float)
    else:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    nup = float(np.linalg.norm(up))
    up = up / nup if nup > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=float)

    left = _mean_landmark_pos(mj_model, mj_data, [
        'L_Shoulder','LeftShoulder','ShoulderL','L_UpperArm','UpperArmL','LeftUpperArm',
        'L_Arm','ArmL','L_Elbow','ElbowL','LeftElbow','L_Hand','HandL','LeftHand',
        'L_Hip','HipL','LeftHip','L_Thigh','ThighL','LeftThigh'
    ])
    right = _mean_landmark_pos(mj_model, mj_data, [
        'R_Shoulder','RightShoulder','ShoulderR','R_UpperArm','UpperArmR','RightUpperArm',
        'R_Arm','ArmR','R_Elbow','ElbowR','RightElbow','R_Hand','HandR','RightHand',
        'R_Hip','HipR','RightHip','R_Thigh','ThighR','RightThigh'
    ])
    if left is not None and right is not None:
        lateral = np.array(right - left, dtype=float)
    else:
        lateral = np.array([-ref_xy[1], ref_xy[0], 0.0], dtype=float)

    # Remove vertical component and normalize.
    lateral = lateral - up * float(np.dot(lateral, up))
    nl = float(np.linalg.norm(lateral))
    lateral = lateral / nl if nl > 1e-8 else np.array([-ref_xy[1], ref_xy[0], 0.0], dtype=float)

    fwd3 = np.cross(lateral, up)
    nf = float(np.linalg.norm(fwd3))
    if nf < 1e-8:
        fwd3 = np.array([ref_xy[0], ref_xy[1], 0.0], dtype=float)
    else:
        fwd3 /= nf

    # Align sign with toe/foot-derived forward reference.
    if float(np.dot(fwd3[:2], ref_xy)) < 0.0:
        lateral = -lateral
        fwd3 = -fwd3

    f2 = np.array(fwd3[:2], dtype=float)
    n2 = float(np.linalg.norm(f2))
    f2 = f2 / n2 if n2 > 1e-8 else ref_xy
    yaw = float(np.degrees(np.arctan2(f2[1], f2[0])))
    return f2, lateral, up, yaw


def _avatar_forward_xy(mj_model, mj_data, pelvis_id, head_id=None):
    """Return avatar anatomical forward/lateral/up in world coordinates."""
    if pelvis_id < 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0,0.0,1.0]), 0.0
    if head_id is None:
        head_id = _body_id(mj_model, 'Head','head')
    return _infer_anatomical_frame(mj_model, mj_data, pelvis_id, head_id)


def _get_actuator_groups(mj_model):
    leg_kw   = {"hip", "knee", "ankle", "foot", "leg"}
    arm_kw   = {"shoulder", "elbow", "wrist", "hand", "arm"}
    torso_kw = {"torso", "spine", "abdomen", "chest"}
    g = {"leg": [], "arm": [], "torso": [], "other": []}
    for i in range(mj_model.nu):
        n = (mujoco.mj_id2name(mj_model, MJOBJ_ACTUATOR, i) or "").lower()
        if   any(k in n for k in leg_kw):   g["leg"].append(i)
        elif any(k in n for k in arm_kw):   g["arm"].append(i)
        elif any(k in n for k in torso_kw): g["torso"].append(i)
        else:                               g["other"].append(i)
    return g


def _apply_anthropometry(mj_model, age, height, weight, sex):
    """Winter (1990) segment mass fractions + age-based gear scaling."""
    fracs_m = {"head":0.0694,"trunk":0.4346,"torso":0.4346,"pelvis":0.1422,
               "upperarm":0.0271,"forearm":0.0162,"hand":0.0061,
               "thigh":0.1000,"shank":0.0465,"foot":0.0145}
    fracs_f = {"head":0.0668,"trunk":0.4257,"torso":0.4257,"pelvis":0.1247,
               "upperarm":0.0255,"forearm":0.0138,"hand":0.0056,
               "thigh":0.1478,"shank":0.0481,"foot":0.0129}
    fracs = fracs_f if sex == "female" else fracs_m
    orig_mass = mj_model.body_mass.copy()
    for i in range(mj_model.nbody):
        bname = (mujoco.mj_id2name(mj_model, MJOBJ_BODY, i) or "").lower()
        frac  = next((v for k,v in fracs.items() if k in bname), None)
        if frac and orig_mass[i] > 1e-6:
            scale = float(np.clip((weight * frac) / orig_mass[i], 0.3, 5.0))
            mj_model.body_mass[i]    = orig_mass[i] * scale
            mj_model.body_inertia[i] = mj_model.body_inertia[i] * scale
    orig_gear = mj_model.actuator_gear[:, 0].copy()
    sf = float(np.clip(muscle_strength_factor(age, sex, weight, height), 0.85, 1.15))
    mj_model.actuator_gear[:, 0] = orig_gear * sf
    return orig_gear, sf


# -----------------------------------------------------------------------------
# EMBEDDING CACHE
# -----------------------------------------------------------------------------

def _load_cached_embeddings(cache_file=_EMBED_CACHE_FILE):
    try:
        if os.path.exists(cache_file):
            obj = torch.load(cache_file, map_location='cpu')
            if all(k in obj for k in ('z_stand','z_sit','z_fall','z_rest')):
                print(f"  [Embed] cache hit -> {cache_file}")
                return obj
    except Exception as e:
        print(f"  [Embed] cache read failed: {e}")
    return None


def _save_cached_embeddings(z_stand, z_sit, z_fall, z_rest, cache_file=_EMBED_CACHE_FILE):
    try:
        torch.save({'z_stand': z_stand.detach().cpu(),
                    'z_sit': z_sit.detach().cpu(),
                    'z_fall': z_fall.detach().cpu(),
                    'z_rest': z_rest.detach().cpu()}, cache_file)
        print(f"  [Embed] cache saved -> {cache_file}")
    except Exception as e:
        print(f"  [Embed] cache save failed: {e}")


# -----------------------------------------------------------------------------
# EMBEDDING INFERENCE  (standalone - no dependency on backward_fall_walking_best)
# -----------------------------------------------------------------------------

def _infer_z_stand(model):
    """Embed 'stand still upright' goal."""
    print("  [Embed] z_stand …")
    env, _ = make_humenv(task="move-ego-0-0")
    pelvis_id = _body_id(env.unwrapped.model, "Pelvis")
    obs_all, rew_all = [], []
    rwd = LocomotionReward(move_speed=0.0, move_angle=0, stand_height=1.4)
    for trial in range(15):
        torch.manual_seed(SEED + 200 + trial)
        z = model.sample_z(1)
        obs, _ = env.reset()
        for _ in range(90):
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act = model.act(obs_t, z).squeeze(0).numpy()
            obs, _, term, trunc, _ = env.step(act)
            pz = float(env.unwrapped.data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            if pz > 0.80:
                obs_all.append(obs["proprio"].copy())
                rew_all.append(rwd.compute(env.unwrapped.model, env.unwrapped.data))
            if term or trunc:
                break
    env.close()
    obs_t = torch.tensor(np.array(obs_all), dtype=torch.float32)
    rew_t = torch.tensor(np.array(rew_all), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        z_stand = model.reward_inference(obs_t, rew_t).mean(dim=0, keepdim=True)
    print(f"  [Embed] z_stand done ({len(obs_all)} upright states)")
    return z_stand


def _infer_z_fall(model):
    """Embed 'collapsed forward / prone on ground' goal."""
    print("  [Embed] z_fall (forward collapse) …")
    env, _ = make_humenv(task="lieonground-up")
    pelvis_id = _body_id(env.unwrapped.model, "Pelvis")
    head_id   = _body_id(env.unwrapped.model, "Head")
    coll_obs  = []
    for trial in range(12):
        obs, _ = env.reset()
        zero = np.zeros(env.action_space.shape)
        for _ in range(120):
            obs, _, term, trunc, _ = env.step(zero)
            pd = env.unwrapped.data
            pz = float(pd.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            if pz < 0.35:
                # prefer states where the head is in avatar-forward direction,
                # not just world +x.
                if head_id >= 0 and pelvis_id >= 0:
                    fwd_xy, _, _, _ = _avatar_forward_xy(env.unwrapped.model, pd, pelvis_id, head_id)
                    hp = pd.xpos[head_id] - pd.xpos[pelvis_id]
                    fwd = float(np.dot(hp[:2], fwd_xy)) > -0.02
                else:
                    fwd = True
                if fwd:
                    coll_obs.append(obs["proprio"].copy())
            if term or trunc:
                break
    env.close()
    if len(coll_obs) >= 20:
        obs_t = torch.tensor(np.array(coll_obs[-300:]), dtype=torch.float32)
        with torch.no_grad():
            z_fall = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_fall = model.sample_z(1)
    print(f"  [Embed] z_fall done ({len(coll_obs)} forward-collapse states)")
    return z_fall


def _infer_z_sit_attempt(model):
    """Embed the forward-lean postural shift used during sit-down onset."""
    print("  [Embed] z_sit …")
    env, _ = make_humenv(task="move-ego-0-0")
    pelvis_id = _body_id(env.unwrapped.model, "Pelvis")
    torso_id  = _body_id(env.unwrapped.model, "Torso")
    head_id   = _body_id(env.unwrapped.model, "Head")
    sit_obs = []
    for trial in range(24):
        torch.manual_seed(SEED + 600 + trial)
        z = model.sample_z(1)
        obs, _ = env.reset()
        for _ in range(100):
            if torso_id >= 0:
                env.unwrapped.data.xfrc_applied[torso_id, 0] = 18.0
                env.unwrapped.data.xfrc_applied[torso_id, 4] = -5.0
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act = model.act(obs_t, z).squeeze(0).numpy()
            obs, _, term, trunc, _ = env.step(act)
            d = env.unwrapped.data
            ph = float(d.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
            if ph > 0.82 and head_id >= 0 and pelvis_id >= 0:
                vec = d.xpos[head_id] - d.xpos[pelvis_id]
                nv = float(np.linalg.norm(vec))
                if nv > 1e-6:
                    lean = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, [0, 0, 1]), -1, 1))))
                    if 5.0 <= lean <= 30.0:
                        sit_obs.append(obs["proprio"].copy())
            if term or trunc:
                break
    env.close()
    if len(sit_obs) >= 40:
        obs_t = torch.tensor(np.array(sit_obs[-300:]), dtype=torch.float32)
        with torch.no_grad():
            z_sit = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_sit = model.sample_z(1)
    print(f"  [Embed] z_sit done ({len(sit_obs)} lean states)")
    return z_sit


def _infer_z_rest(model):
    """Embed a stable forward-prone rest posture instead of a generic ground pose."""
    print("  [Embed] z_rest (prone settle) …")
    env, _ = make_humenv(task="lieonground-up")
    pelvis_id = _body_id(env.unwrapped.model, "Pelvis")
    head_id   = _body_id(env.unwrapped.model, "Head")
    torso_id  = _body_id(env.unwrapped.model, "Torso")
    prone_obs = []
    for trial in range(18):
        obs, _ = env.reset()
        zero = np.zeros(env.action_space.shape)
        for _ in range(180):
            d = env.unwrapped.data
            if torso_id >= 0:
                d.xfrc_applied[torso_id, 0] = 120.0
                d.xfrc_applied[torso_id, 4] = -30.0
            if pelvis_id >= 0:
                d.xfrc_applied[pelvis_id, 2] = -50.0
            obs, _, term, trunc, _ = env.step(zero)
            if pelvis_id >= 0 and head_id >= 0:
                p = d.xpos[pelvis_id]
                h = d.xpos[head_id]
                qn = float(np.linalg.norm(d.qvel))
                vec = h - p
                nv = float(np.linalg.norm(vec))
                tilt = float(np.degrees(np.arccos(np.clip(np.dot(vec / nv, [0, 0, 1]), -1, 1)))) if nv > 1e-6 else 0.0
                if float(p[2]) < 0.22 and float(h[2]) < 0.38 and tilt > 68.0 and qn < 1.8 and float(h[0]) > float(p[0]) - 0.15:
                    prone_obs.append(obs["proprio"].copy())
            if term or trunc:
                break
    env.close()
    if len(prone_obs) >= 40:
        obs_t = torch.tensor(np.array(prone_obs[-300:]), dtype=torch.float32)
        with torch.no_grad():
            z_rest = model.goal_inference(obs_t).mean(dim=0, keepdim=True)
    else:
        z_rest = model.sample_z(1)
    print(f"  [Embed] z_rest done ({len(prone_obs)} prone-settle states)")
    return z_rest


def _smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


def _blend_z(za, zb, alpha):
    a = _smoothstep(float(alpha))
    return (1.0 - a) * za + a * zb


# -----------------------------------------------------------------------------
# LEAN IMU
# -----------------------------------------------------------------------------

class LeanIMU:
    CLIP_A = 16.0 * 9.81   # ±16 g

    def __init__(self, mj_model, mj_data, age, height):
        self.mj_model = mj_model
        self.mj_data  = mj_data
        self.body_id  = _body_id(mj_model, "Torso", "torso", "Pelvis")
        self.pelvis_id= _body_id(mj_model, "Pelvis")
        af = max(1.0, 1.0 + (age - 60) * 0.015) if age > 60 else 1.0
        self.an = 0.08 * af;  self.gn = 0.015 * af
        self.ab = np.random.normal(0, 0.025 * af, 3)
        self.gb = np.random.normal(0, 0.007 * af, 3)
        self.sta= 0.04 * (height / 1.75)
        self._lpA = np.array([0., 0., 9.81]); self._lpG = np.zeros(3)
        self._pv  = None
        self.buf  = {k: [] for k in ("t","ax","ay","az","gx","gy","gz",
                                      "pelvis_z","pelvis_vx","impact")}

    def log(self, t):
        d = self.mj_data
        bid = self.body_id
        R   = d.xmat[bid].reshape(3, 3)
        v6  = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, d, MJOBJ_BODY, bid, v6, 1)
        vw  = R @ v6[3:]
        gw  = np.array([0., 0., -9.81])
        if self._pv is not None:
            afd  = (vw - self._pv) / (1/30.0)
            araw = R.T @ (afd - gw)
            n = np.linalg.norm(araw)
            if n > self.CLIP_A:
                araw *= self.CLIP_A / n
        else:
            araw = np.array([0., 0., 9.81])
        self._pv = vw.copy()
        sta_v = self.sta * math.sin(2*math.pi*0.25*t + 1.2)
        noisy_a = araw + self.ab + np.random.normal(0, self.an, 3) + sta_v
        noisy_a = np.clip(noisy_a, -self.CLIP_A, self.CLIP_A)
        self._lpA = 0.76 * self._lpA + 0.24 * noisy_a
        v6g = np.zeros(6)
        mujoco.mj_objectVelocity(self.mj_model, d, MJOBJ_BODY, bid, v6g, 1)
        noisy_g = v6g[:3] + self.gb + np.random.normal(0, self.gn, 3)
        self._lpG = 0.76 * self._lpG + 0.24 * noisy_g
        # pelvis kinematics
        pz  = float(d.xpos[self.pelvis_id][2]) if self.pelvis_id >= 0 else 0.0
        v6p = np.zeros(6)
        if self.pelvis_id >= 0:
            mujoco.mj_objectVelocity(self.mj_model, d, MJOBJ_BODY, self.pelvis_id, v6p, 0)
        # impact
        imp = 0.0
        for i in range(d.ncon):
            c  = d.contact[i]
            b1 = int(self.mj_model.geom_bodyid[int(c.geom1)])
            b2 = int(self.mj_model.geom_bodyid[int(c.geom2)])
            if b1 == bid or b2 == bid:
                fw = np.zeros(6)
                mujoco.mj_contactForce(self.mj_model, d, i, fw)
                imp = max(imp, float(max(0., fw[0])))
        self.buf["t"].append(t)
        self.buf["ax"].append(float(self._lpA[0]))
        self.buf["ay"].append(float(self._lpA[1]))
        self.buf["az"].append(float(self._lpA[2]))
        self.buf["gx"].append(float(self._lpG[0]))
        self.buf["gy"].append(float(self._lpG[1]))
        self.buf["gz"].append(float(self._lpG[2]))
        self.buf["pelvis_z"].append(pz)
        self.buf["pelvis_vx"].append(float(v6p[3]))
        self.buf["impact"].append(min(imp, 15000.0))
        return float(np.linalg.norm(self._lpA))

    def export_csv(self, fname, meta=None):
        t_nat = np.array(self.buf["t"])
        if len(t_nat) < 2:
            return {"filename": fname, "frames": 0}
        t100 = np.arange(t_nat[0], t_nat[-1]+1e-9, 0.01)
        def _i(k):
            return np.interp(t100, t_nat, np.array(self.buf[k]))
        ax,ay,az = _i("ax"),_i("ay"),_i("az")
        gx,gy,gz = _i("gx"),_i("gy"),_i("gz")
        pz,pvx   = _i("pelvis_z"),_i("pelvis_vx")
        imp      = _i("impact")
        amag = np.sqrt(ax**2+ay**2+az**2)
        n = len(t100); falls = 0; rows = []
        for i in range(n):
            fall = int(pz[i] < 0.40)
            falls += fall
            rows.append([round(t100[i],4),
                         round(ax[i],5),round(ay[i],5),round(az[i],5),
                         round(gx[i],5),round(gy[i],5),round(gz[i],5),
                         round(pz[i],5),round(pvx[i],5),round(imp[i],3),
                         round(amag[i],5), fall])
        with open(fname,"w",newline="") as f:
            for k,v in (meta or {}).items():
                f.write(f"# {k}: {v}\n")
            w = csv.writer(f)
            w.writerow(["t","ax","ay","az","gx","gy","gz",
                        "pelvis_z","pelvis_vx","impact_n","accel_mag","fall"])
            w.writerows(rows)
        print(f"  [IMU]  ? {fname}  ({n} frames | {falls} fall frames)")
        return {"filename": fname, "frames": n, "falls_detected": falls}


# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------

def _validate(imu, perturb_t, body_mass):
    t   = np.array(imu.buf["t"])
    pz  = np.array(imu.buf["pelvis_z"])
    pvx = np.array(imu.buf["pelvis_vx"])
    imp = np.array(imu.buf["impact"])
    ax  = np.array(imu.buf["ax"]); ay=np.array(imu.buf["ay"]); az=np.array(imu.buf["az"])
    amag = np.sqrt(ax**2+ay**2+az**2)
    k7 = np.hanning(7); k7/=k7.sum()
    afilt = np.convolve(amag, k7, mode="same") if len(amag)>=7 else amag.copy()
    peak_filt = float(np.max(afilt))
    BW = body_mass * 9.81
    i0 = int(np.searchsorted(t, perturb_t, side="left"))
    h0 = float(np.max(pz[max(0,i0-5):i0+3])) if i0 < len(pz) else 1.0
    onset = next((j for j in range(i0, len(t))
                  if pz[j] < max(0.65*h0, h0-0.15) and abs(pvx[j]) > 0.30), i0)
    settle= next((j for j in range(onset, len(t))
                  if pz[j] < 0.22 and abs(pvx[j]) < 0.15 and j > onset+10), None)
    dur   = float(t[settle]-t[onset]) if settle else float(t[-1]-t[onset])
    peak_imp = float(np.max(imp[i0:])) if i0<len(imp) else 0.0
    checks = {
        "peak_accel_in_SISFall_range": 15 <= peak_filt <= 130,
        "fall_duration_realistic":     0.5 <= dur      <= 6.0,
        "peak_impact_in_range":        BW   <= peak_imp <= 22*BW,
        "pelvis_reached_floor":        float(np.min(pz[i0:])) < 0.30,
    }
    score = float(np.mean([1.0 if v else 0.5 for v in checks.values()]))
    cls   = ("HIGH_CONFIDENCE" if score > 0.85 else
             "MODERATE_CONFIDENCE" if score > 0.65 else "LOW_CONFIDENCE")
    return {"score":score,"classification":cls,"checks":checks,
            "fall_duration_s":round(dur,3),
            "peak_accel_filt":round(peak_filt,2),
            "peak_impact_n":  round(peak_imp,1),
            "sisfall_compliant": (15<=peak_filt<=130 and 0.5<=dur<=6.0)}


# -----------------------------------------------------------------------------
# PROMPT HELPERS
# -----------------------------------------------------------------------------

def _pf(label, default, unit='', lo=None, hi=None):
    while True:
        raw = input(f"    {label} [{default}{' ' + unit if unit else ''}]: ").strip()
        if raw == '':
            return default
        try:
            v = float(raw)
            if lo is not None and v < lo:
                print(f"      Must be >= {lo}.")
                continue
            if hi is not None and v > hi:
                print(f"      Must be <= {hi}.")
                continue
            return v
        except ValueError:
            print("      Enter a number.")


def _ps(label, default, choices):
    while True:
        raw = input(f"    {label} [{default}] ({'/'.join(choices)}): ").strip().lower()
        if raw == '':
            return default
        if raw in choices:
            return raw
        print(f"      Choose: {choices}")


def _po(label, unit='', lo=None, hi=None):
    while True:
        raw = input(f"    {label} [auto{' ' + unit if unit else ''}]: ").strip()
        if raw == '':
            return None
        try:
            v = float(raw)
            if lo is not None and v < lo:
                print(f"      Must be >= {lo}.")
                continue
            if hi is not None and v > hi:
                print(f"      Must be <= {hi}.")
                continue
            return v
        except ValueError:
            print("      Enter a number or press Enter for auto.")


# -----------------------------------------------------------------------------
# MAIN  run()
# -----------------------------------------------------------------------------

def run(subject_params: dict | None = None) -> dict:
    """
    Entry-point called by fall_dispatcher or directly.
    subject_params: {age, height, sex, weight (optional)}
    """
    if subject_params is None:
        print("\n" + "=" * 70)
        print("  Scenario 20 - Forward fall when trying to sit down")
        print("=" * 70)
        print("\n  Enter subject parameters (Enter = default):\n")
        subject_params = {
            "age": int(_pf("Age", 70, "years", lo=1, hi=120)),
            "height": float(_pf("Height", 1.65, "m", lo=0.5, hi=2.5)),
            "sex": _ps("Sex", "male", ["male", "female"]),
            "weight": _po("Weight", "kg", lo=25.0, hi=250.0),
        }

    age    = int(subject_params["age"])
    height = float(subject_params["height"])
    sex    = str(subject_params.get("sex","male")).lower()
    weight, bmi, w_src = _resolve_weight(height, age, sex, subject_params.get("weight"))
    w_str  = f"{subject_params['weight']:.1f}kg (user)" if subject_params.get("weight") is not None else f"{weight:.1f}kg (auto BMI={bmi:.1f})"

    print("\n" + "="*70)
    print("  SCENARIO 20 - Forward fall when trying to sit down")
    print(f"  Subject: age={age}yr  h={height}m  {w_str}  sex={sex}")
    print("="*70)
    print_subject_profile(age, sex, height, weight)

    # -- timing constants ------------------------------------------------------
    STAND_S    = 150   # 5.0 s  stand still
    SITDOWN_S  =  90   # 3.0 s  guided sit-down
    COLLAPSE_S =  30   # 1.0 s  forward topple
    FALL_S     = 400   # 13.3 s  fall + settle
    TOTAL      = STAND_S + SITDOWN_S + COLLAPSE_S + FALL_S

    # -- subject params --------------------------------------------------------
    rt_steps  = max(1, round(reaction_delay_seconds(age, sex, height) * 30))
    bal_imp   = balance_impairment(age, sex, weight)
    sf        = float(np.clip(muscle_strength_factor(age, sex, weight, height), 0.85, 1.15))
    min_gear  = float(weakening_config(age, sex, weight)["min_factor"])
    print(f"\n  Phases: stand={STAND_S}  sit_down={SITDOWN_S}  "
          f"collapse={COLLAPSE_S}  fall={FALL_S}  total={TOTAL}")
    print(f"  sf={sf:.3f}  rt_steps={rt_steps}  bal={bal_imp:.3f}  min_gear={min_gear:.3f}")

    # -- load model ------------------------------------------------------------
    print("\n  [1/4] Loading Meta Motivo …")
    motivo = FBcprModel.from_pretrained("facebook/metamotivo-M-1")
    motivo.eval()
    print(f"        Loaded on {next(motivo.parameters()).device}")

    # -- task embeddings -------------------------------------------------------
    print("\n  [2/4] Inferring task embeddings …")
    z_stand = _infer_z_stand(motivo)
    z_sit   = _infer_z_sit_attempt(motivo)
    z_fall  = _infer_z_fall(motivo)
    z_rest  = _infer_z_rest(motivo)

    # -- environment -----------------------------------------------------------
    print("\n  [3/4] Creating environment …")
    env, _ = make_humenv(task="move-ego-0-0")
    obs, _ = env.reset()
    mj_model = env.unwrapped.model
    mj_data  = env.unwrapped.data

    # -- anthropometry ---------------------------------------------------------
    orig_gear, _ = _apply_anthropometry(mj_model, age, height, weight, sex)
    body_mass    = float(np.sum(mj_model.body_mass))
    BW           = body_mass * 9.81
    print(f"        Scaled body mass = {body_mass:.2f} kg  BW = {BW:.1f} N")

    grp       = _get_actuator_groups(mj_model)
    pelvis_id = _body_id(mj_model, "Pelvis","pelvis")
    torso_id  = _body_id(mj_model, "Torso", "torso")
    head_id   = _body_id(mj_model, "Head",  "head")
    hand_ids  = [bid for bid in [
        _body_id(mj_model, "L_Hand", "HandL", "LeftHand"),
        _body_id(mj_model, "R_Hand", "HandR", "RightHand"),
        _body_id(mj_model, "L_ForeArm", "ForeArmL", "LeftForeArm"),
        _body_id(mj_model, "R_ForeArm", "ForeArmR", "RightForeArm"),
    ] if bid >= 0]
    leg_trail_ids = [bid for bid in [
        _body_id(mj_model, "L_Ankle", "AnkleL", "LeftAnkle"),
        _body_id(mj_model, "R_Ankle", "AnkleR", "RightAnkle"),
        _body_id(mj_model, "L_Foot", "FootL", "LeftFoot"),
        _body_id(mj_model, "R_Foot", "FootR", "RightFoot"),
        _body_id(mj_model, "L_Shank", "ShankL", "LeftShank"),
        _body_id(mj_model, "R_Shank", "ShankR", "RightShank"),
    ] if bid >= 0]

    fall_fwd_xy = np.array([1.0, 0.0], dtype=float)
    fall_lat_3d = np.array([0.0, 1.0, 0.0], dtype=float)
    fall_up_3d = np.array([0.0, 0.0, 1.0], dtype=float)
    heading_locked = False

    def _lock_heading():
        nonlocal fall_fwd_xy, fall_lat_3d, fall_up_3d, heading_locked
        fall_fwd_xy, fall_lat_3d, fall_up_3d, yaw = _avatar_forward_xy(mj_model, mj_data, pelvis_id, head_id)
        heading_locked = True
        print(f"        Heading locked: fwd=({fall_fwd_xy[0]:+.3f},{fall_fwd_xy[1]:+.3f})  yaw={yaw:+.1f} deg | lat=({fall_lat_3d[0]:+.3f},{fall_lat_3d[1]:+.3f},{fall_lat_3d[2]:+.3f})")

    def _apply_body_forward_wrench(body_id, fwd_n=0.0, down_n=0.0, pitch_nm=0.0):
        if body_id < 0:
            return
        fx, fy = fall_fwd_xy
        lx, ly, lz = fall_lat_3d
        mj_data.xfrc_applied[body_id, 0] += float(fwd_n) * fx
        mj_data.xfrc_applied[body_id, 1] += float(fwd_n) * fy
        mj_data.xfrc_applied[body_id, 2] -= float(down_n)
        # world torque around avatar-lateral axis -> forward pitch
        # Positive torque around the avatar's +lateral axis pitches the trunk/head
        # forward in a right-handed world frame. The previous sign was reversed,
        # which made young/strong subjects rotate head-backward during collapse
        # even while the translational force was forward.
        mj_data.xfrc_applied[body_id, 3] += float(pitch_nm) * lx
        mj_data.xfrc_applied[body_id, 4] += float(pitch_nm) * ly
        mj_data.xfrc_applied[body_id, 5] += float(pitch_nm) * lz

    def _head_forward_rel():
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        hp = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
        return float(hp[0] * fall_fwd_xy[0] + hp[1] * fall_fwd_xy[1])

    def _torso_prone_score():
        """+1 ~= chest/front points down (prone), -1 ~= chest/front points up."""
        if pelvis_id < 0 or head_id < 0:
            return 0.0
        f2_now, lat_now, up_now, _ = _avatar_forward_xy(mj_model, mj_data, pelvis_id, head_id)
        front_now = np.cross(lat_now, up_now)
        nf = float(np.linalg.norm(front_now))
        if nf < 1e-8:
            return 0.0
        front_now /= nf
        if float(np.dot(front_now[:2], fall_fwd_xy)) < 0.0:
            front_now = -front_now
        return float(-front_now[2])

    def _draw_virtual_chair(viewer):
        if chair_center_xy is None or viewer is None:
            return
        try:
            scn = viewer.user_scn
            scn.ngeom = 0
            # chair yaw so backrest is behind the subject
            yaw = math.atan2(fall_fwd_xy[1], fall_fwd_xy[0])
            cy, sy = math.cos(yaw), math.sin(yaw)
            # seat box (viewer-only)
            seat_pos = np.array([chair_center_xy[0], chair_center_xy[1], chair_seat_z], dtype=float)
            seat_mat = np.array([[cy, -sy, 0.0],[sy, cy, 0.0],[0.0, 0.0, 1.0]], dtype=float).reshape(-1)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([0.23, 0.20, 0.025], dtype=float),
                                seat_pos, seat_mat,
                                np.array([0.45, 0.30, 0.18, 0.55], dtype=float))
            scn.ngeom += 1
            # backrest box behind seat
            back_offset = -0.20 * fall_fwd_xy
            back_pos = np.array([chair_center_xy[0] + back_offset[0], chair_center_xy[1] + back_offset[1], chair_seat_z + 0.25], dtype=float)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([0.025, 0.20, 0.25], dtype=float),
                                back_pos, seat_mat,
                                np.array([0.40, 0.27, 0.15, 0.50], dtype=float))
            scn.ngeom += 1
        except Exception:
            pass

    # -- IMU -------------------------------------------------------------------
    imu = LeanIMU(mj_model, mj_data, age, height)
    print(f"        IMU mounted on body_id={imu.body_id}")

    # -- embedding blend state -------------------------------------------------
    z_cur = z_stand.clone()

    def _blend_to(z_new, steps=25):
        """Yield smoothly interpolated z from z_cur to z_new over `steps` steps."""
        for k in range(steps + 1):
            yield _blend_z(z_cur, z_new, k / steps)

    _blend_gen  = iter([z_stand])   # idle generator
    _blend_done = True

    def _start_blend(z_new, steps=25):
        nonlocal _blend_gen, _blend_done, z_cur
        _blend_gen  = iter(list(_blend_to(z_new, steps)))
        _blend_done = False

    def _step_blend():
        nonlocal z_cur, _blend_done
        try:
            z_cur = next(_blend_gen)
        except StopIteration:
            _blend_done = True

    # -- sim state -------------------------------------------------------------
    phase              = "stand"
    perturb_t          = 0.0
    trigger_fired      = False
    collapse_start_step = STAND_S + SITDOWN_S
    sit_end_strength   = 0.40
    rest_mode          = False
    rest_ctr           = 0
    rest_anchor_xy     = None
    sit_anchor_xy      = None
    chair_center_xy    = None
    chair_seat_z       = 0.50
    metrics            = []
    age_frac           = float(np.clip((age - 55.0) / 25.0, 0.0, 1.0))
    fall_resid_scale   = float(np.clip(1.00 - 0.55 * age_frac - 0.35 * bal_imp, 0.22, 1.00))
    collapse_pitch_scale = float(np.clip(1.00 - 0.45 * age_frac - 0.30 * bal_imp, 0.35, 1.00))
    collapse_force_scale = float(np.clip(1.00 - 0.20 * age_frac - 0.15 * bal_imp, 0.65, 1.00))
    settle_target_deg = float(np.clip(86.0 - 4.0 * age_frac - 6.0 * bal_imp, 76.0, 88.0))
    prone_target = float(np.clip(0.42 - 0.10 * age_frac - 0.08 * bal_imp, 0.22, 0.42))
    chair_trigger_pz = float(np.clip(0.56 - 0.03 * age_frac - 0.03 * bal_imp, 0.48, 0.56))
    sit_trigger_local = int(round(20 + 8 * age_frac))
    leg_floor = float(np.clip(0.18 - 0.05 * age_frac - 0.04 * bal_imp, 0.12, 0.18))
    torso_floor = float(np.clip(0.16 - 0.04 * age_frac - 0.03 * bal_imp, 0.10, 0.16))
    arm_floor = float(np.clip(0.34 - 0.12 * age_frac - 0.10 * bal_imp, 0.18, 0.34))
    action_hist        = deque(maxlen=3)

    print(f"\n  [4/4] Running {TOTAL}-step simulation …\n")
    print(f"  {'step':>5} {'phase':>9} {'pz':>7} {'trunk°':>7} {'head_f':>7} {'prone':>7} "
          f"{'vx':>6} {'gear%':>6} {'ncon':>5}  status")
    print("  " + "-"*70)

    # -- MAIN LOOP -------------------------------------------------------------
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance  = 3.6
        viewer.cam.elevation = -8
        viewer.cam.azimuth   = 90

        for step in range(TOTAL):
            sim_t = float(mj_data.time)
            mj_data.xfrc_applied[:] = 0.0     # clear external forces each step

            # --- PHASE: STAND ----------------------------------------------
            if step < STAND_S:
                phase = "stand"
                if step == 0:
                    _start_blend(z_stand, steps=15)
                # Keep leg gears at full strength for stable standing
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * sf
                # Damp arm action - quiet arms while standing
                arm_damp = 0.20

            # --- PHASE: SIT_DOWN -------------------------------------------
            elif step < collapse_start_step:
                phase  = "sit_down"
                local  = step - STAND_S

                if local == 0:
                    print(f"\n  [Step {step}] SIT_DOWN phase - guided lowering begins")
                    _lock_heading()
                    try:
                        view_yaw = float(np.degrees(np.arctan2(fall_fwd_xy[1], fall_fwd_xy[0])))
                        viewer.cam.azimuth = view_yaw + 90.0
                        viewer.cam.elevation = -6.0
                        viewer.cam.distance = 3.3
                        print(f"        Camera locked side-on: azimuth={viewer.cam.azimuth:+.1f} deg")
                    except Exception:
                        pass
                    _start_blend(z_sit, steps=30)

                ramp = local / max(SITDOWN_S - 1, 1)
                bell = math.sin(math.pi * ramp)

                # Controlled sit-down should stay almost in place, with the hips
                # moving slightly backward toward the chair while the trunk leans
                # forward. Do NOT let the locomotion policy turn this into walking.
                down_n = (0.02 + 0.05 * bell) * BW
                torso_fwd_n = 0.05 * BW * bell
                torso_pitch_nm = 5.5 * bell
                if torso_id >= 0:
                    _apply_body_forward_wrench(torso_id,
                                               fwd_n=torso_fwd_n,
                                               pitch_nm=torso_pitch_nm,
                                               down_n=0.12 * down_n)
                if pelvis_id >= 0:
                    mj_data.xfrc_applied[pelvis_id, 2] -= 0.88 * down_n
                    # steer pelvis backward toward the chair center and suppress walk drift
                    if chair_center_xy is not None:
                        cur_xy = mj_data.xpos[pelvis_id][:2].copy()
                        err_xy = chair_center_xy - cur_xy
                        back_dir = -fall_fwd_xy
                        back_err = float(np.dot(err_xy, back_dir))
                        lat_xy = np.array([fall_lat_3d[0], fall_lat_3d[1]], dtype=float)
                        lat_n = float(np.linalg.norm(lat_xy))
                        lat_xy = lat_xy / max(lat_n, 1e-8)
                        lat_err = float(np.dot(err_xy, lat_xy))
                        back_force = np.clip(back_err, -0.18, 0.22) * (0.95 * BW)
                        lat_force  = np.clip(lat_err,  -0.12, 0.12) * (0.55 * BW)
                        mj_data.xfrc_applied[pelvis_id, 0] += back_force * back_dir[0] + lat_force * lat_xy[0]
                        mj_data.xfrc_applied[pelvis_id, 1] += back_force * back_dir[1] + lat_force * lat_xy[1]
                    # strong root translation damping to prevent walk-like drift
                    if mj_data.qvel.shape[0] >= 2:
                        mj_data.qvel[0] *= 0.30
                        mj_data.qvel[1] *= 0.30

                # -- Leg gear decays from 100% -> 55% (controlled eccentric sit) --
                leg_factor = 1.0 - 0.22 * ramp
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(leg_factor, 0.78)
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(0.70, 1.0 - 0.18 * ramp)

                arm_damp = 0.45

                # Check trigger only once the avatar is actually low enough to be
                # near the seat height. This avoids an early walk-like collapse.
                pz_now = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
                head_f_now = _head_forward_rel()
                if not trigger_fired and pz_now < chair_trigger_pz and local >= sit_trigger_local and head_f_now > 0.02:
                    trigger_fired = True
                    collapse_start_step = min(collapse_start_step, step + 1)
                    perturb_t = sim_t
                    print(f"  [Step {step}] Collapse trigger - pelvis_z={pz_now:.3f} m | collapse starts at step {collapse_start_step}")

            # --- PHASE: COLLAPSE -------------------------------------------
            elif step < collapse_start_step + COLLAPSE_S:
                phase   = "collapse"
                local_c = step - collapse_start_step

                if local_c == 0:
                    if not trigger_fired:
                        trigger_fired = True
                        perturb_t     = sim_t
                        print(f"  [Step {step}] Fallback collapse trigger")
                    print(f"\n  [Step {step}] COLLAPSE phase - forward topple force applied")
                    _start_blend(z_fall, steps=20)

                prog  = local_c / max(COLLAPSE_S - 1, 1)
                ramp_f = 0.5 * (1 - math.cos(math.pi * prog))  # 0?1
                if mj_data.qvel.shape[0] >= 2:
                    mj_data.qvel[0] *= 0.35
                    mj_data.qvel[1] *= 0.35

                # Failed sit-down should rotate the body forward, not merely
                # translate the whole avatar forward. Use a rotational couple:
                #   1) forward/down pull on upper trunk (and head if available)
                #   2) slight backward brake on the pelvis
                # This makes young/strong subjects pitch forward early instead
                # of visually "sliding then flipping" later in free-fall.
                early = 0.35 + 0.65 * ramp_f
                torso_fwd = 0.22 * BW * early * collapse_force_scale
                torso_down = 0.12 * BW * early
                torso_pitch = 30.0 * early * collapse_pitch_scale
                if torso_id >= 0:
                    _apply_body_forward_wrench(torso_id,
                                               fwd_n=torso_fwd,
                                               down_n=torso_down,
                                               pitch_nm=torso_pitch)
                if head_id >= 0:
                    _apply_body_forward_wrench(head_id,
                                               fwd_n=0.16 * BW * early * collapse_force_scale,
                                               down_n=0.08 * BW * early,
                                               pitch_nm=18.0 * early * collapse_pitch_scale)
                if pelvis_id >= 0:
                    mj_data.xfrc_applied[pelvis_id, 2] -= 0.06 * BW * early
                    mj_data.xfrc_applied[pelvis_id, 0] -= 0.12 * BW * early * fall_fwd_xy[0]
                    mj_data.xfrc_applied[pelvis_id, 1] -= 0.12 * BW * early * fall_fwd_xy[1]
                    # brief seat/backrest reaction: pelvis is still near the chair,
                    # so the seat region resists backward collapse and helps pitch the
                    # trunk forward like a failed sit-down rather than a walk-fall.
                    if chair_center_xy is not None and local_c < 12:
                        cur_xy = mj_data.xpos[pelvis_id][:2].copy()
                        rel_xy = cur_xy - chair_center_xy
                        back_align = float(np.dot(rel_xy, -fall_fwd_xy))
                        if back_align > -0.08:
                            mj_data.xfrc_applied[pelvis_id, 0] += 0.10 * BW * early * (-fall_fwd_xy[0])
                            mj_data.xfrc_applied[pelvis_id, 1] += 0.10 * BW * early * (-fall_fwd_xy[1])
                if hand_ids:
                    reach_gain = 0.55 + 0.45 * ramp_f
                    for bid in hand_ids:
                        _apply_body_forward_wrench(bid,
                                                   fwd_n=0.035 * BW * reach_gain,
                                                   down_n=-0.006 * BW * reach_gain,
                                                   pitch_nm=0.0)
                if leg_trail_ids and (_head_forward_rel() > 0.12 or prog > 0.25):
                    trail_gain = 0.35 + 0.65 * ramp_f
                    for bid in leg_trail_ids:
                        _apply_body_forward_wrench(bid,
                                                   fwd_n=-0.025 * BW * trail_gain,
                                                   down_n=0.008 * BW * trail_gain,
                                                   pitch_nm=0.0)

                # If the head is still behind the pelvis during collapse, add
                # extra upper-body forward pitch immediately instead of waiting
                # until the later fall phase.
                head_f_c = _head_forward_rel()
                prone_c = _torso_prone_score()
                if torso_id >= 0 and (head_f_c < 0.08 or prone_c < prone_target * 0.35):
                    corr_h = np.clip(0.08 - head_f_c, 0.0, 0.20) / 0.20
                    corr_p = np.clip(prone_target * 0.35 - prone_c, 0.0, 0.40) / 0.40
                    corr = max(float(corr_h), float(corr_p))
                    _apply_body_forward_wrench(torso_id,
                                               fwd_n=0.16 * BW * corr,
                                               down_n=0.05 * BW * corr,
                                               pitch_nm=26.0 * corr)
                    if pelvis_id >= 0:
                        mj_data.xfrc_applied[pelvis_id, 0] -= 0.08 * BW * corr * fall_fwd_xy[0]
                        mj_data.xfrc_applied[pelvis_id, 1] -= 0.08 * BW * corr * fall_fwd_xy[1]

                # -- All muscles weaken toward min_gear -----------------------
                gear_f = sit_end_strength - prog * (sit_end_strength - leg_floor)
                torso_f = 0.46 - prog * (0.46 - torso_floor)
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(gear_f, leg_floor)
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * max(torso_f, torso_floor)
                for idx in grp["arm"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * arm_floor
                arm_damp = 0.80

            # --- PHASE: FALL -----------------------------------------------
            else:
                phase = "fall"
                local_f = step - (collapse_start_step + COLLAPSE_S)

                if local_f == 0:
                    print(f"\n  [Step {step}] FALL phase - free fall")

                # Keep a short decaying forward-prone residual so the avatar
                # completes rotation instead of rebounding backward.
                if local_f < 34 and torso_id >= 0:
                    decay = 0.90 ** local_f
                    resid_fwd = 0.05 * BW * decay * fall_resid_scale
                    resid_pitch = 16.0 * decay * fall_resid_scale
                    _apply_body_forward_wrench(torso_id,
                                               fwd_n=resid_fwd,
                                               pitch_nm=resid_pitch)
                    if pelvis_id >= 0:
                        mj_data.xfrc_applied[pelvis_id, 0] += 0.03 * BW * decay * fall_resid_scale * fall_fwd_xy[0]
                        mj_data.xfrc_applied[pelvis_id, 1] += 0.03 * BW * decay * fall_resid_scale * fall_fwd_xy[1]

                # Keep some residual leg and arm control so the body does not fold unnaturally.
                mj_model.actuator_gear[:, 0] = orig_gear * min_gear
                for idx in grp["leg"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * leg_floor
                for idx in grp["torso"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * torso_floor
                for idx in grp["arm"]:
                    mj_model.actuator_gear[idx, 0] = orig_gear[idx] * arm_floor

                pz_f   = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 1.0
                v6_p   = np.zeros(6)
                if pelvis_id >= 0:
                    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, v6_p, 0)
                spd    = float(np.linalg.norm(v6_p[3:5]))

                trunk_f = 0.0
                if pelvis_id >= 0 and head_id >= 0:
                    vec_f = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
                    nv_f = float(np.linalg.norm(vec_f))
                    if nv_f > 1e-8:
                        trunk_f = float(np.degrees(np.arccos(np.clip(np.dot(vec_f / nv_f, [0.0, 0.0, 1.0]), -1.0, 1.0))))
                head_fwd = _head_forward_rel()

                prone_now = _torso_prone_score()
                # Keep driving rotation until the body is not just horizontal but
                # also anatomically prone (torso forward axis pointing downward).
                if not rest_mode and torso_id >= 0 and local_f < 100:
                    need_forward = (head_fwd < 0.18) or (trunk_f < settle_target_deg - 2.0) or (prone_now < prone_target)
                    if need_forward:
                        gain = max(0.0, 1.0 - local_f / 100.0)
                        soft = float(np.clip(1.00 - 0.35 * age_frac - 0.20 * bal_imp, 0.55, 1.00))
                        extra_fwd = 0.10 * BW * gain * soft
                        extra_pitch = 22.0 * gain * soft
                        _apply_body_forward_wrench(torso_id, fwd_n=extra_fwd, down_n=0.03*BW*gain*soft, pitch_nm=extra_pitch)
                        if pelvis_id >= 0:
                            mj_data.xfrc_applied[pelvis_id, 0] -= 0.05 * BW * gain * soft * fall_fwd_xy[0]
                            mj_data.xfrc_applied[pelvis_id, 1] -= 0.05 * BW * gain * soft * fall_fwd_xy[1]
                if local_f < 24:
                    if hand_ids:
                        rg = max(0.0, 1.0 - local_f / 24.0)
                        for bid in hand_ids:
                            _apply_body_forward_wrench(bid,
                                                       fwd_n=0.025 * BW * rg,
                                                       down_n=-0.004 * BW * rg,
                                                       pitch_nm=0.0)
                    if leg_trail_ids and trunk_f > 45.0:
                        tg = max(0.0, 1.0 - local_f / 24.0)
                        for bid in leg_trail_ids:
                            _apply_body_forward_wrench(bid,
                                                       fwd_n=-0.018 * BW * tg,
                                                       down_n=0.006 * BW * tg,
                                                       pitch_nm=0.0)

                # Over-rotation control for older / weaker profiles
                if trunk_f > settle_target_deg + 6.0 and mj_data.qvel.shape[0] >= 6:
                    mj_data.qvel[3] *= 0.65
                    mj_data.qvel[4] *= 0.65
                    mj_data.qvel[5] *= 0.82

                settled_ok = (
                    pz_f < 0.23 and spd < 0.16 and mj_data.ncon >= 6
                    and head_fwd > 0.18 and prone_now > prone_target
                    and (settle_target_deg - 8.0) <= trunk_f <= (settle_target_deg + 10.0)
                )
                if settled_ok:
                    rest_ctr += 1
                else:
                    rest_ctr = 0

                if rest_ctr >= 5 and not rest_mode:
                    rest_mode = True
                    rest_anchor_xy = mj_data.qpos[:2].copy() if mj_data.qpos.shape[0] >= 2 else None
                    print(f"  [Step {step}] Rest mode - body settled")
                    mj_model.dof_damping[:] *= 2.4
                    _start_blend(z_rest, steps=25)

                if rest_mode:
                    if mj_data.qvel.shape[0] >= 6:
                        mj_data.qvel[0] *= 0.01
                        mj_data.qvel[1] *= 0.01
                        mj_data.qvel[2] *= 0.88
                        mj_data.qvel[3] *= 0.35
                        mj_data.qvel[4] *= 0.35
                        mj_data.qvel[5] *= 0.55
                        if rest_anchor_xy is not None and mj_data.qpos.shape[0] >= 2:
                            mj_data.qpos[0] = 0.94 * float(mj_data.qpos[0]) + 0.06 * float(rest_anchor_xy[0])
                            mj_data.qpos[1] = 0.94 * float(mj_data.qpos[1]) + 0.06 * float(rest_anchor_xy[1])
                arm_damp = 1.0

            # -- advance embedding ---------------------------------------------
            _step_blend()

            # -- policy action -------------------------------------------------
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = motivo.act(obs_t, z_cur).squeeze(0).numpy()

            # suppress arm swing while standing; partial in collapse
            for idx in grp["arm"]:
                action[idx] *= arm_damp
            if phase == "sit_down":
                # This is a near-stationary stand-to-sit attempt, not walking.
                # Strongly reduce policy drive so external seat/hip mechanics dominate.
                for idx in grp["leg"]:
                    action[idx] *= 0.40
                for idx in grp["torso"]:
                    action[idx] *= 0.32
                for idx in grp["arm"]:
                    action[idx] *= 0.35
            elif phase == "collapse":
                # Collapse is dominated by failed extensor control and external
                # rotational mechanics. Keep policy action very small so the
                # controller does not hold the trunk upright in young subjects.
                for idx in grp["leg"]:
                    action[idx] *= 0.18
                for idx in grp["torso"]:
                    action[idx] *= 0.12
                for idx in grp["arm"]:
                    action[idx] *= 0.85
            elif phase == "fall":
                for idx in grp["leg"]:
                    action[idx] *= 0.12
                for idx in grp["torso"]:
                    action[idx] *= 0.18
                for idx in grp["arm"]:
                    action[idx] *= 0.45
            # in rest mode zero everything
            if rest_mode:
                action[:] = 0.0

            # temporal smoothing (reduces jitter)
            action_hist.append(action.copy())
            if len(action_hist) >= 2:
                wts = np.array([0.30, 0.70]) if phase == "stand" else np.array([0.45, 0.55])
                action = np.average(list(action_hist)[-2:], weights=wts, axis=0)

            # -- step env -----------------------------------------------------
            obs, _, terminated, truncated, _ = env.step(action)

            # -- IMU log -------------------------------------------------------
            imu_pk = imu.log(sim_t)
            _draw_virtual_chair(viewer)

            # -- console log ---------------------------------------------------
            log_now = (step % 30 == 0 or phase == "collapse"
                       or (phase == "fall" and step % 10 == 0))
            if log_now:
                pz_  = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.0
                vv   = np.zeros(6)
                if pelvis_id >= 0:
                    mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vv, 0)
                if heading_locked:
                    vx_ = float(np.dot(vv[3:5], fall_fwd_xy))
                else:
                    vx_ = float(vv[3])
                td_  = 0.0
                if pelvis_id >= 0 and head_id >= 0:
                    vec = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
                    n_  = np.linalg.norm(vec)
                    if n_ > 1e-8:
                        td_ = float(np.degrees(np.arccos(
                            np.clip(np.dot(vec/n_, [0.,0.,1.]), -1.,1.))))
                g_pct = (mj_model.actuator_gear[grp["leg"][0],0] /
                         max(orig_gear[grp["leg"][0]],1e-9) * 100) if grp["leg"] else 100.
                hf_ = _head_forward_rel() if heading_locked else 0.0
                pr_ = _torso_prone_score() if heading_locked else 0.0
                st   = "FALL!" if pz_ < 0.35 else "ok"
                print(f"  {step:>5d} {phase:>9s} {pz_:>7.3f} {td_:>7.1f} {hf_:>7.3f} {pr_:>7.3f}  "
                      f"{vx_:>6.2f} {g_pct:>5.0f}% {mj_data.ncon:>5d}  {st}")

            pz_m = float(mj_data.xpos[pelvis_id][2]) if pelvis_id >= 0 else 0.0
            vv_m = np.zeros(6)
            if pelvis_id >= 0:
                mujoco.mj_objectVelocity(mj_model, mj_data, MJOBJ_BODY, pelvis_id, vv_m, 0)
            vx_m = float(np.dot(vv_m[3:5], fall_fwd_xy)) if heading_locked else float(vv_m[3])
            td_m = 0.0
            if pelvis_id >= 0 and head_id >= 0:
                vec_m = mj_data.xpos[head_id] - mj_data.xpos[pelvis_id]
                n_m = float(np.linalg.norm(vec_m))
                if n_m > 1e-8:
                    td_m = float(np.degrees(np.arccos(np.clip(np.dot(vec_m / n_m, [0.0, 0.0, 1.0]), -1.0, 1.0))))
            metrics.append({"step": step, "phase": phase, "pelvis_h": pz_m, "trunk_deg": td_m, "fwd_v": vx_m, "head_fwd": (_head_forward_rel() if heading_locked else 0.0), "prone": (_torso_prone_score() if heading_locked else 0.0)})

            viewer.sync()

            # reset only during stand if env signals terminal
            if terminated and phase == "stand":
                obs, _ = env.reset()
                mj_model.actuator_gear[:, 0] = orig_gear * sf

    # -- export ----------------------------------------------------------------
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    pfx = f"fall_scenario20_age{age}_{ts}"
    imu_r = imu.export_csv(pfx + ".csv", meta={
        "scenario_id": 20,
        "description": "Forward fall when trying to sit down",
        "age": age, "height_m": height, "sex": sex, "weight_kg": weight,
    })
    val = _validate(imu, perturb_t, body_mass)

    # -- validation report -----------------------------------------------------
    rpt = pfx + "_validation.txt"
    with open(rpt, "w") as f:
        f.write("="*60 + "\nSCENARIO 20 VALIDATION REPORT\n" + "="*60 + "\n")
        f.write(f"age={age}  h={height}m  w={weight:.1f}kg  sex={sex}\n\n")
        f.write(f"Score:          {val['score']:.1%}\n")
        f.write(f"Classification: {val['classification']}\n")
        f.write(f"Fall duration:  {val['fall_duration_s']:.2f} s\n")
        f.write(f"Peak accel:     {val['peak_accel_filt']:.1f} m/s² (filtered)\n")
        f.write(f"Peak impact:    {val['peak_impact_n']:.0f} N\n")
        f.write(f"SISFall:        {val['sisfall_compliant']}\n\nChecks:\n")
        for k, v in val["checks"].items():
            f.write(f"  {'PASS' if v else 'FAIL'}  {k}\n")

    print("\n" + "=" * 70)
    print("  SCENARIO 20 POST-SIM PHYSICS AUDIT")
    print("=" * 70)
    for ph_name in ("stand", "sit_down", "collapse", "fall"):
        pm = [m for m in metrics if m["phase"] == ph_name]
        if not pm:
            continue
        hs = [m["pelvis_h"] for m in pm]
        ts = [m["trunk_deg"] for m in pm]
        vs = [m["fwd_v"] for m in pm]
        hfs = [m['head_fwd'] for m in pm]
        prs = [m['prone'] for m in pm]
        print(f"  {ph_name:>8s}: pelvis {hs[0]:.3f}->{hs[-1]:.3f} m | trunk {ts[0]:.1f}->{ts[-1]:+.1f} deg | head_f {hfs[0]:+.3f}->{hfs[-1]:+.3f} m | prone {prs[0]:+.3f}->{prs[-1]:+.3f} | fwd_v max {max(vs):+.3f} m/s")
    fall_pm = [m for m in metrics if m["phase"] == "fall"]
    if fall_pm:
        final_trunk = fall_pm[-1]["trunk_deg"]
        peak_fwd = max(m["fwd_v"] for m in fall_pm)
        print(f"  final trunk lean : {final_trunk:.1f} deg")
        print(f"  max fall fwd vel : {peak_fwd:+.3f} m/s")

    env.close()
    mj_model.actuator_gear[:, 0] = orig_gear.copy()

    print("\n" + "="*70)
    print("  SCENARIO 20 COMPLETE")
    print("="*70)
    print(f"  Score:         {val['score']:.1%}  ({val['classification']})")
    print(f"  Fall duration: {val['fall_duration_s']:.2f} s")
    print(f"  Peak accel:    {val['peak_accel_filt']:.1f} m/s²")
    print(f"  SISFall:       {val['sisfall_compliant']}")
    print(f"  IMU CSV  ?     {imu_r['filename']}")
    print(f"  Validation ?   {rpt}")
    print("="*70)

    return {
        "scenario_id":    20,
        "classification": val["classification"],
        "score":          val["score"],
        "sisfall":        val["sisfall_compliant"],
        "imu_csv":        imu_r["filename"],
        "validation_txt": rpt,
    }


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run({"age": 70, "height": 1.65, "sex": "female", "weight": None})