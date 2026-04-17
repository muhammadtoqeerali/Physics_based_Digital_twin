# -*- coding: utf-8 -*-
"""
biofidelic_profile.py
Continuous, literature-grounded anthropometric parameterisation for the
humanoid fall simulator.

This module is adapted from the integration checklist supplied by the user,
with a thin compatibility layer for backward_fall_walking_best_weight.py.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

# =============================================================================
# 1. DE LEVA (1996) SEGMENT MASS FRACTIONS - sex-stratified
# =============================================================================

_DE_LEVA_MALE = {
    'head':      0.0694,
    'trunk':     0.4346,
    'thorax':    0.4346,
    'torso':     0.4346,
    'spine':     0.4346,
    'abdomen':   0.1390,
    'pelvis':    0.1422,
    'upperarm':  0.0271,
    'forearm':   0.0162,
    'hand':      0.0061,
    'thigh':     0.1000,
    'shank':     0.0465,
    'foot':      0.0145,
}

_DE_LEVA_FEMALE = {
    'head':      0.0668,
    'trunk':     0.4257,
    'thorax':    0.4257,
    'torso':     0.4257,
    'spine':     0.4257,
    'abdomen':   0.1390,
    'pelvis':    0.1247,
    'upperarm':  0.0255,
    'forearm':   0.0138,
    'hand':      0.0056,
    'thigh':     0.1478,
    'shank':     0.0481,
    'foot':      0.0129,
}


def get_segment_mass_fractions(sex: str = 'male') -> dict:
    return dict(_DE_LEVA_FEMALE if str(sex).lower() == 'female' else _DE_LEVA_MALE)


# =============================================================================
# 2. CONTINUOUS GAIT SPEED
# =============================================================================

def gait_speed_mps(age: float, sex: str, height_m: float) -> float:
    age = float(age)
    height_m = float(height_m)
    sex = str(sex).lower()

    L_leg_ref = 0.530 * 1.75
    L_leg = 0.530 * height_m
    froude_scale = np.sqrt(L_leg / L_leg_ref)

    v_ref_40 = 1.40 * froude_scale
    v_ref_25 = v_ref_40 / (1.0 - 15 * 0.0032)

    if age <= 40:
        decline = max(0.0, age - 25) * 0.0005
    elif age <= 60:
        decline = 15 * 0.0005 + (age - 40) * 0.0032
    elif age <= 75:
        decline = 15 * 0.0005 + 20 * 0.0032 + (age - 60) * 0.0040
    else:
        decline = 15 * 0.0005 + 20 * 0.0032 + 15 * 0.0040 + (age - 75) * 0.0048

    v = v_ref_25 * (1.0 - decline)
    if sex == 'female':
        v *= 0.93
    return float(np.clip(v, 0.35, 2.20))


# =============================================================================
# 3. CONTINUOUS DOUBLE-SUPPORT FRACTION
# =============================================================================

def double_support_fraction(age: float, sex: str) -> float:
    age = float(age)
    sex = str(sex).lower()
    base = 0.19
    rate = 0.0035 if sex == 'male' else 0.00380
    ds = base + max(0.0, age - 28) * rate
    return float(np.clip(ds, 0.17, 0.55))


# =============================================================================
# 4. MUSCLE STRENGTH FACTOR
# =============================================================================

def muscle_strength_factor(age: float, sex: str, body_mass_kg: float = 70.0, height_m: float = 1.75) -> float:
    age = float(age)
    sex = str(sex).lower()
    body_mass_kg = float(max(body_mass_kg, 25.0))

    peak_male = 200.0
    sex_ratio = 0.622
    ref_peak = peak_male if sex == 'male' else peak_male * sex_ratio

    if age <= 30:
        torque = ref_peak
    elif age <= 60:
        rate_per_yr = 0.0080 if sex == 'male' else 0.0090
        torque = ref_peak * np.exp(-rate_per_yr * (age - 30))
    else:
        torque_at_60 = ref_peak * np.exp(-(0.0080 if sex == 'male' else 0.0090) * 30.0)
        rate_post60 = 0.0150 if sex == 'male' else 0.0140
        torque = torque_at_60 * np.exp(-rate_post60 * (age - 60))

    torque = max(torque, 30.0)
    ref_mass = 70.0
    allometric = (body_mass_kg / ref_mass) ** (2.0 / 3.0)
    strength = (torque * allometric) / peak_male
    return float(np.clip(strength, 0.22, 1.15))


# =============================================================================
# 5. REACTION DELAY
# =============================================================================

def reaction_delay_seconds(age: float, sex: str, height_m: float = 1.75) -> float:
    age = float(age)
    sex = str(sex).lower()
    height_m = float(height_m)

    base_rt = 0.200 if sex == 'male' else 0.185
    age_slowing = max(0.0, age - 30.0) * (0.0040 if sex == 'male' else 0.0035)
    crt = (base_rt + age_slowing) * 1.60
    height_conduction = max(0.0, height_m - 1.70) * 0.013
    return float(np.clip(crt + height_conduction, 0.18, 0.58))


# =============================================================================
# 6. BALANCE IMPAIRMENT COMPOSITE
# =============================================================================

def balance_impairment(age: float, sex: str, body_mass_kg: float = 70.0) -> float:
    age = float(age)
    sex = str(sex).lower()
    sf = muscle_strength_factor(age, sex, body_mass_kg)
    strength_loss = float(np.clip(1.0 - sf, 0.0, 0.80))
    proprio_loss = float(np.clip(max(0.0, (age - 50.0) / 40.0) ** 1.5, 0.0, 0.50))
    vest_loss = float(np.clip(max(0.0, (age - 65.0) / 25.0) ** 1.2, 0.0, 0.35))
    fear_onset_age = 60.0 if sex == 'female' else 65.0
    fear_loss = float(np.clip(max(0.0, (age - fear_onset_age) / 25.0), 0.0, 0.20))
    composite = 0.35 * strength_loss + 0.30 * proprio_loss + 0.20 * vest_loss + 0.15 * fear_loss
    return float(np.clip(composite, 0.0, 0.80))


# =============================================================================
# 7. FORWARD STOOP ANGLE
# =============================================================================

def forward_stoop_angle_deg(age: float, sex: str) -> float:
    age = float(age)
    sex = str(sex).lower()
    base_stand = 4.5 if sex == 'male' else 5.0
    rate = 0.120 if sex == 'male' else 0.160
    stoop = base_stand + max(0.0, age - 30.0) * rate
    return float(np.clip(stoop, 3.5, 22.0))


# =============================================================================
# 8. PERTURBATION FORCE
# =============================================================================

def perturbation_force_config(body_mass_kg: float, age: float, sex: str, bw_fraction: float = 1.15) -> dict:
    bw = float(body_mass_kg) * 9.81
    magnitude = float(np.clip(bw_fraction * bw, 200.0, 1600.0))
    return {
        'magnitude': magnitude,
        'ramp_up': 30,
        'direction': np.array([-1.0, 0.0, 0.30], dtype=float),
        'application_point': 'Pelvis',
        'bw_fraction': float(bw_fraction),
        'body_mass_kg': float(body_mass_kg),
    }


# =============================================================================
# 9. WEAKENING CONFIG
# =============================================================================

def weakening_config(age: float, sex: str, body_mass_kg: float = 70.0) -> dict:
    age = float(age)
    sex = str(sex).lower()
    age_frac = float(np.clip((age - 60.0) / 35.0, 0.0, 1.0))
    min_factor = 0.05 - 0.025 * age_frac
    if sex == 'female' and age >= 65:
        min_factor -= 0.005 * min(1.0, (age - 65.0) / 20.0)
    min_factor = float(np.clip(min_factor, 0.02, 0.07))
    decay_time = int(round(45 - 17 * age_frac))
    decay_time = max(20, decay_time)
    return {
        'initial_factor': 1.0,
        'min_factor': min_factor,
        'decay_time': decay_time,
        'recovery_time': 120,
    }


# =============================================================================
# 10. PHASE TIMING
# =============================================================================

def phase_timing(age: float, sex: str) -> dict:
    age = float(age)
    age_frac = float(np.clip((age - 30.0) / 60.0, 0.0, 1.0))
    stand_steps = int(round(150 + 60 * age_frac))
    walk_steps = 300
    perturb_steps = 60
    rt = reaction_delay_seconds(age, sex)
    react_steps = max(12, int(round(rt * 30.0 * 1.8)))
    fall_steps = int(round(380 + 120 * age_frac))
    return {
        'stand': stand_steps,
        'walk': walk_steps,
        'perturb': perturb_steps,
        'react': react_steps,
        'fall': fall_steps,
    }


# =============================================================================
# 11. MASTER FUNCTION: get_age_style_v2
# =============================================================================

def get_age_style_v2(age_years: float, height_m: float = 1.75, sex: str = 'male', body_mass_kg: float = 70.0) -> dict:
    age = float(age_years)
    height_m = float(height_m)
    sex = str(sex).lower()
    body_mass_kg = float(body_mass_kg)

    target_walk_speed = gait_speed_mps(age, sex, height_m)
    expected_ds = double_support_fraction(age, sex)
    stoop_stand_deg = forward_stoop_angle_deg(age, sex)
    stoop_walk_deg = stoop_stand_deg + 1.5

    sf = muscle_strength_factor(age, sex, body_mass_kg, height_m)
    bal = balance_impairment(age, sex, body_mass_kg)
    rt = reaction_delay_seconds(age, sex, height_m)

    walk_leg_gain = float(np.clip(0.88 + 0.12 * sf, 0.88, 1.00))
    stand_leg_gain = float(np.clip(0.90 + 0.10 * sf, 0.90, 1.00))

    arm_gain = float(np.clip(1.00 - 0.30 * max(0.0, age - 30.0) / 50.0, 0.70, 1.00))
    if sex == 'female':
        arm_gain *= 0.96

    older_weight = float(np.clip(0.20 + 0.25 * max(0.0, age - 30.0) / 60.0, 0.20, 0.45))
    smoothing_weights = np.array([older_weight, 1.0 - older_weight], dtype=float)

    policy_speed_blend = float(np.clip(0.40 + 0.25 * max(0.0, age - 30.0) / 60.0, 0.40, 0.65))
    control_speed_gain = float(np.clip(sf * 1.02, 0.88, 1.02))
    max_speed_ratio = float(np.clip(2.30 - 0.65 * max(0.0, age - 30.0) / 60.0, 1.65, 2.30))

    guidance_lat_scale = float(np.clip(1.40 - 0.20 * max(0.0, age - 30.0) / 60.0, 1.15, 1.40))
    guidance_yaw_scale = float(np.clip(1.60 - 0.25 * max(0.0, age - 30.0) / 60.0, 1.30, 1.60))
    guidance_alpha = float(np.clip(0.72 + 0.12 * max(0.0, age - 30.0) / 60.0, 0.72, 0.84))
    braking_scale = float(np.clip(1.00 + 0.45 * max(0.0, age - 30.0) / 60.0, 1.00, 1.45))
    speed_supervision_gain = float(np.clip(0.26 - 0.10 * max(0.0, age - 25.0) / 65.0, 0.16, 0.26))

    if age < 35:
        label = 'under_35'
    elif age < 55:
        label = '35_to_55'
    elif age < 70:
        label = '55_to_70'
    else:
        label = '70_plus'

    return {
        'label': label,
        'age': age,
        'age_years': age,
        'sex': sex,
        'height_m': height_m,
        'body_mass_kg': body_mass_kg,
        'target_walk_speed': target_walk_speed,
        'literature_target_speed': target_walk_speed,
        'expected_double_support': expected_ds,
        'stand_stoop_target_deg': stoop_stand_deg,
        'walk_stoop_target_deg': stoop_walk_deg,
        'stoop_target_deg': stoop_walk_deg,
        'walk_leg_gain': walk_leg_gain,
        'stand_leg_gain': stand_leg_gain,
        'arm_gain': arm_gain,
        'smoothing_weights': smoothing_weights,
        'control_speed_gain': control_speed_gain,
        'policy_speed_blend': policy_speed_blend,
        'max_speed_ratio': max_speed_ratio,
        'min_adapted_speed': float(np.clip(0.50 * target_walk_speed, 0.18, 0.60)),
        'guidance_lat_scale': guidance_lat_scale,
        'guidance_lateral_scale': guidance_lat_scale,
        'guidance_yaw_scale': guidance_yaw_scale,
        'guidance_alpha': guidance_alpha,
        'speed_supervision_gain': speed_supervision_gain,
        'braking_scale': braking_scale,
        'muscle_strength_factor': sf,
        'strength_factor_raw': sf,
        'balance_impairment': bal,
        'reaction_delay_s': rt,
        'reaction_delay': rt,
        'proprioception_scale': float(np.clip(1.0 - 0.5 * bal, 0.50, 1.00)),
    }


# =============================================================================
# 12. APPLY AGE EFFECTS PATCH
# =============================================================================

def apply_age_effects_v2(mj_model, age: float, sex: str, height: float, body_mass_kg: Optional[float] = None, original_gear: Optional[np.ndarray] = None) -> dict:
    sf = muscle_strength_factor(age, sex, body_mass_kg if body_mass_kg else 70.0, height)
    rt = reaction_delay_seconds(age, sex, height)
    bim = balance_impairment(age, sex, body_mass_kg if body_mass_kg else 70.0)
    baseline_sf = float(np.clip(sf, 0.85, 1.15))

    if original_gear is not None:
        mj_model.actuator_gear[:, 0] = np.asarray(original_gear, dtype=float) * baseline_sf
    else:
        for i in range(mj_model.nu):
            mj_model.actuator_gear[i, 0] *= baseline_sf

    proprio_scale = float(np.clip(1.0 - 0.55 * bim, 0.45, 1.00))
    return {
        'strength_factor': baseline_sf,
        'strength_factor_raw': float(sf),
        'reaction_delay': float(rt),
        'reaction_delay_s': float(rt),
        'balance_impairment': float(bim),
        'proprioception_scale': proprio_scale,
        'height_penalty': float(max(0.0, (height - 1.5) * 0.002)),
        'age_penalty': float(max(0.0, age - 30.0) * 0.0005),
        'leg_length': float(height * 0.530),
        'body_mass_kg': float(body_mass_kg) if body_mass_kg else None,
        'age_years': float(age),
        'height_m': float(height),
        'sex': str(sex),
    }


# =============================================================================
# 13. AGE REFERENCE BAND
# =============================================================================

def get_age_reference_band_v2(age: float, sex: str, height_m: float, body_mass_kg: float = 70.0) -> dict:
    speed = gait_speed_mps(age, sex, height_m)
    ds = double_support_fraction(age, sex)
    sd_spd = 0.18
    sd_ds = 0.05
    label = get_age_style_v2(age, height_m, sex, body_mass_kg)['label']
    return {
        'label': label,
        'comfortable_speed_band_mps': (
            float(np.clip(speed - sd_spd, 0.25, 2.0)),
            float(np.clip(speed + sd_spd, 0.40, 2.2)),
        ),
        'double_support_band': (
            float(np.clip(ds - sd_ds, 0.12, 0.50)),
            float(np.clip(ds + sd_ds, 0.20, 0.60)),
        ),
        'reference_speed_mps': speed,
        'reference_double_support': ds,
    }


# =============================================================================
# 14. SELF-CONSISTENCY AUDIT
# =============================================================================

def print_subject_profile(age: float, sex: str, height_m: float, body_mass_kg: float, verbose: bool = True) -> dict:
    style = get_age_style_v2(age, height_m, sex, body_mass_kg)
    pert = perturbation_force_config(body_mass_kg, age, sex)
    weak = weakening_config(age, sex, body_mass_kg)
    phases = phase_timing(age, sex)
    fracs = get_segment_mass_fractions(sex)
    bmi = body_mass_kg / max(height_m ** 2, 0.01)
    bw_n = body_mass_kg * 9.81

    if verbose:
        print('=' * 65)
        print('  BIOFIDELIC SUBJECT PROFILE v2')
        print(f'  age={age}y  sex={sex}  h={height_m:.2f}m  m={body_mass_kg:.1f}kg  BMI={bmi:.1f}')
        print('=' * 65)
        print('  GAIT')
        print(f"    walk speed                     : {style['target_walk_speed']:.3f} m/s")
        print(f"    double support                : {style['expected_double_support']:.1%}")
        print(f"    stoop stand/walk              : {style['stand_stoop_target_deg']:.1f}Ã‚Â° / {style['walk_stoop_target_deg']:.1f}Ã‚Â°")
        print('  MUSCLE / BALANCE')
        print(f"    strength factor               : {style['muscle_strength_factor']:.3f}")
        print(f"    arm gain                      : {style['arm_gain']:.3f}")
        print(f"    balance impairment            : {style['balance_impairment']:.3f}")
        print('  NEURAL')
        print(f"    reaction delay                : {style['reaction_delay_s'] * 1000:.0f} ms")
        print(f"    proprioception scale          : {style['proprioception_scale']:.3f}")
        print('  PERTURBATION')
        print(f"    magnitude                     : {pert['magnitude']:.1f} N ({pert['magnitude']/bw_n:.2f} BW)")
        print('  WEAKENING')
        print(f"    min strength floor            : {weak['min_factor']:.3f}")
        print(f"    decay time                    : {weak['decay_time']} steps ({weak['decay_time']/30:.2f}s)")
        print('  PHASE TIMING (steps @ 30 Hz)')
        print(f"    stand={phases['stand']}  walk={phases['walk']}  perturb={phases['perturb']}  react={phases['react']}  fall={phases['fall']}")
        print('  SEGMENT FRACTIONS')
        print(f"    thigh={fracs['thigh']:.4f}  trunk={fracs['trunk']:.4f}  head={fracs['head']:.4f}  foot={fracs['foot']:.4f}")
        print('=' * 65)

    return {
        'style': style,
        'perturbation': pert,
        'weakening': weak,
        'phases': phases,
        'seg_fracs': fracs,
    }


if __name__ == '__main__':
    cases = [
        (25, 'male', 1.80, 78.0),
        (50, 'female', 1.62, 65.0),
        (75, 'male', 1.72, 80.0),
        (80, 'female', 1.58, 62.0),
    ]
    for age, sex, h, m in cases:
        print_subject_profile(age, sex, h, m)
        print()
