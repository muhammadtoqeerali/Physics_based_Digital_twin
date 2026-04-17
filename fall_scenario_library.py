# -*- coding: utf-8 -*-
"""
fall_scenario_library.py
========================
Biomechanically grounded scenario definitions for 22 fall types.

Each scenario is a Python dataclass that fully characterises:
  - Initial locomotion state (walking / sitting / elevated / backward-moving)
  - Perturbation mechanism (push | trip | slip | faint | step-off)
  - Fall direction and lie-down orientation
  - Phase timing overrides
  - Scenario-specific physics hooks (friction, foot blocking, muscle collapse)

References
----------
  Casilari et al. (2017) Sensors. Multi-sensor SISFall analysis.
  Winter (1990) Biomechanics and motor control of human movement.
  Hof et al. (2005) J. Biomech. XCoM / capture-point theory.
  Muybridge (1887) Animal locomotion — gait kinematics reference.
  Strandberg & Lanshammar (1981) Human Factors. Slip mechanics.
  Grieve (1968) J. Physiol. Jogging kinematics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np


# -----------------------------------------------------------------------------
# ENUMERATIONS (plain string constants — no enum overhead)
# -----------------------------------------------------------------------------
class FallDirection:
    FORWARD       = "forward"
    BACKWARD      = "backward"
    LATERAL_LEFT  = "lateral_left"
    LATERAL_RIGHT = "lateral_right"

class InitialState:
    WALKING          = "walking"
    JOGGING          = "jogging"
    SITTING_DOWN     = "sitting_down"    # mid-lowering motion
    GETTING_UP       = "getting_up"      # mid-rising motion
    SITTING          = "sitting"         # static seated
    STANDING         = "standing"
    MOVING_BACKWARD  = "moving_backward"
    ELEVATED         = "elevated"        # on a step / ladder

class PerturbationType:
    EXTERNAL_PUSH    = "push"        # xfrc_applied force
    TRIP             = "trip"        # abrupt foot blocking
    SLIP             = "slip"        # floor-friction reduction
    FAINT            = "faint"       # sudden muscle collapse
    STEP_OFF         = "step_off"    # leave elevated surface


# -----------------------------------------------------------------------------
# SCENARIO CONFIG DATACLASS
# -----------------------------------------------------------------------------
@dataclass
class ScenarioConfig:
    # -- Identity -------------------------------------------------------------
    scenario_id:   int   = 34
    description:   str   = "Backward fall while walking caused by a slip"
    category:      str   = "Moving falls"

    # -- Physics state --------------------------------------------------------
    initial_state: str   = InitialState.WALKING
    fall_direction:str   = FallDirection.BACKWARD

    # -- Perturbation ---------------------------------------------------------
    perturbation_type:       str             = PerturbationType.EXTERNAL_PUSH
    force_direction:         np.ndarray      = field(default_factory=lambda: np.array([-1.0, 0.0, 0.3]))
    force_bw_fraction:       float           = 1.15   # fraction of body-weight
    application_point:       str             = "Pelvis"
    perturbation_timing:     str             = "mid_stance"
    force_ramp_steps:        int             = 30

    # -- Slip parameters ------------------------------------------------------
    slip_friction_coeff:     Optional[float] = None   # floor µ during slip
    slip_lateral_only:       bool            = False  # for lateral slip

    # -- Trip parameters ------------------------------------------------------
    trip_blocking_bw:        float           = 8.0    # foot blocking force / BW
    trip_duration_steps:     int             = 6
    trip_foot:               str             = "L_Ankle"  # leading foot body

    # -- Faint parameters -----------------------------------------------------
    faint_collapse_steps:    int             = 20     # steps to full collapse
    faint_residual_gear:     float           = 0.00   # remaining actuator gear
    faint_direction_force:   Optional[np.ndarray] = None  # small bias
    protective_arms:         bool            = False  # keep arm actuators active

    # -- Initial conditions ---------------------------------------------------
    initial_height_offset:   float           = 0.0   # extra pelvis Z in metres
    initial_backward_speed:  float           = 0.0   # m/s for backward-moving
    initial_pitch_deg:       float           = 0.0   # body tilt (ladder)
    jogging_speed:           float           = 2.1   # m/s for jogging scenarios

    # -- Sit/stand transition -------------------------------------------------
    # Approximate sitting via a "crouch" phase: pelvis lowered to this height
    sit_target_pelvis_z:     float           = 0.48  # m (chair ~0.45 m)
    sit_crouch_steps:        int             = 60    # steps to reach seat height
    sit_direction_bias:      np.ndarray      = field(default_factory=lambda: np.array([0.0, 0.0, -0.5]))

    # -- Lie-down reward orientation ------------------------------------------
    lie_orient:              str             = "up"   # up|down|left|right

    # -- Z-embedding speed reference ------------------------------------------
    z_walk_speed_ref:        float           = 1.44  # literature reference speed
    z_jog_speed_ref:         float           = 2.20  # jogging reference

    # -- Phase timing tweaks (None = use biofidelic_profile defaults) ---------
    stand_steps_override:    Optional[int]   = None
    walk_steps_override:     Optional[int]   = None
    perturb_steps_override:  Optional[int]   = None
    react_steps_override:    Optional[int]   = None
    fall_steps_override:     Optional[int]   = None

    # -- Misc -----------------------------------------------------------------
    save_imu_csv:            bool            = True
    save_biomechanics:       bool            = True
    viewer_enabled:          bool            = True
    use_avatar_frame:       bool            = False  # interpret force/bias in avatar local frame

    def resolve_phases(self, base_phases: dict) -> dict:
        """Merge per-scenario overrides into the biofidelic base phases."""
        p = dict(base_phases)
        if self.stand_steps_override   is not None: p["stand"]   = self.stand_steps_override
        if self.walk_steps_override    is not None: p["walk"]    = self.walk_steps_override
        if self.perturb_steps_override is not None: p["perturb"] = self.perturb_steps_override
        if self.react_steps_override   is not None: p["react"]   = self.react_steps_override
        if self.fall_steps_override    is not None: p["fall"]    = self.fall_steps_override
        return p


# -----------------------------------------------------------------------------
# FACTORY FUNCTIONS — one per scenario ID
# -----------------------------------------------------------------------------

def make_scenario_20() -> ScenarioConfig:
    """Forward fall when trying to sit down.
    Uses avatar-local directions so the task remains forward regardless of yaw.
    """
    return ScenarioConfig(
        scenario_id=20,
        description="Forward fall when trying to sit down",
        category="Sitting transitions",
        initial_state=InitialState.SITTING_DOWN,
        fall_direction=FallDirection.FORWARD,
        perturbation_type=PerturbationType.EXTERNAL_PUSH,
        force_direction=np.array([1.0, 0.0, -0.18]),   # local [forward, lateral, vertical]
        force_bw_fraction=0.78,
        application_point="Torso",
        perturbation_timing="mid_stance",
        force_ramp_steps=18,
        lie_orient="down",
        sit_target_pelvis_z=0.50,
        sit_crouch_steps=55,
        sit_direction_bias=np.array([0.35, 0.0, -0.70]),
        stand_steps_override=120,
        walk_steps_override=0,
        perturb_steps_override=55,
        react_steps_override=12,
        fall_steps_override=360,
        z_walk_speed_ref=0.0,
        use_avatar_frame=True,
    )


def make_scenario_21() -> ScenarioConfig:
    """Backward fall when trying to sit down.
    Biomechanics: Subject leans too far back while lowering ? posterior CoM
    displacement outside heel support. Quadriceps cannot recover.
    """
    return ScenarioConfig(
        scenario_id=21,
        description="Backward fall when trying to sit down",
        category="Sitting transitions",
        initial_state=InitialState.SITTING_DOWN,
        fall_direction=FallDirection.BACKWARD,
        perturbation_type=PerturbationType.EXTERNAL_PUSH,
        force_direction=np.array([-0.6, 0.0, -0.2]),
        force_bw_fraction=0.70,
        application_point="Pelvis",
        perturbation_timing="mid_stance",
        force_ramp_steps=15,
        lie_orient="up",
        sit_target_pelvis_z=0.50,
        sit_crouch_steps=55,
        sit_direction_bias=np.array([-0.05, 0.0, -0.6]),
        stand_steps_override=120,
        walk_steps_override=0,
        perturb_steps_override=55,
        react_steps_override=12,
        fall_steps_override=360,
        z_walk_speed_ref=0.0,
    )


def make_scenario_22() -> ScenarioConfig:
    """Lateral fall when trying to sit down.
    Biomechanics: Asymmetric weight transfer; medial knee adductor failure.
    """
    return ScenarioConfig(
        scenario_id=22,
        description="Lateral fall when trying to sit down",
        category="Sitting transitions",
        initial_state=InitialState.SITTING_DOWN,
        fall_direction=FallDirection.LATERAL_LEFT,
        perturbation_type=PerturbationType.EXTERNAL_PUSH,
        force_direction=np.array([0.0, -0.8, -0.1]),
        force_bw_fraction=0.60,
        application_point="Pelvis",
        perturbation_timing="mid_stance",
        force_ramp_steps=12,
        lie_orient="left",
        sit_target_pelvis_z=0.50,
        sit_crouch_steps=55,
        sit_direction_bias=np.array([0.0, -0.05, -0.6]),
        stand_steps_override=120,
        walk_steps_override=0,
        perturb_steps_override=55,
        react_steps_override=12,
        fall_steps_override=360,
        z_walk_speed_ref=0.0,
    )


def make_scenario_23() -> ScenarioConfig:
    """Forward fall when trying to get up.
    Biomechanics: Rising from chair; momentum carries COM too far forward;
    ankle strategy insufficient ? forward fall from partial standing height.
    """
    return ScenarioConfig(
        scenario_id=23,
        description="Forward fall when trying to get up",
        category="Sitting transitions",
        initial_state=InitialState.GETTING_UP,
        fall_direction=FallDirection.FORWARD,
        perturbation_type=PerturbationType.EXTERNAL_PUSH,
        force_direction=np.array([0.7, 0.0, -0.1]),
        force_bw_fraction=0.55,
        application_point="Torso",
        perturbation_timing="mid_stance",
        force_ramp_steps=15,
        lie_orient="down",
        sit_target_pelvis_z=0.50,
        sit_crouch_steps=55,
        sit_direction_bias=np.array([0.0, 0.0, -0.6]),
        stand_steps_override=120,
        walk_steps_override=0,
        perturb_steps_override=55,
        react_steps_override=12,
        fall_steps_override=360,
        initial_height_offset=-0.35,   # start in crouched/seated height
        z_walk_speed_ref=0.0,
    )


def make_scenario_24() -> ScenarioConfig:
    """Lateral fall when trying to get up.
    Biomechanics: Rotational instability during STS (sit-to-stand); hip
    abductors fail to stabilise pelvis laterally.
    """
    return ScenarioConfig(
        scenario_id=24,
        description="Lateral fall when trying to get up",
        category="Sitting transitions",
        initial_state=InitialState.GETTING_UP,
        fall_direction=FallDirection.LATERAL_RIGHT,
        perturbation_type=PerturbationType.EXTERNAL_PUSH,
        force_direction=np.array([0.0, 0.8, -0.1]),
        force_bw_fraction=0.55,
        application_point="Pelvis",
        perturbation_timing="mid_stance",
        force_ramp_steps=12,
        lie_orient="right",
        sit_target_pelvis_z=0.50,
        sit_crouch_steps=55,
        sit_direction_bias=np.array([0.0, 0.05, -0.6]),
        stand_steps_override=120,
        walk_steps_override=0,
        perturb_steps_override=55,
        react_steps_override=12,
        fall_steps_override=360,
        initial_height_offset=-0.35,
        z_walk_speed_ref=0.0,
    )


def make_scenario_25() -> ScenarioConfig:
    """Forward fall while sitting, caused by fainting.
    Biomechanics: Sudden syncope ? total neuromuscular inhibition ?
    anterior trunk collapse over extended legs.
    """
    return ScenarioConfig(
        scenario_id=25,
        description="Forward fall while sitting, caused by fainting",
        category="Fainting",
        initial_state=InitialState.SITTING,
        fall_direction=FallDirection.FORWARD,
        perturbation_type=PerturbationType.FAINT,
        force_direction=np.array([0.0, 0.0, 0.0]),
        force_bw_fraction=0.0,
        faint_collapse_steps=18,
        faint_residual_gear=0.0,
        faint_direction_force=np.array([0.08, 0.0, -0.04]),  # subtle anterior bias
        lie_orient="down",
        sit_target_pelvis_z=0.50,
        sit_crouch_steps=45,
        stand_steps_override=120,
        walk_steps_override=0,
        perturb_steps_override=45,
        react_steps_override=8,
        fall_steps_override=400,
        z_walk_speed_ref=0.0,
    )


def make_scenario_26() -> ScenarioConfig:
    """Lateral fall while sitting, caused by fainting.
    Biomechanics: Gravitational collapse with lateral tilt; no protective
    arm extension (unconscious).
    """
    return ScenarioConfig(
        scenario_id=26,
        description="Lateral fall while sitting, caused by fainting",
        category="Fainting",
        initial_state=InitialState.SITTING,
        fall_direction=FallDirection.LATERAL_LEFT,
        perturbation_type=PerturbationType.FAINT,
        force_direction=np.array([0.0, 0.0, 0.0]),
        force_bw_fraction=0.0,
        faint_collapse_steps=18,
        faint_residual_gear=0.0,
        faint_direction_force=np.array([0.0, -0.10, -0.04]),
        lie_orient="left",
        sit_target_pelvis_z=0.50,
        sit_crouch_steps=45,
        stand_steps_override=120,
        walk_steps_override=0,
        perturb_steps_override=45,
        react_steps_override=8,
        fall_steps_override=400,
        z_walk_speed_ref=0.0,
    )


def make_scenario_27() -> ScenarioConfig:
    """Backward fall while sitting, caused by fainting.
    Biomechanics: Posterior COM shift in seated subject; trunk falls backward.
    """
    return ScenarioConfig(
        scenario_id=27,
        description="Backward fall while sitting, caused by fainting",
        category="Fainting",
        initial_state=InitialState.SITTING,
        fall_direction=FallDirection.BACKWARD,
        perturbation_type=PerturbationType.FAINT,
        force_direction=np.array([0.0, 0.0, 0.0]),
        force_bw_fraction=0.0,
        faint_collapse_steps=18,
        faint_residual_gear=0.0,
        faint_direction_force=np.array([-0.08, 0.0, -0.04]),
        lie_orient="up",
        sit_target_pelvis_z=0.50,
        sit_crouch_steps=45,
        stand_steps_override=120,
        walk_steps_override=0,
        perturb_steps_override=45,
        react_steps_override=8,
        fall_steps_override=400,
        z_walk_speed_ref=0.0,
    )


def make_scenario_28() -> ScenarioConfig:
    """Forward fall while walking caused by fainting.
    Biomechanics: Walking momentum combined with sudden loss of anti-gravity
    muscle activation ? forward pitch, body-weight fall with no protective
    response.  (Khambhati 2019: mean fall velocity ˜ 1.7–2.5 m/s)
    """
    return ScenarioConfig(
        scenario_id=28,
        description="Vertical/forward fall while walking caused by fainting",
        category="Fainting",
        initial_state=InitialState.WALKING,
        fall_direction=FallDirection.FORWARD,
        perturbation_type=PerturbationType.FAINT,
        force_direction=np.array([0.0, 0.0, 0.0]),
        force_bw_fraction=0.0,
        faint_collapse_steps=12,
        faint_residual_gear=0.0,
        faint_direction_force=np.array([0.06, 0.0, -0.04]),
        lie_orient="down",
        z_walk_speed_ref=1.44,
    )


def make_scenario_29() -> ScenarioConfig:
    """Fall while walking using hands to dampen.
    Biomechanics: Subject stumbles / trips; conscious protective response
    extends arms before ground contact, reducing head/trunk impact by up to
    40% (DeGoede & Ashton-Miller 2002).  Arm actuators are kept partially
    active throughout the fall.
    """
    return ScenarioConfig(
        scenario_id=29,
        description="Fall while walking, hands used to dampen (protective reach-out)",
        category="Fainting",
        initial_state=InitialState.WALKING,
        fall_direction=FallDirection.FORWARD,
        perturbation_type=PerturbationType.EXTERNAL_PUSH,
        force_direction=np.array([0.5, 0.0, -0.3]),
        force_bw_fraction=0.80,
        application_point="Torso",
        perturbation_timing="mid_stance",
        force_ramp_steps=12,
        protective_arms=True,           # arms remain active
        lie_orient="down",
        z_walk_speed_ref=1.44,
    )


def make_scenario_30() -> ScenarioConfig:
    """Forward fall while walking caused by a trip.
    Biomechanics: Swing foot catches an obstacle at heel-strike; foot
    decelerated abruptly while COM continues forward (Eng et al. 1994).
    Blocking force: 5–10 BW applied to leading ankle for ~5 steps.
    """
    return ScenarioConfig(
        scenario_id=30,
        description="Forward fall while walking caused by a trip",
        category="Moving falls",
        initial_state=InitialState.WALKING,
        fall_direction=FallDirection.FORWARD,
        perturbation_type=PerturbationType.TRIP,
        force_direction=np.array([0.0, 0.0, 0.0]),  # trip uses foot blocking
        force_bw_fraction=0.0,
        perturbation_timing="heel_strike",
        trip_blocking_bw=8.0,
        trip_duration_steps=6,
        trip_foot="L_Ankle",
        lie_orient="down",
        z_walk_speed_ref=1.44,
    )


def make_scenario_31() -> ScenarioConfig:
    """Forward fall while jogging caused by a trip.
    Biomechanics: Higher COM velocity (˜2.1 m/s) at foot-catch creates
    greater angular momentum; faster and more violent forward rotation
    than walking trip (Pijnappels et al. 2005).
    """
    return ScenarioConfig(
        scenario_id=31,
        description="Forward fall while jogging caused by a trip",
        category="Moving falls",
        initial_state=InitialState.JOGGING,
        fall_direction=FallDirection.FORWARD,
        perturbation_type=PerturbationType.TRIP,
        force_direction=np.array([0.0, 0.0, 0.0]),
        force_bw_fraction=0.0,
        perturbation_timing="heel_strike",
        trip_blocking_bw=10.0,
        trip_duration_steps=5,
        trip_foot="L_Ankle",
        lie_orient="down",
        jogging_speed=2.1,
        z_walk_speed_ref=2.20,
        z_jog_speed_ref=2.20,
    )


def make_scenario_32() -> ScenarioConfig:
    """Forward fall while walking caused by a slip.
    Biomechanics: Stance foot slips BACKWARD (heel slide) ? anterior trunk
    rotation.  Floor µ reduced to 0.10 (Strandberg 1983 slip threshold).
    A small backward impulse on the rear foot initiates the slide.
    """
    return ScenarioConfig(
        scenario_id=32,
        description="Forward fall while walking caused by a slip",
        category="Moving falls",
        initial_state=InitialState.WALKING,
        fall_direction=FallDirection.FORWARD,
        perturbation_type=PerturbationType.SLIP,
        force_direction=np.array([0.3, 0.0, -0.1]),  # small forward bias
        force_bw_fraction=0.30,
        application_point="Torso",
        perturbation_timing="mid_stance",
        force_ramp_steps=10,
        slip_friction_coeff=0.10,
        lie_orient="down",
        z_walk_speed_ref=1.44,
    )


def make_scenario_33() -> ScenarioConfig:
    """Lateral fall while walking caused by a slip.
    Biomechanics: One foot slips medially; hip abductors cannot maintain
    balance; lateral collapse (Lockhart et al. 2005).
    """
    return ScenarioConfig(
        scenario_id=33,
        description="Lateral fall while walking caused by a slip",
        category="Moving falls",
        initial_state=InitialState.WALKING,
        fall_direction=FallDirection.LATERAL_LEFT,
        perturbation_type=PerturbationType.SLIP,
        force_direction=np.array([0.0, -0.6, 0.1]),
        force_bw_fraction=0.50,
        application_point="Pelvis",
        perturbation_timing="mid_stance",
        force_ramp_steps=10,
        slip_friction_coeff=0.08,
        slip_lateral_only=True,
        lie_orient="left",
        z_walk_speed_ref=1.44,
    )


def make_scenario_34() -> ScenarioConfig:
    """Backward fall while walking caused by a slip.
    Biomechanics: Lead foot slips anteriorly; posterior COM shift exceeds
    heel-extension limit.  This is the fully validated baseline scenario
    (v25-armquietsettle, 93–98% HIGH_CONFIDENCE across subject profiles).
    """
    return ScenarioConfig(
        scenario_id=34,
        description="Backward fall while walking caused by a slip",
        category="Moving falls",
        initial_state=InitialState.WALKING,
        fall_direction=FallDirection.BACKWARD,
        perturbation_type=PerturbationType.EXTERNAL_PUSH,
        force_direction=np.array([-1.0, 0.0, 0.3]),
        force_bw_fraction=1.15,
        application_point="Pelvis",
        perturbation_timing="mid_stance",
        force_ramp_steps=30,
        slip_friction_coeff=0.10,
        lie_orient="up",
        z_walk_speed_ref=1.44,
    )


def make_scenario_37() -> ScenarioConfig:
    """Backward fall while slowly moving back.
    Biomechanics: Backward locomotion is inherently unstable; slow-walking
    CoM velocity ˜ 0.5 m/s.  Small perturbation or obstacle causes
    uncontrolled backward rotation (Thigpen et al. 2000).
    """
    return ScenarioConfig(
        scenario_id=37,
        description="Backward fall while slowly moving back",
        category="Elevation",
        initial_state=InitialState.MOVING_BACKWARD,
        fall_direction=FallDirection.BACKWARD,
        perturbation_type=PerturbationType.EXTERNAL_PUSH,
        force_direction=np.array([-0.8, 0.0, 0.2]),
        force_bw_fraction=0.70,
        application_point="Torso",
        perturbation_timing="mid_stance",
        force_ramp_steps=20,
        initial_backward_speed=0.45,
        lie_orient="up",
        z_walk_speed_ref=0.60,
    )


def make_scenario_38() -> ScenarioConfig:
    """Backward fall while quickly moving back.
    Biomechanics: Higher backward velocity (˜1.0 m/s) limits recovery time;
    angular momentum of the falling body is substantially higher.
    """
    return ScenarioConfig(
        scenario_id=38,
        description="Backward fall while quickly moving back",
        category="Elevation",
        initial_state=InitialState.MOVING_BACKWARD,
        fall_direction=FallDirection.BACKWARD,
        perturbation_type=PerturbationType.EXTERNAL_PUSH,
        force_direction=np.array([-1.0, 0.0, 0.25]),
        force_bw_fraction=0.90,
        application_point="Pelvis",
        perturbation_timing="mid_stance",
        force_ramp_steps=15,
        initial_backward_speed=0.90,
        lie_orient="up",
        z_walk_speed_ref=1.00,
    )


def make_scenario_39() -> ScenarioConfig:
    """Forward fall from height (step, platform).
    Biomechanics: Subject at ~0.5 m elevation; tips forward off edge.
    Impact velocity ˜ v(2 g h) ˜ 3.1 m/s (free-fall component).
    """
    return ScenarioConfig(
        scenario_id=39,
        description="Forward fall from height (step/platform)",
        category="Elevation",
        initial_state=InitialState.ELEVATED,
        fall_direction=FallDirection.FORWARD,
        perturbation_type=PerturbationType.STEP_OFF,
        force_direction=np.array([0.5, 0.0, -0.4]),
        force_bw_fraction=0.60,
        application_point="Torso",
        perturbation_timing="mid_stance",
        force_ramp_steps=12,
        initial_height_offset=0.50,
        lie_orient="down",
        z_walk_speed_ref=1.44,
        stand_steps_override=100,
        walk_steps_override=0,
        perturb_steps_override=30,
        react_steps_override=8,
        fall_steps_override=420,
    )


def make_scenario_40() -> ScenarioConfig:
    """Backward fall from height.
    Biomechanics: Subject near edge; backward perturbation causes step-off.
    Free-fall plus rotational component creates high-energy posterior impact.
    """
    return ScenarioConfig(
        scenario_id=40,
        description="Backward fall from height (step/platform)",
        category="Elevation",
        initial_state=InitialState.ELEVATED,
        fall_direction=FallDirection.BACKWARD,
        perturbation_type=PerturbationType.STEP_OFF,
        force_direction=np.array([-0.8, 0.0, 0.3]),
        force_bw_fraction=0.80,
        application_point="Pelvis",
        perturbation_timing="mid_stance",
        force_ramp_steps=15,
        initial_height_offset=0.50,
        lie_orient="up",
        z_walk_speed_ref=1.44,
        stand_steps_override=100,
        walk_steps_override=0,
        perturb_steps_override=35,
        react_steps_override=8,
        fall_steps_override=440,
    )


def make_scenario_41() -> ScenarioConfig:
    """Backward fall while climbing UP a ladder.
    Biomechanics: At mid-climb height (~0.8 m), hand grip fails or foot
    misplaces; COM is posterior to feet ? backward rotation about the ladder.
    Forward body pitch of ~15° simulates ladder-face lean.
    """
    return ScenarioConfig(
        scenario_id=41,
        description="Backward fall while climbing up the ladder",
        category="Elevation",
        initial_state=InitialState.ELEVATED,
        fall_direction=FallDirection.BACKWARD,
        perturbation_type=PerturbationType.STEP_OFF,
        force_direction=np.array([-0.9, 0.0, 0.1]),
        force_bw_fraction=0.85,
        application_point="Torso",
        perturbation_timing="mid_stance",
        force_ramp_steps=10,
        initial_height_offset=0.75,
        initial_pitch_deg=15.0,       # leaning into ladder
        lie_orient="up",
        z_walk_speed_ref=0.80,
        stand_steps_override=100,
        walk_steps_override=0,
        perturb_steps_override=30,
        react_steps_override=10,
        fall_steps_override=460,
    )


def make_scenario_42() -> ScenarioConfig:
    """Backward fall while climbing DOWN a ladder.
    Biomechanics: Descending face-out; foot misses rung ? uncontrolled
    backward rotation from greater height (˜1.0 m). Higher impact energy
    than climbing-up scenario.
    """
    return ScenarioConfig(
        scenario_id=42,
        description="Backward fall while climbing down the ladder",
        category="Elevation",
        initial_state=InitialState.ELEVATED,
        fall_direction=FallDirection.BACKWARD,
        perturbation_type=PerturbationType.STEP_OFF,
        force_direction=np.array([-1.0, 0.0, 0.15]),
        force_bw_fraction=0.90,
        application_point="Pelvis",
        perturbation_timing="mid_stance",
        force_ramp_steps=8,
        initial_height_offset=1.00,
        initial_pitch_deg=5.0,
        lie_orient="up",
        z_walk_speed_ref=0.80,
        stand_steps_override=100,
        walk_steps_override=0,
        perturb_steps_override=28,
        react_steps_override=10,
        fall_steps_override=500,
    )


# -----------------------------------------------------------------------------
# REGISTRY
# -----------------------------------------------------------------------------
_FACTORY_MAP: dict[int, Callable[[], ScenarioConfig]] = {
    20: make_scenario_20,
    21: make_scenario_21,
    22: make_scenario_22,
    23: make_scenario_23,
    24: make_scenario_24,
    25: make_scenario_25,
    26: make_scenario_26,
    27: make_scenario_27,
    28: make_scenario_28,
    29: make_scenario_29,
    30: make_scenario_30,
    31: make_scenario_31,
    32: make_scenario_32,
    33: make_scenario_33,
    34: make_scenario_34,
    37: make_scenario_37,
    38: make_scenario_38,
    39: make_scenario_39,
    40: make_scenario_40,
    41: make_scenario_41,
    42: make_scenario_42,
}

def get_scenario(scenario_id: int) -> ScenarioConfig:
    """Return the ScenarioConfig for a given ID, or raise KeyError."""
    if scenario_id not in _FACTORY_MAP:
        raise KeyError(f"No scenario defined for ID {scenario_id}. "
                       f"Available: {sorted(_FACTORY_MAP.keys())}")
    return _FACTORY_MAP[scenario_id]()


def list_scenarios() -> list[tuple[int, str, str]]:
    """Return (id, description, category) for every registered scenario."""
    return [(sid, fn().description, fn().category)
            for sid, fn in sorted(_FACTORY_MAP.items())]


# -----------------------------------------------------------------------------
# PERTURBATION PHYSICS HELPERS
# -----------------------------------------------------------------------------

def apply_slip_to_floor(mj_model, friction_coeff: float, lateral_only: bool = False):
    """Reduce floor geom friction to simulate a slippery surface.

    Parameters
    ----------
    friction_coeff:  sliding friction µ (0.08-0.15 for ice/wet floor)
    lateral_only:    if True only reduce lateral (y-axis) component
    """
    import mujoco
    plane_type = getattr(getattr(mujoco, "mjtGeom", object), "mjGEOM_PLANE", None)
    for g in range(mj_model.ngeom):
        gname = (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        is_floor = ("floor" in gname or "ground" in gname or "plane" in gname or
                    (plane_type is not None and mj_model.geom_type[g] == plane_type))
        if is_floor:
            fr = mj_model.geom_friction[g].copy()
            if lateral_only:
                fr[1] = friction_coeff  # only lateral torsional component
            else:
                fr[0] = friction_coeff
                fr[1] = max(fr[1] * 0.3, 0.003)
            mj_model.geom_friction[g] = fr


def apply_trip_blocking(mj_model, mj_data, body_name: str, blocking_bw: float,
                        body_mass: float):
    """Apply a sudden backward+downward blocking force on the foot body.

    Simulates the foot catching an obstacle: a large impulsive force
    opposing forward motion is applied for `trip_duration_steps` steps
    by the caller.
    """
    import mujoco
    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        # Fallback — try common alternative names
        for alt in ("L_Foot", "FootL", "L_Ankle", "AnkleL", "LeftFoot"):
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, alt)
            if bid >= 0:
                break
    if bid < 0:
        return
    bw = body_mass * 9.81
    fx = -blocking_bw * bw   # opposing forward motion
    fz = -0.20 * bw           # slight downward pressing
    mj_data.xfrc_applied[bid, 0] += fx
    mj_data.xfrc_applied[bid, 2] += fz


def apply_faint_collapse(mj_model, original_gear: np.ndarray,
                          step_in_faint: int, collapse_steps: int,
                          residual_gear: float = 0.0,
                          direction_force_body: str = "Torso",
                          direction_force: Optional[np.ndarray] = None,
                          body_mass: float = 70.0,
                          mj_data=None):
    """Progressive actuator-gear reduction to simulate syncope.

    Uses an exponential decay: gear(t) = g0 · exp(-k·t) where k is
    chosen so that after `collapse_steps` we reach `residual_gear`.
    """
    import mujoco
    progress = float(step_in_faint) / max(collapse_steps, 1)
    # Exponential collapse — feels more biological than linear
    factor = residual_gear + (1.0 - residual_gear) * np.exp(-5.0 * progress)
    factor = float(np.clip(factor, residual_gear, 1.0))
    mj_model.actuator_gear[:, 0] = original_gear * factor

    # Small directional bias (gravity does the rest)
    if direction_force is not None and mj_data is not None:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, direction_force_body)
        if bid >= 0:
            bw = body_mass * 9.81
            mj_data.xfrc_applied[bid, :3] = direction_force * bw * float(progress)


def elevate_initial_position(mj_data, height_offset: float, pitch_deg: float = 0.0):
    """Offset the root-joint position and optionally pitch the body.

    This must be called BEFORE the first env.reset() or immediately after
    an env.reset() so that qpos is mutable.

    qpos layout for humenv free joint: [x, y, z, qw, qx, qy, qz, ...]
    """
    if mj_data.qpos.shape[0] < 7:
        return
    mj_data.qpos[2] += float(height_offset)

    if abs(pitch_deg) > 0.1:
        # Apply pitch rotation to the root quaternion (qw, qx, qy, qz)
        import math
        half = math.radians(pitch_deg) * 0.5
        dq = np.array([math.cos(half), 0.0, math.sin(half), 0.0])  # pitch about Y
        # Quaternion multiply: q_new = dq ? q_root
        qw, qx, qy, qz = mj_data.qpos[3:7]
        nw = dq[0]*qw - dq[1]*qx - dq[2]*qy - dq[3]*qz
        nx = dq[0]*qx + dq[1]*qw + dq[2]*qz - dq[3]*qy
        ny = dq[0]*qy - dq[1]*qz + dq[2]*qw + dq[3]*qx
        nz = dq[0]*qz + dq[1]*qy - dq[2]*qx + dq[3]*qw
        mj_data.qpos[3:7] = [nw, nx, ny, nz]


def set_backward_velocity(mj_data, speed: float):
    """Imprint a backward initial velocity on the root DOF.

    qvel layout for free joint: [vx, vy, vz, wx, wy, wz, ...]
    A negative x-velocity drives the avatar backward.
    """
    if mj_data.qvel.shape[0] < 3:
        return
    mj_data.qvel[0] = -abs(float(speed))


def apply_sit_down_force(mj_model, mj_data, pelvis_id: int,
                          pelvis_z: float, target_z: float,
                          direction_bias: np.ndarray,
                          body_mass: float, step: int, total_steps: int):
    """Progressive downward + directional force to guide avatar into sitting.

    Applied to Torso/Pelvis. Magnitude scales with how far we still need
    to descend, then smoothly fades to allow policy to settle.
    """
    if pelvis_id < 0:
        return
    bw = body_mass * 9.81
    height_err = max(0.0, pelvis_z - target_z)
    ramp = float(step) / max(total_steps, 1)
    # Smooth bell: strong in middle, gentle at start & end
    smooth = np.sin(np.pi * ramp) ** 0.6
    down_force = -2.5 * bw * smooth * np.clip(height_err / 0.5, 0.0, 1.0)
    bias = np.asarray(direction_bias, dtype=float)
    bias_norm = float(np.linalg.norm(bias))
    bias_unit = bias / max(bias_norm, 1e-9)

    # Torso body
    torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "Torso")
    target_id = torso_id if torso_id >= 0 else pelvis_id
    mj_data.xfrc_applied[target_id, 2] += down_force
    mj_data.xfrc_applied[target_id, 0] += bias_unit[0] * abs(down_force) * 0.20
    mj_data.xfrc_applied[target_id, 1] += bias_unit[1] * abs(down_force) * 0.20

try:
    import mujoco
except ImportError:
    pass  # Allow import of this module without mujoco installed
