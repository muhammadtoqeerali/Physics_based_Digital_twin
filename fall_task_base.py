"""Shared high-level task architecture for per-fall modularization.

This file does not replace the existing simulator core by itself.
It defines the task contract that each isolated fall script should follow,
so each task can have its own motion logic instead of reusing the task-34 loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TaskAcceptance:
    pre_motion: str
    trigger_logic: str
    dominant_fall_direction: str
    expected_first_contact: str
    expected_final_pose: str


@dataclass(frozen=True)
class TaskPhase:
    name: str
    duration_steps: int
    description: str


@dataclass(frozen=True)
class TaskDefinition:
    task_id: int
    title: str
    category: str
    family: str
    description: str
    phases: Tuple[TaskPhase, ...]
    acceptance: TaskAcceptance
    notes: Tuple[str, ...] = field(default_factory=tuple)


TASK_FAMILY_RULES: Dict[str, str] = {
    "sit_down": "Use a stand->descent->seat-loading->loss-of-balance state machine.",
    "stand_up": "Use a seat->unweight->rise->instability state machine.",
    "seated_faint": "Use a seated posture followed by progressive support loss, not an external push.",
    "walking_trip": "Use gait with swing-foot obstruction and forward pitch.",
    "walking_slip": "Use gait with reduced friction and stance-foot excursion.",
    "walking_faint": "Use gait plus sudden bilateral support failure / torque collapse.",
    "reverse": "Use backward stepping before loss of balance.",
    "height": "Use elevated COM and downward drop mechanics.",
    "ladder": "Use climbing posture, elevated feet/hands pattern, then backward loss of support.",
}


TASK20 = TaskDefinition(
    task_id=20,
    title="Forward fall when trying to sit down",
    category="Sitting transitions",
    family="sit_down",
    description="Subject begins from quiet standing, rotates hips/knees to descend toward a virtual chair, begins seat-loading, then loses balance forward before stable sitting is achieved.",
    phases=(
        TaskPhase("stand", 120, "Quiet standing and stabilization."),
        TaskPhase("descent", 150, "Controlled sit-down descent with hip/knee flexion and forward trunk lean."),
        TaskPhase("seat_loading", 45, "Near-chair state where body mass begins to transfer rearward/downward."),
        TaskPhase("imbalance", 45, "Forward loss of balance triggered during incomplete seating."),
        TaskPhase("reaction", 18, "Delayed protective response."),
        TaskPhase("fall", 300, "Forward collapse and settlement."),
    ),
    acceptance=TaskAcceptance(
        pre_motion="Pelvis lowers and trunk leans forward before the fall trigger.",
        trigger_logic="Instability occurs during sit-down, not during walking and not from an arbitrary lateral shove.",
        dominant_fall_direction="Forward",
        expected_first_contact="Hands / knees / forearms or anterior torso depending recovery success.",
        expected_final_pose="Prone or semi-prone forward-down posture, not supine backward rest.",
    ),
    notes=(
        "A virtual chair proxy is acceptable if the environment lacks an actual chair mesh.",
        "Do not reuse the task-34 gait gate or mid-stance perturbation window.",
        "Validator should check pelvis descent before fall onset.",
    ),
)


TASKS: Dict[int, TaskDefinition] = {20: TASK20}
