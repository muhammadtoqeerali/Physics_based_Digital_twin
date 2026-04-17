# myosuite_age_profile.py
# Re-export shim — everything now lives in biofidelic_profile.py
from biofidelic_profile import (
    get_sarcopenia_params,
    get_myosuite_fall_weakening,
    hill_fv_scale,
    apply_joint_aging,
    MyoSuiteMuscleDynamics,
    myosuite_weakening_config,
)
__all__ = [
    'get_sarcopenia_params','get_myosuite_fall_weakening',
    'hill_fv_scale','apply_joint_aging',
    'MyoSuiteMuscleDynamics','myosuite_weakening_config',
]
