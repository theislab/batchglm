from enum import Enum


class TrainingStrategies(Enum):

    AUTO = None
    DEFAULT = [
        {
            "max_steps": 1000,
            "method_scale": "brent",
            "update_scale_freq": 5,
            "ftol_scale": 1e-6,
            "max_iter_scale": 1000,
        },
    ]
    GD = [
        {"max_steps": 1000, "method_scale": "gd", "update_scale_freq": 5, "ftol_scale": 1e-6, "max_iter_scale": 100},
    ]
