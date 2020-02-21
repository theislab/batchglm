from enum import Enum


class TrainingStrategies(Enum):

    AUTO = None
    DEFAULT = [
        {
            "max_steps": 1000,
            "method_b": "brent",
            "update_b_freq": 5,
            "ftol_b": 1e-6,
            "max_iter_b": 1000
        },
    ]
    GD = [
        {
            "max_steps": 1000,
            "method_b": "gd",
            "update_b_freq": 5,
            "ftol_b": 1e-6,
            "max_iter_b": 100
        },
    ]
