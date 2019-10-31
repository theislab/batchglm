from enum import Enum


class TrainingStrategies(Enum):

    AUTO = None
    DEFAULT = [
        {
            "max_steps": 1000,
            "update_b_freq": 5
        },
    ]
