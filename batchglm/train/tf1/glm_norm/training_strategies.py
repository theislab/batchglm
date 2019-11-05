from enum import Enum


class TrainingStrategies(Enum):

    AUTO = None
    DEFAULT = [
        {
            "convergence_criteria": "all_converged",
            "use_batching": False,
            "optim_algo": "irls_tr",
        },
    ]
    IRLS = [
        {
            "convergence_criteria": "all_converged",
            "use_batching": False,
            "optim_algo": "irls_tr",
        },
    ]
    IRLS_BATCHED = [
        {
            "convergence_criteria": "all_converged",
            "use_batching": True,
            "optim_algo": "irls_tr",
        },
    ]
