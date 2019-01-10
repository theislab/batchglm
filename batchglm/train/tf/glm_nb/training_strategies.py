from enum import Enum

class TrainingStrategies(Enum):

    AUTO = None
    DEFAULT = [
        {
            "convergence_criteria": "all_converged_ll",
            "stopping_criteria": 1e-6,
            "use_batching": False,
            "optim_algo": "irls",
        },
    ]
    QUICK = [
        {
            "convergence_criteria": "all_converged_ll",
            "stopping_criteria": 1e-3,
            "use_batching": True,
            "optim_algo": "irls",
        },
        {
            "convergence_criteria": "all_converged_ll",
            "stopping_criteria": 1e-6,
            "use_batching": False,
            "optim_algo": "irls",
        },
    ]
    INEXACT = [
        {
            "convergence_criteria": "all_converged_ll",
            "stopping_criteria": 1e-4,
            "use_batching": False,
            "optim_algo": "irls",
        },
    ]
    EXACT = [
        {
            "convergence_criteria": "all_converged_ll",
            "stopping_criteria": 1e-8,
            "use_batching": False,
            "optim_algo": "irls",
        },
    ]