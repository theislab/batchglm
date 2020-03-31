from enum import Enum

class TrainingStrategies(Enum):

    AUTO = None
    DEFAULT = [
        {
            "convergence_criteria": "all_converged",
            "use_batching": False,
            "optim_algo": "irls_gd_tr",
        },
    ]
    IRLS = [
        {
            "convergence_criteria": "all_converged",
            "use_batching": False,
            "optim_algo": "irls_gd_tr",
        },
    ]
    IRLS_BATCHED = [
        {
            "convergence_criteria": "all_converged",
            "use_batching": True,
            "optim_algo": "irls_gd_tr",
        },
    ]
    ADAM_BATCHED = [
        {
            "convergence_criteria": "all_converged",
            "use_batching": True,
            "optim_algo": "adam",
        },
    ]
    ADAM = [
        {
            "convergence_criteria": "all_converged",
            "use_batching": False,
            "optim_algo": "adam",
        },
    ]
    IRLS_LS = [
        {
            "convergence_criteria": "all_converged",
            "use_batching": False,
            "optim_algo": "irls_ls_tr",
        },
    ]
