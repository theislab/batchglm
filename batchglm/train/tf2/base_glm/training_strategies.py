from enum import Enum


class TrainingStrategies(Enum):

    AUTO = None

    DEFAULT = \
        {
            "optim_algo": "default_adam",
            "jacobian": True,
            "hessian": False,
            "fim": False,
            "concat_grads": True
        }

    GD = \
        {
            "optim_algo": "gd",
            "jacobian": True,
            "hessian": False,
            "fim": False,
            "concat_grads": True
        }

    ADAM =  \
        {
            "optim_algo": "adam",
            "jacobian": True,
            "hessian": False,
            "fim": False,
            "concat_grads": True
        }

    ADAGRAD = \
        {
            "optim_algo": "adagrad",
            "jacobian": True,
            "hessian": False,
            "fim": False,
            "concat_grads": True
        }

    RMSPROP = \
        {
            "optim_algo": "rmsprop",
            "jacobian": True,
            "hessian": False,
            "fim": False,
            "concat_grads": True
        }

    IRLS = \
        {
            "optim_algo": "irls",
            "jacobian": False,
            "hessian": False,
            "fim": True,
            "concat_grads": False,
            "calc_separated": True
        }

    IRLS_TR = \
        {
            "optim_algo": "irls_tr",
            "jacobian": False,
            "hessian": False,
            "fim": True,
            "concat_grads": False,
            "calc_separated": True
        }

    IRLS_GD = \
        {
            "optim_algo": "irls_gd",
            "jacobian": False,
            "hessian": False,
            "fim": True,
            "concat_grads": False,
            "calc_separated": True
        }

    IRLS_GD_TR = \
        {
            "optim_algo": "irls_gd_tr",
            "jacobian": False,
            "hessian": False,
            "fim": True,
            "concat_grads": False,
            "calc_separated": True
        }

    NR = \
        {
            "optim_algo": "nr",
            "jacobian": False,
            "hessian": True,
            "fim": False,
            "concat_grads": True,
            "calc_separated": False
        }

    NR_TR = \
        {
            "optim_algo": "nr_tr",
            "jacobian": False,
            "hessian": True,
            "fim": False,
            "concat_grads": True,
            "calc_separated": False
        }
