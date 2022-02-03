import multiprocessing
import os

ACCURACY_MARGIN_RELATIVE_TO_LIMIT = float(os.environ.get('BATCHGLM_ACCURACY_MARGIN', 2.5))
FIM_MODE = str(os.environ.get('FIM_MODE', "analytic"))
HESSIAN_MODE = str(os.environ.get('HESSIAN_MODE', "analytic"))
JACOBIAN_MODE = str(os.environ.get('JACOBIAN_MODE', "analytic"))
CHOLESKY_LSTSQS = False
CHOLESKY_LSTSQS_BATCHED = False
EVAL_ON_BATCHED = False

# Trust region hyper parameters:
TRUST_REGION_RADIUS_INIT = 100.
TRUST_REGION_ETA0 = 0.
TRUST_REGION_ETA1 = 0.25
TRUST_REGION_ETA2 = 0.25
TRUST_REGION_T1 = 0.5  # Fast collapse to avoid trailing.
TRUST_REGION_T2 = 1.5  # Allow expansion if not shrinking.
TRUST_REGION_UPPER_BOUND = 1e5

TRUST_REGIONT_T1_IRLS_GD_TR_SCALE = 1

# Convergence hyper-parameters:
LLTOL_BY_FEATURE = 1e-10
XTOL_BY_FEATURE_LOC = 1e-8
XTOL_BY_FEATURE_SCALE = 1e-6
GTOL_BY_FEATURE_LOC = 1e-8
GTOL_BY_FEATURE_SCALE = 1e-8
