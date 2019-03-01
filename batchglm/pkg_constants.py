import os
import multiprocessing

import tensorflow as tf

TF_NUM_THREADS = int(os.environ.get('TF_NUM_THREADS', 0))
TF_LOOP_PARALLEL_ITERATIONS = int(os.environ.get('TF_LOOP_PARALLEL_ITERATIONS', 10))

ACCURACY_MARGIN_RELATIVE_TO_LIMIT = float(os.environ.get('BATCHGLM_ACCURACY_MARGIN', 2.5))
FIM_MODE = str(os.environ.get('FIM_MODE', "analytic"))
HESSIAN_MODE = str(os.environ.get('HESSIAN_MODE', "analytic"))
JACOBIAN_MODE = str(os.environ.get('JACOBIAN_MODE', "analytic"))
CHOLESKY_LSTSQS = True
CHOLESKY_LSTSQS_BATCHED = False

XARRAY_NETCDF_ENGINE = "h5netcdf"

TF_CONFIG_PROTO = tf.ConfigProto()
TF_CONFIG_PROTO.allow_soft_placement = True
TF_CONFIG_PROTO.log_device_placement = False
TF_CONFIG_PROTO.gpu_options.allow_growth = True

TF_CONFIG_PROTO.inter_op_parallelism_threads = 0 if TF_NUM_THREADS == 0 else 1
TF_CONFIG_PROTO.intra_op_parallelism_threads = TF_NUM_THREADS

if TF_NUM_THREADS == 0:
    TF_NUM_THREADS = multiprocessing.cpu_count()

# Trust region hyper parameters:
TRUST_REGION_RADIUS_INIT = 4.
TRUST_REGION_ETA0 = 0.
TRUST_REGION_ETA1 = 0.25
TRUST_REGION_ETA2 = 0.25  # Allow expansion if not shrinking.
TRUST_REGION_T1 = 0.1  # Fast collapse to avoid trailing.
TRUST_REGION_T2 = 2.  # Very conservative expansion to run updates once valid region is reached.
TRUST_REGION_UPPER_BOUND = 1e2  # Low upper limit so that collapse to valid region does not cause feature to trail.

# Convergence hyper-parameters:
XTOL_LL_BY_FEATURE_LOC = 1e-8
XTOL_LL_BY_FEATURE_SCALE = 1e-4
GTOL_LL_BY_FEATURE_LOC = 1e-8
GTOL_LL_BY_FEATURE_SCALE = 1e-8
