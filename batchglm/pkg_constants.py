import os
import multiprocessing

import tensorflow as tf

TF_NUM_THREADS = int(os.environ.get('TF_NUM_THREADS', 0))
TF_LOOP_PARALLEL_ITERATIONS = int(os.environ.get('TF_LOOP_PARALLEL_ITERATIONS', 10))

ACCURACY_MARGIN_RELATIVE_TO_LIMIT = float(os.environ.get('BATCHGLM_ACCURACY_MARGIN', 2.5))
HESSIAN_MODE = str(os.environ.get('HESSIAN_MODE', "obs_batched"))
JACOBIAN_MODE = str(os.environ.get('JACOBIAN_MODE', "analytic"))
CHOLESKY_LSTSQS = True

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
TRUST_REGION_RADIUS_INIT = 10
TRUST_REGION_ETA0 = 0
TRUST_REGION_ETA1 = 0.25
TRUST_REGION_ETA2 = 0.75
TRUST_REGION_T1 = 2./3.
TRUST_REGION_T2 = 3./2.
TRUST_REGION_UPPER_BOUND = 100

# Convergence hyperparameters:
THETA_MIN_LL_BY_FEATURE = 1e-6
