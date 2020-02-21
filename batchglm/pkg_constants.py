import os
import multiprocessing

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

try:
    import tensorflow as tf

    TF_NUM_THREADS = int(os.environ.get('TF_NUM_THREADS', 0))
    TF_LOOP_PARALLEL_ITERATIONS = int(os.environ.get('TF_LOOP_PARALLEL_ITERATIONS', 10))

    TF_CONFIG_PROTO = tf.compat.v1.ConfigProto()
    TF_CONFIG_PROTO.allow_soft_placement = True
    TF_CONFIG_PROTO.log_device_placement = False
    TF_CONFIG_PROTO.gpu_options.allow_growth = True
    TF_CONFIG_PROTO.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

    TF_CONFIG_PROTO.inter_op_parallelism_threads = TF_NUM_THREADS
    TF_CONFIG_PROTO.intra_op_parallelism_threads = TF_NUM_THREADS

    if TF_NUM_THREADS == 0:
        TF_NUM_THREADS = multiprocessing.cpu_count()

except ImportError:
    tf = None
