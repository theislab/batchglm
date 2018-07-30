import os

import tensorflow as tf

TF_NUM_THREADS = int(os.environ.get('TF_NUM_THREADS', 1))
TF_LOOP_PARALLEL_ITERATIONS = int(os.environ.get('TF_NUM_THREADS', 10))

XARRAY_NETCDF_ENGINE = "h5netcdf"

TF_CONFIG_PROTO = tf.ConfigProto()
TF_CONFIG_PROTO.allow_soft_placement = True
TF_CONFIG_PROTO.log_device_placement = False
TF_CONFIG_PROTO.gpu_options.allow_growth = True

TF_CONFIG_PROTO.inter_op_parallelism_threads = 0 if TF_NUM_THREADS == 0 else 1
TF_CONFIG_PROTO.intra_op_parallelism_threads = TF_NUM_THREADS
