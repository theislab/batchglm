import scipy.sparse as sp
import numpy as np
import tensorflow as tf
print(tf.__version__)

# version 2 works!!!

NOBS = 50

class Input:
    def __init__(self):
        self.n_features = 7
        self.a = np.ones([NOBS, self.n_features])
        self.a_sparse = sp.csr_matrix(self.a)
        self.a_dense = self.a

    def fetch_sparse(self, idx):
        x = self.a_sparse[idx]
        x_val = np.asarray(x.data, np.float32)
        x_idx = np.asarray(np.vstack(x.nonzero()).T, np.int64)
        x_shape = np.asarray(x.shape, np.int64)
        return (x_idx, x_val, x_shape)

    def fetch_dense(self, idx):
        x = np.asarray(self.a_dense[idx,:], np.float32)
        return x

class TF1:
    """
    Using a generator, here the indexing is unclear as the
    sparse tensor is likely consumed by ENTRY and not by observation vector,
    this does work however.
    """
    def __init__(self, input: Input):
        x = input.a_sparse
        shape = np.asarray(x.shape, np.int64)
        def generator():
            indices = np.asarray(np.vstack(x.nonzero()).T, np.int64)
            values = np.asarray(x.data, np.float32)
            yield (indices, values)

        batched_data = tf.data.Dataset.from_generator(generator, (tf.int64, tf.float32))
        batched_data = batched_data.map(lambda i, v: tf.SparseTensor(i, v, shape))
        batched_data = batched_data.batch(5)
        batched_data = batched_data.prefetch(1)

        self.llb = batched_data.reduce(
            initial_state=tf.zeros([1], dtype=np.float32),
            reduce_func=lambda old, new: tf.add(self.eval(new), old)
        )

    def eval(self, a):
        x = a
        pred = tf.sparse.reduce_sum(x, keepdims=False)
        return pred



class TF2:
    """
    Using py_func which yields a tensor representation of SparseTensor which is
    then mapped to a SparseTensor.
    """
    def __init__(self, input: Input):
        def fetch_dense(idx):
            x = tf.py_function(
                func=input.fetch_dense,
                inp=[idx],
                Tout=tf.float32
            )
            x.set_shape(idx.get_shape().as_list() + [input.n_features])
            x = (x,)
            constant = tf.constant(4., shape=())
            return idx, (x, constant)

        def fetch_sparse(idx):
            x_idx, x_val, x_shape = tf.py_function(
                func=input.fetch_sparse,
                inp=[idx],
                Tout=[tf.int64, tf.float32, tf.int64]
            )
            x = (x_idx, x_val, x_shape)
            constant = tf.constant(4., shape=())
            return idx, (x, constant)

        indices = tf.range(NOBS)
        batched_data = tf.data.Dataset.from_tensor_slices((indices))
        batched_data = batched_data.batch(5)
        batched_data = batched_data.map(fetch_dense, num_parallel_calls=1)

        def map_sparse(idx, data):
            x_ls, const = data
            if len(x_ls) > 1:
                x = tf.SparseTensor(x_ls[0], x_ls[1], x_ls[2])
            else:
                x = x_ls[0]
            return idx, (x, const)

        batched_data = batched_data.map(map_sparse, num_parallel_calls=1)
        batched_data = batched_data.prefetch(1)

        self.llb = batched_data.reduce(
            initial_state=tf.zeros([1], dtype=np.float32),
            reduce_func=lambda old, new: tf.add(self.eval(new[0], new[1]), old)
        )

    def eval(self, i, a):
        x, c = a
        if isinstance(x, tf.SparseTensor):
            pred = tf.sparse.reduce_sum(x, keepdims=False)
        else:
            pred = tf.reduce_sum(x, keepdims=False)
        return pred


class TF3:
    """
    Via a sliced SparseTensor. Does not work, something dodgy is going on
    with the dimensions.
    """
    def __init__(self, input: Input):
        x = input.a_sparse
        a = tf.SparseTensor(
            indices=tf.cast(np.asarray(np.vstack(x.nonzero()).T, np.int64), tf.int64),
            values=tf.cast(np.asarray(x.data, np.float32), tf.float32),
            dense_shape=tf.cast(np.asarray(x.shape, np.int64), tf.int64)
        )
        batched_data = tf.data.Dataset.from_sparse_tensor_slices((a))
        batched_data = batched_data.batch(5)
        batched_data = batched_data.prefetch(1)

        self.llb = batched_data.reduce(
            initial_state=tf.zeros([1], dtype=np.float32),
            reduce_func=lambda old, new: tf.add(self.eval(new), old)
        )

    def eval(self, a):
        i, v, s = a
        x = tf.SparseTensorValue(
            indices=i[:,:,0],
            values=v[:,0],
            dense_shape=s[:,0]
        )
        pred = tf.sparse.reduce_sum(x, keepdims=False)
        return pred

input = Input()
tf_example = TF2(input)
ll1 = tf_example.llb

with tf.Session() as sess:
    print("starting...")
    print(sess.run(ll1))