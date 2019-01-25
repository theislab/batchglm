import scipy.sparse as sp
import numpy as np
import tensorflow as tf
print(tf.__version__)

NOBS = 50
SPARSE = True
CUSTOM = True

def map_reduce(last_elem, data: tf.data.Dataset, map_fn, reduce_fn=tf.add, **kwargs):
    iterator = data.make_initializable_iterator()

    def cond(idx, val):
        return tf.not_equal(tf.gather(idx, tf.size(idx) - 1), last_elem)

    def body_fn(old_idx, old_val):
        idx, val = iterator.get_next()
        print(reduce_fn(old_val, map_fn(idx, val)).shape)

        return idx, reduce_fn(old_val, map_fn(idx, val))

    def init_vals():
        idx, val = iterator.get_next()
        return idx, map_fn(idx, val)

    with tf.control_dependencies([iterator.initializer]):
        _, reduced = tf.while_loop(cond, body_fn, init_vals(), **kwargs)

    print("flag")
    return reduced


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
        x_shape_0 = np.asarray(x.shape[0], np.int64)
        return (x_val, x_idx, x_shape_0)

    def fetch_dense(self, idx):
        x = np.asarray(self.a_dense[idx], np.float32)
        return x


class Model_sparse:
    def __init__(self, a):
        idx, (x_val, x_idx, X_shape) = a

        a = tf.SparseTensor(
            indices=tf.cast(x_idx, dtype=tf.int64),
            values=tf.cast(x_val, dtype=tf.float32),
            dense_shape=X_shape
        )
        pred = tf.sparse.reduce_sum(a, keepdims=True)
        pred = tf.reshape(pred, shape=[1])
        print(pred.shape)
        self.pred = pred


class Model_dense:
    def __init__(self, a):
        x = a
        pred = tf.reduce_sum(x)
        self.pred = pred


class TF:
    def __init__(self, input: Input):
        def fetch_tf_sparse(idx):
            x_val, x_idx, x_shape_0 = tf.py_function(
                func=input.fetch_sparse,
                inp=[idx],
                Tout=[tf.float32, tf.int64, tf.int64]
            )
            x_shape_0.set_shape([1])
            x_idx.set_shape(idx.get_shape().as_list() + [2])
            x_val.set_shape(idx.get_shape().as_list())
            # Need this to properly propagate the size of the first dimension:
            if idx.get_shape().as_list()[0] is None:
                X_shape = tf.concat([x_shape_0, tf.constant([input.n_features], dtype=tf.int64)], axis=0)
            else:
                X_shape = tf.constant([idx.get_shape().as_list()[0], input.n_features], dtype=tf.int64)

            return idx, (x_val, x_idx, X_shape)

        def fetch_tf_dense(idx):
            x = tf.py_func(
                func=input.fetch_dense,
                inp=[idx],
                Tout=tf.float32,
                stateful=False
            )
            x.set_shape([idx.get_shape().as_list()[0], input.n_features])
            return x

        indices = tf.range(NOBS)
        dataset = tf.data.Dataset.from_tensor_slices((indices))
        batched_data = dataset.batch(5)
        if SPARSE:
            batched_data = batched_data.map(fetch_tf_sparse, 1)
        else:
            batched_data = batched_data.map(fetch_tf_dense, 1)
        batched_data = batched_data.prefetch(1)

        if SPARSE:
            if CUSTOM:
                def map_model(idx, data):
                    return Model_sparse((idx, data)).pred

                self.llb = map_reduce(
                    last_elem=tf.gather(indices, tf.size(indices) - 1),
                    data=batched_data,
                    map_fn=lambda idx, data: map_model(idx, data),
                    reduce_fn=tf.add,
                    parallel_iterations=1,
                )
            else:
                self.llb = batched_data.reduce(
                    initial_state=tf.zeros([1], dtype=np.float32),
                    reduce_func=lambda old, new: tf.add(Model_sparse(new).pred, old)
                )
        else:
            self.llb = batched_data.reduce(
                initial_state=tf.zeros([1], dtype=np.float32),
                reduce_func=lambda old, new: tf.add(Model_dense(new).pred, old)
            )

input = Input()
tf_example = TF(input)
ll1 = tf_example.llb

with tf.Session() as sess:
    print("starting...")
    print(sess.run(ll1))