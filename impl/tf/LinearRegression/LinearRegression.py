import tensorflow as tf


def linear_regression(X: tf.Tensor, y: tf.Tensor, weight_matrix=None, l2_reg=0.0, fast=True):
    """
    This method performs one or more linear regression problems: t(X) * b = y

    :param X: Tensor of shape ([...], M, N)
    :param y: Tensor of shape ([...], M, K)
    :param weight_matrix:   | if specified, the least-squares solution will be weighted by this matrix:
                            | t(y - Xb) * weight_matrix * (y - Xb)
    :param l2_reg: \lambda regularization
    :param fast: use closed-form solution to calculate 'b'
    :return:    | tuple(b, least_squares)
                | b is a Tensor of shape ([...], N, K)
    """
    # lambda_I = tf.tile(l2_reg, (tf.shape(X)[-2], tf.shape(X)[-2]))

    b = None
    if fast:
        Xt = tf.transpose(X, name="Xt")
        if weight_matrix is not None:
            Xt = tf.matmul(Xt, weight_matrix, name="XtM")

        b = tf.matmul(tf.matrix_inverse(Xt @ X - l2_reg), Xt @ y, name="weight")
    else:
        b_shape = X.get_shape().as_list()[0:-2] + [X.get_shape().as_list()[-1], y.get_shape().as_list()[-1]]
        b = tf.Variable(tf.random_normal(b_shape, dtype=X.dtype), name='weight')

    diff = y - X @ b
    squared_diff = tf.square(diff, name="squared_diff")
    if weight_matrix is not None:
        squared_diff = tf.matmul(squared_diff, weight_matrix, name="weighted_squared_diff")

    loss = tf.add(tf.reduce_sum(squared_diff) / 2, (l2_reg / 2) * tf.square(tf.norm(b)), name="loss")

    return b, loss
