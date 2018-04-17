import tensorflow as tf

import impl.tf.stats as stats


class LinearRegression:
    X: tf.Tensor
    y: tf.Tensor
    weight_matrix: tf.Tensor
    l2_reg: tf.Tensor
    b: tf.Tensor
    squared_error: tf.Tensor
    fast: bool

    def __init__(self, X: tf.Tensor, y: tf.Tensor, weight_matrix=None, l2_reg=0.0, fast=True, name="linear_regression"):
        """
            This class solves one or more linear regression problems: t(X) * b = y

            :param X: Tensor of shape ([...], M, N)
            :param y: Tensor of shape ([...], M, K)
            :param weight_matrix:   | if specified, the least-squares solution will be weighted by this matrix:
                                    | t(y - Xb) * weight_matrix * (y - Xb)
            :param l2_reg: \lambda - regularization
            :param fast: use closed-form solution to calculate 'b'
            :return:    | tuple(b, least_squares)
                        | b is a Tensor of shape ([...], N, K)
            """
        l2_reg = tf.convert_to_tensor(l2_reg, dtype=X.dtype, name="l2_reg")
        # lambda_I = tf.tile(l2_reg, (tf.shape(X)[-2], tf.shape(X)[-2]))

        with tf.name_scope(name):
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

        self.X = X
        self.y = y
        self.weight_matrix = weight_matrix
        self.l2_reg = l2_reg
        self.b = b
        self.squared_error = loss
        self.fast = fast

    @property
    def estimated_params(self) -> tf.Tensor:
        """
        alias for `b`
        
        :return: self.b
        """
        return self.b

    def rmsd(self, b_obs: tf.Tensor, name="RMSD") -> tf.Tensor:
        """
        Calculate the root of the mean squared deviation between the estimated weights `b` and the observed `b`
        
        :param b_obs: Tensor representing the observed weights `b`
        :return: \sqrt{mean{(b_{estim} - b_{obs})^2}}
        """
        return stats.rmsd(self.b, b_obs, name=name)

    def mae(self, b_obs: tf.Tensor, name="MAE") -> tf.Tensor:
        """
        Calculate the mean absolute error between the estimated weights `b` and the observed `b`
        
        :param b_obs: Tensor representing the observed weights `b`
        :return: mean{(b_{estim} - b_{obs})}
        """
        return stats.mae(self.b, b_obs, name=name)

    def normalized_rmsd(self, b_obs: tf.Tensor, name="NRMSD") -> tf.Tensor:
        """
        Calculate the normalized RMSD between the estimated weights `b` and the observed `b`
        
        :param b_obs: Tensor representing the observed weights `b`
        :return: \frac{RMSD}{max(b_{obs}) - min(b_{obs})}
        """
        return stats.normalized_rmsd(self.b, b_obs, name=name)

    def normalized_mae(self, b_obs: tf.Tensor, name="NMAE") -> tf.Tensor:
        """
        Calculate the normalized MAE between the estimated weights `b` and the observed `b`
        
        :param b_obs: Tensor representing the observed weights `b`
        :return: \frac{MAE}{max(b_{obs}) - min(b_{obs})}
        """
        return stats.normalized_mae(self.b, b_obs, name=name)
