import tensorflow as tf


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
            :param l2_reg: \lambda regularization
            :param fast: use closed-form solution to calculate 'b'
            :return:    | tuple(b, least_squares)
                        | b is a Tensor of shape ([...], N, K)
            """
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
    def estimated_params(self):
        """
        alias for `b`
        
        :return: self.b
        """
        return self.b
    
    def rmsd(self, true_b: tf.Tensor):
        """
        Calculate the root of the mean squared deviation between the estimated weights `b` and the true `b`
        
        :param true_b: Tensor representing the true weights `b`
        :return: \sqrt{mean{(b_{estim} - b_{true})^2}}
        """
        with tf.name_scope("RMSD"):
            rmsd = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.b, true_b)))
            rmsd = tf.identity(rmsd, name="RMSD")
            return rmsd
    
    def mae(self, true_b: tf.Tensor):
        """
        Calculate the mean absolute error between the estimated weights `b` and the true `b`
        
        :param true_b: Tensor representing the true weights `b`
        :return: mean{(b_{estim} - b_{true})}
        """
        with tf.name_scope("RMSD"):
            mae = tf.reduce_mean(self.b - true_b)
            mae = tf.identity(mae, name="MAE")
            return mae
    
    def normalize(self, measure: tf.Tensor, true_b: tf.Tensor):
        """
        Normalize measure (e.g. `RMSD` or `MAE`) with the range of `true_b`
        
        :param true_b: Tensor representing the true weights `b`
        :return: \frac{RMSD}{max(b_{true}) - min(b_{true})}
        """
        norm = measure / (tf.maximum(true_b) - tf.minimum(true_b))
        norm = tf.identity(norm, name="normalize")
        return norm
    
    def normalized_rmsd(self, true_b: tf.Tensor):
        """
        Calculate the normalized RMSD between the estimated weights `b` and the true `b`
        
        :param true_b: Tensor representing the true weights `b`
        :return: \frac{RMSD}{max(b_{true}) - min(b_{true})}
        """
        return tf.identity(self.normalize(self.rmsd(true_b)), name="NRMSD")
    
    def normalized_mae(self, true_b: tf.Tensor):
        """
        Calculate the normalized MAE between the estimated weights `b` and the true `b`
        
        :param true_b: Tensor representing the true weights `b`
        :return: \frac{MAE}{max(b_{true}) - min(b_{true})}
        """
        return tf.identity(self.normalize(self.mae(true_b)), name="NMAE")
