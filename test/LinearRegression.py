import tensorflow as tf
import pandas as pd

from sklearn import datasets
from scipy import polyval, stats

import matplotlib.pyplot as plt

from impl.tf.LinearRegression import linear_regression


def plot_iris_fit(fit):
    axes = iris.plot(x="petal length (cm)", y="petal width (cm)", kind="scatter", color="red")

    for i in fit:
        line = polyval(fit[i][::-1], iris["petal length (cm)"])
        plt.plot(iris["petal length (cm)"], line, color='blue', linewidth=3, label=i)

    plt.show()


if __name__ == "__main__":
    data = datasets.load_iris()
    iris = pd.DataFrame(data.data, columns=data.feature_names)

    slope, intercept, _, _, _ = stats.linregress(iris["petal length (cm)"],
                                                 iris["petal width (cm)"])

    # tensorflow:
    sess = tf.Session()

    X = pd.DataFrame(iris["petal length (cm)"])
    X.insert(0, "intercept", 1)
    X = tf.constant(X)

    y = pd.DataFrame(iris["petal width (cm)"])
    y = tf.constant(y)

    b_trained, loss_trained = linear_regression(X, y, fast=False)

    # define train function
    train_op = tf.train.AdamOptimizer(learning_rate=0.05)
    train_op = train_op.minimize(loss_trained, global_step=tf.train.get_global_step())

    initializer_op = tf.global_variables_initializer()

    sess.run(initializer_op)
    for i in range(1000):
        b_trained_res, loss_trained_res, _ = sess.run((b_trained, loss_trained, train_op))

    b_fast, loss_fast = linear_regression(X, y)
    b_fast_res, loss_fast_res = sess.run((b_fast, loss_fast))

    fit = pd.DataFrame({"SciPy": [intercept, slope],
                        "TF_trained": b_trained_res[:, 0],
                        "TF_fast": b_fast_res[:, 0]})
    print(fit)

    plot_iris_fit(fit)
