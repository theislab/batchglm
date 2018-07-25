
Fitting models with stochastic gradient descent

Tensorflow optimization for recent MacBooks:
from http://www.andrewclegg.org/tech/TensorFlowLaptopCPU.html
1. Get newest TensorFlow repo from github vi git clone.
2. cd into tensorflow/
3. ./configure
4. brew install bazel
5. bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
6. bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
7. pip install /tmp/tensorflow_pkg/tensorflow-<blah>.whl
