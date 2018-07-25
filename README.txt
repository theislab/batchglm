
Fitting models with stochastic gradient descent

Tensorflow optimization for recent MacBooks:
1. Get newest TensorFlow repo from github vi git clone.
2. cd into tensorflow/
3. ./configure
4. brew install bazel
5. bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 -k \
  //tensorflow/tools/pip_package:build_pip_package
