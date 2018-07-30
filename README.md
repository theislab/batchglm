
# Fitting models with stochastic gradient descent

## Installing Tensorflow:
### Via pip:
- CPU-only: <br/>
  `pip install tensorflow`
- GPU: <br/>
  `pip install tensorflow-gpu`

### Compiling from source: 
Compiling tensorflow from source can significantly improve the performance,
since this allows tensorflow to make use of all available CPU-
specific instructions.

#### Pre-requirements
First, you have to install bazel (a build tool).
- On MacBook:<br/>
  `brew install bazel`
- On linux:
  * Use Anaconda/Miniconda:<br/>
    `conda install bazel`
  * Official / Distribution-specific ways: 
    https://docs.bazel.build/versions/master/install.html
  
#### Compilation
1. Get newest TensorFlow repo from github via git clone:<br/>
    `git clone https://github.com/tensorflow/tensorflow.git`
2. `cd tensorflow/`
3. `git checkout <release>`
4. `./configure`
5. `bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`
6. `bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`
7. `pip install /tmp/tensorflow_pkg/tensorflow-<blah>.whl`

## Building the documentation
The documentation is maintained in the `docs/` directory.

The built documentation will be saved in `build/docs`. 
 
- `cd docs/`
- `make html`

