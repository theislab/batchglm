
# Fast and scalable fitting of over-determined generalized-linear models (GLMs)

batchglm was developed in the context of [diffxpy](https://github.com/theislab/diffxpy) to allow fast model fitting for differential expression analysis for single-cell RNA-seq data. However, one can use batchglm or its concepts in other scenarios where over-determined GLMs are encountered. batchglm is based on TensorFlow 

# Installation
1. Install [tensorflow](https://www.tensorflow.org/install/), see below. Please use the pip installation if you are unsure.
2. Clone the GitHub repository of batchglm.
3. cd into the clone.
4. pip install -e .

## Tensorflow installation
Tensorflow can be installed like any other package or can be compiled from source to allow for optimization of the software to the given hardware. Compiling tensorflow from source can significantly improve the performance, since this allows tensorflow to make use of all available CPU-specific instructions. Hardware optimization takes longer but is only required once during installation and is recommended if batchglm is used often or on large data sets. We summarize a few key steps here, an extensive up-to-date installation guide can be found here: https://www.tensorflow.org/install/

### Out-of-the-box tensorflow installation
You can install [tensorflow](https://www.tensorflow.org/install/) via pip or via conda.

#### pip
- CPU-only: <br/>
  `pip install tf-nightly`
- GPU: <br/>
  `pip install tf-nightly-gpu`
  
### Hardware-optimized tensorflow installation (compiling from source)
Please refer to https://www.tensorflow.org/install/ .

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
This does not work yet.
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
 
1. Make sure `sphinx`, `sphinx-autodoc-typehints` and `sphinx_rtd_theme` packages are installed (install via pip for example). 
2. `cd docs/`
3. `make html`
