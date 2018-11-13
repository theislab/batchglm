
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
  `pip install tensorflow`
- GPU: <br/>
  `pip install tensorflow-gpu`
  
### Hardware-optimized tensorflow installation (compiling from source)
Please refer to https://www.tensorflow.org/install/.
