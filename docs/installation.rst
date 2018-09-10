Installation
============

We assume that you have a python environment set up.

Firstly, you need to install tensorflow and tensorflow-probability.
You can install these dependencies from source to optimze them to your hardware which can improve performance.
Note that both packages also have GPU versions which allows you to run the run-time limiting steps of batchglm on GPUs.
The simplest installation of these dependencies is via pip: call::

    pip install tf-nightly
    pip install tfp-nightly

The nightly versions of tensorflow and tensorflow-probability are up-to-date versions of these packages.
Alternatively you can also install the major releases: call::

    pip install tensorflow
    pip install tensorflow-probability


You can then install batchglm from source by using the repository on `GitHub
<https://github.com/theislab/batchglm>`__: 
``pip3 install git+https://github.com/theislab/batchglm.git``

You can now use batchglm in a python session by via the following import: call::

    import batchglm.api as glm
