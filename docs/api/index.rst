.. automodule:: batchglm.api

API
===


Import batchglm's high-level API as::

   import batchglm.api as glm


Fitting models
-----------------------------------

All models are collected in the :mod:`train` and `model` module.
Each model consists of at least:

1) `models.glm_nb.Model` class which basicially describes the model
2) `trian.xxxxx.InputData` class which collects the data, design matrices, etc. in a single object
3) `train.xxxxx.Estimator` class which takes a `Model` object and fits the corresponding model onto it.

where `xxxxxx` is the backend desired, like `tf2`, `numpy` or `statsmodel`.

Currently implemented models:

Negative Binomial
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   models.glm_nb.Model
   train.numpy.glm_nb.Estimator

