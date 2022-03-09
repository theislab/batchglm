.. automodule:: batchglm

API
===


Import batchglm's high-level API as::

   import batchglm.api as glm


Fitting models
-----------------------------------

All models are collected in the :mod:`train` and `model` module.
Each model consists of at least:

1) `models.glm_nb.Model` class which basicially describes the model
3) `train.xxxxx.Estimator` class which takes a `Model` object and fits the corresponding model onto it.

where `xxxxxx` is the backend desired, like `tf2`, `numpy` or `statsmodel`.

For example, here is a short snippet to give a sense of how the API might work::

   from batchglm.models.glm_nb import Model as NBModel
   from batchglm.train.numpy.glm_nb import Estimator as NBEstimator  
   from batchglm.utils.input import InputDataGLM

   input_data = InputDataGLM(data=data_matrix, design_loc=_design_loc, design_scale=_design_scale, as_dask=as_dask)
   model = NBModel(input_data=input_data)
   estimator = NBEstimator(model=model, init_location="standard", init_scale="standard")
   estimator.initialize()
   estimator.train_sequence(training_strategy="DEFAULT")
   # Now you can perform statistical tests, for example, on parameters like model.theta_location.

Currently implemented models:

Negative Binomial
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   models.glm_nb.Model
   train.numpy.glm_nb.Estimator

Planned or Incomplete Models:

Beta
~~~~~~~~~~~~~~~~~

Normal
~~~~~~~~~~~~~~~~~

Poisson
~~~~~~~~~~~~~~~~~

Data Utilities
-----------------------------------
We also provide some data utilities for working with things like design and constraint matrices.

.. autosummary::
   :toctree: .

   utils.data.bin_continuous_covariate
   utils.data.constraint_matrix_from_string
   utils.data.constraint_system_from_star
   utils.data.design_matrix
   utils.data.preview_coef_names
   utils.data.string_constraints_from_dict
   utils.data.view_coef_names
   utils.input.InputDataGLM