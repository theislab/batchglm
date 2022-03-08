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

For example, here is a short snippet to give a sense of how the API might work

```python
input_data = InputDataGLM(data=data_matrix, design_loc=_design_loc, design_scale=_design_scale, as_dask=as_dask)
model = NBModel(input_data=input_data)
estimator = NBEstimator(model=model, init_location="standard", init_scale="standard")
estimator.initialize()
estimator.train_sequence(training_strategy="DEFAULT")
# Now you can perform statistical tests, for example, on the model's parameters.
```

Currently implemented models:

Negative Binomial
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   models.glm_nb.Model
   train.numpy.glm_nb.Estimator

Planned or Incomplete (missing an `Estimator`) Models

Beta
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   models.glm_beta.Model

Normal
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   models.glm_norm.Model

Poisson
~~~~~~~~~~~~~~~~~