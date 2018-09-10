.. automodule:: batchglm.api

API
===


Import batchglm's high-level API as::

   import batchglm.api as glm

Preprocessing
-----------------------------------

:mod:`batchglm.api.data` simplifies loading of mtx files and generating design matrices.

.. For visual quality control, see :func:`~scanpy.api.pl.highest_expr_gens` and
.. :func:`~scanpy.api.pl.filter_genes_dispersion` in the :doc:`plotting API <plotting>`.

.. autosummary::
   :toctree: .

   data.design_matrix
   data.design_matrix_from_xarray
   data.design_matrix_from_anndata
   data.sample_description_from_xarray
   data.sample_description_from_anndata
   data.load_mtx_to_adata
   data.load_mtx_to_xarray
   data.load_recursive_mtx
   data.xarray_from_data


Fitting models
-----------------------------------

All models are collected in the :mod:`models` module.
Each model consists of at least:

1) `Model` class which basicially describes the model
#) `InputData` class which collects the data, design matrices, etc. in a single object
#) `Simulator` class which allows to simulate data of the corresponding model
#) `Estimator` class which takes an `InputData` object and fits the corresponding model onto it.

Currently implemented models:

Negative Binomial
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   models.nb.Model
   models.nb.InputData
   models.nb.Simulator
   models.nb.Estimator

General Linearized Model with negative binomial noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   models.nb_glm.Model
   models.nb_glm.InputData
   models.nb_glm.Simulator
   models.nb_glm.Estimator

