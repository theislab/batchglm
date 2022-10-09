batchglm
===========================

|PyPI| |Python Version| |License| |Read the Docs| |Build| |Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/batchglm.svg
   :target: https://pypi.org/project/batchglm/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/batchglm
   :target: https://pypi.org/project/batchglm
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/theislab/batchglm
   :target: https://opensource.org/licenses/BSD
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/batchglm/latest.svg?label=Read%20the%20Docs
   :target: https://batchglm.readthedocs.io/
   :alt: Read the documentation at https://batchglm.readthedocs.io/
.. |Build| image:: https://github.com/theislab/batchglm/workflows/Build%20batchglm%20Package/badge.svg
   :target: https://github.com/theislab/batchglm/actions?workflow=Package
   :alt: Build Package Status
.. |Tests| image:: https://github.com/theislab/batchglm/workflows/Run%20batchglm%20Tests/badge.svg
   :target: https://github.com/theislab/batchglm/actions?workflow=Tests
   :alt: Run Tests Status
.. |Codecov| image:: https://codecov.io/gh/theislab/batchglm/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/theislab/batchglm
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black


Features
--------

- Fit many (i.e a batch!) of GLM's all at once using `numpy` (coming soon: `tensorflow2` or `statsmodels`) with a simple API
- Integrates with and provides utilities for working with familiar libraries like `patsy` and `dask`.

Installation
------------

You can install *batchglm* via pip_ from PyPI_:

.. code:: console

   $ pip install batchglm


Usage
-----

Please see the API documentation for details or the jupyter notebook tutorials (TODO: need notebooks - separate docs?)


Credits
-------

This package was created with cookietemple_ using Cookiecutter_ based on Hypermodern_Python_Cookiecutter_.

.. _cookietemple: https://cookietemple.com
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _PyPI: https://pypi.org/
.. _Hypermodern_Python_Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _pip: https://pip.pypa.io/
.. _Usage: https://batchglm.readthedocs.io/en/latest/usage.html
