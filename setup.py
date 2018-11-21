from setuptools import setup, find_packages
import versioneer

author = 'Florian R. Hölzlwimmer, David S. Fischer'
author_email='batchglm@frhoelzlwimmer.de, david.fischer@helmholtz-muenchen.de'
description="Fast and scalable fitting of over-determined generalized-linear models (GLMs)"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='batchglm',
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=1.10.0',
        'tensorflow-probability',
        'numpy>=1.14.0',
        'scipy',
        'pandas',
        'xarray',
        'dask',
        # HDF5 / NetCDF support
        'h5netcdf',
        'netcdf4',  # currently needed for 'xr.open_mfdataset'
        'toolz',
        # design matrix generation
        'patsy',
    ],
    extras_require={
        'scanpy_deps': [
            "scanpy",
            "anndata"
        ],
        # 'plotting_deps': [
        #     "matplotlib",
        #     "plotnine",
        #     "seaborn"
        # ],
        'tensorflow_gpu': [
            "tensorflow-gpu",
            "tensorflow-probability-gpu",
        ],
        'benchmarks': [
            'PyYAML',
            "plotnine",
            "matplotlib",
        ],
        'tutorials': [
            'sklearn',
            'mlxtend',
        ],
        'docs': [
            'sphinx',
            'sphinx-autodoc-typehints',
            'sphinx_rtd_theme',
            'jinja2',
            'docutils',
        ],
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
