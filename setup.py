from setuptools import setup, find_packages
import versioneer

author = 'Florian R. Hölzlwimmer, David S. Fischer'

setup(
    name='batchglm',
    author=author,
    author_email='florian.hoelzlwimmer@helmholtz-muenchen.de',
    packages=find_packages(),
    install_requires=[
        'tensorflow=1.10.0',
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
        'PyYAML',

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
            'yaml',
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
