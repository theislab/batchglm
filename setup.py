from setuptools import setup, find_packages

author = 'Florian R. Hölzlwimmer, David S. Fischer'

setup(
    name='sgdglm',
    author=author,
    author_email='florian.hoelzlwimmer@helmholtz-muenchen.de',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
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
        'benchmarks': [
            'yaml',
            "plotnine",
            "matplotlib",
        ],
        'examples': [
            'sklearn',
            'mlxtend',
        ]
    }

)
