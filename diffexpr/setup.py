from setuptools import setup, find_packages

setup(
    name='diffexpr',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'patsy',
    ],
    extras_require={
        # 'scanpy_deps': [
        #     "scanpy",
        #     "anndata"
        # ],
        # 'plotting_deps': [
        #     "plotnine",
        #     "matplotlib"
        # ]
    }

)
