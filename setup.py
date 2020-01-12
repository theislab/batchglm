from setuptools import setup, find_packages
import versioneer

author = 'David S. Fischer, Florian R. Hölzlwimmer, Sabrina Richter'
author_email='david.fischer@helmholtz-muenchen.de'
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
        'anndata',
        'numpy>=1.16.4',
        'scipy>=1.2.1',
        'pandas',
        'dask',
        'toolz',
        'patsy',
        'sparse',
        'dask'
    ],
    extras_require={
        'optional': [
            'tensorflow>=1.14.0',
            'tensorflow-gpu>=1.14.0'
        ],
        'plotting_deps': [
             "matplotlib",
             "seaborn"
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
