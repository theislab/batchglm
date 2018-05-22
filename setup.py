from setuptools import setup, find_packages

setup(
    name='rsa',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'pandas',
        'tables',  # pytables
        'patsy'  # design matrix generation
    ],
    extras_require={
        'plotting': [
            "matplotlib",
            "plotnine",
            "plotly"
        ]
    }

)
