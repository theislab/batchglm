import os

import matplotlib.pyplot as plt

from mlxtend.plotting import ecdf

import numpy as np
import scipy.sparse
import xarray as xr
import pandas as pd

from api.models.nb_glm import Estimator, InputData

import data as data_utils
import anndata
import scanpy.api as sc

WORKING_DIR = "data/real_data/"
MTX_FILE = "~/Masterarbeit/data/sample_data.zip"

file = MTX_FILE


def load_data_from_mtx(path) -> anndata.AnnData:
    """
    Combines MTX files from a given path.

    WARNING:
        Use only, if all MTX files' first dimensions (i.e. the variables) are equal!

        This implementation does not join any variables!

    :param path: Path to zip-file or folder containing MTX-files
    :return: anndata.AnnData object
    """
    adatas = data_utils.load_recursive_mtx(path, target_format="adata")
    adatas_list = list(adatas.values())

    adata = anndata.AnnData(scipy.sparse.vstack([x.X for x in adatas.values()]))
    # adata.var_names = adatas_list[0].var_names
    adata.var = adatas_list[0].var
    # adata.obs_names = np.concatenate([x.obs_names for x in adatas.values()], axis=0)

    for path, array in adatas.items():
        name = os.path.basename(path)
        (_, _, condition, batch) = name.split("_")

        array.obs["batch"] = batch
        array.obs["condition"] = condition
        array.obs["source"] = name

    adata.obs = pd.concat([x.obs for x in adatas.values()], axis=0)

    return adata


def log_ecdf(data: xr.DataArray):
    ax, _, _ = ecdf(data)
    ax.set_xscale("log")
    plt.axvline(x=10)
    plt.show()


def load_and_preprocess_data() -> anndata.AnnData:
    file = os.path.join(WORKING_DIR, "data.h5")
    if os.path.exists(file):
        # dataset = xr.open_dataset(file, engine=pkg_constants.XARRAY_NETCDF_ENGINE)
        dataset = anndata.read_h5ad(file)
    else:
        dataset = load_data_from_mtx(MTX_FILE)
        # dataset.to_netcdf(file, engine=pkg_constants.XARRAY_NETCDF_ENGINE,
        #                   encoding={"sample_data": {"compression": "gzip"}})
        dataset.write(file)
    # # define unique sample and gene labels
    # sample_data["samples"] = np.arange(sample_data.shape[0])
    # sample_data["genes"] = np.arange(sample_data.shape[1])

    # filter genes and cells
    sc.pp.filter_cells(dataset, min_genes=200)
    sc.pp.filter_genes(dataset, min_cells=3)

    return dataset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs=1, help='folder for sample data')
    args, unknown = parser.parse_known_args()

    data_folder = args.data
    if data_folder is None:
        data_folder = "data/"

    dataset = load_and_preprocess_data()
    data_utils.design_matrix_from_anndata(dataset, formula="~ 1 + batch + condition", append=True)

    input_data = InputData(dataset)
    estimator = Estimator(input_data, batch_size=500)
    estimator.initialize()
    estimator.train(learning_rate=0.5)
