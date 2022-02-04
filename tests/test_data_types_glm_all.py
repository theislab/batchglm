import anndata

try:
    from anndata.base import Raw
except ImportError:
    from anndata import Raw

import logging
import unittest

import numpy as np
import scipy.sparse

import batchglm.api as glm
from batchglm.models.base_glm import InputDataGLM
from tests.test_graph_glm_all import _TestGraphGlmAll

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _TestDataTypesGlmAll(_TestGraphGlmAll):
    """
    Test various input data types including outlier features.

    These unit tests cover a range of input data and check whether
    the overall graph works with different inputs. Only one
    training strategy is tested here. The cases tested are:

        - Dense X matrix: test_numpy_dense()
        - Sparse X matrix: test_scipy_sparse()
        - Dense X in anndata: test_anndata_dense()
        - Sparse X in anndata: test_anndata_sparse()
    """

    noise_model: str

    def input_data(self, data, design_loc, design_scale):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if not self.noise_model == "nb":
                raise ValueError("noise_model not recognized")

        return InputDataGLM(
            data=data,
            design_loc=design_loc,
            design_scale=design_scale,
        )

    def simulator(self, train_loc):
        return self.sim1

    def _test_numpy(self, sparse):
        self.simulate1()
        x = self.sim1.input_data.x
        if sparse:
            x = scipy.sparse.csr_matrix(x)
        self.sim1.input_data.x = x

        return self.basic_test_one_algo(batched=False, train_loc=True, train_scale=True, algo="IRLS", sparse=False)

    def _test_anndata(self, sparse):
        self.simulate1()
        x = self.sim1.input_data.x
        if sparse:
            x = scipy.sparse.csr_matrix(x)
        self.sim1.input_data.x = anndata.AnnData(X=x)

        return self.basic_test_one_algo(batched=False, train_loc=True, train_scale=True, algo="IRLS", sparse=False)

    def _test_anndata_raw(self, sparse):
        self.simulate1()
        x = self.sim1.input_data.x
        if sparse:
            x = scipy.sparse.csr_matrix(x)
        self.sim1.input_data.x = anndata.AnnData(X=x)
        self.sim1.input_data.x = Raw(self.sim1.input_data.x)

        return self.basic_test_one_algo(batched=False, train_loc=True, train_scale=True, algo="IRLS", sparse=False)


class TestDataTypesGlmNB(_TestDataTypesGlmAll, unittest.TestCase):
    """
    Test whether training graphs work for negative binomial noise.
    """

    def test_standard(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_DataTypes_GLM_NB.test_standard_nb()")

        self.noise_model = "nb"
        np.random.seed(1)
        self._test_numpy(sparse=False)
        self._test_numpy(sparse=True)

        return True

    def test_anndata(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_DataTypes_GLM_NB.test_anndata_nb()")

        self.noise_model = "nb"
        np.random.seed(1)
        self._test_anndata(sparse=False)
        self._test_anndata(sparse=True)
        self._test_anndata_raw(sparse=False)
        self._test_anndata_raw(sparse=True)

        return True


if __name__ == "__main__":
    unittest.main()
