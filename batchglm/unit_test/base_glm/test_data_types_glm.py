import abc
from typing import List
import unittest
import scipy.sparse

from batchglm.models.base_glm import _Estimator_GLM, InputData, _Simulator_GLM


class _Test_DataTypes_GLM_Estim():

    def __init__(
            self,
            estimator: _Estimator_GLM
    ):
        self.estimator = estimator

    def test_estimation(self):
        self.estimator.initialize()

        self.estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": 1e-1,
                "use_batching": False,
                "optim_algo": "Newton",
            },
        ])
        estimator_store = self.estimator.finalize()
        return estimator_store


class Test_DataTypes_GLM(unittest.TestCase, metaclass=abc.ABCMeta):
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
    sim: _Simulator_GLM
    _estims: List[_Estimator_GLM]

    def setUp(self):
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.estimator.close_session()

    @abc.abstractmethod
    def get_simulator(self):
        pass

    def simulate(self):
        self.sim = self.get_simulator()
        self.sim.generate_sample_description(num_batches=2, num_conditions=2)
        self.sim.generate()

    @abc.abstractmethod
    def input_data(
            self,
            data,
            design_loc,
            design_scale
    ) -> InputData:
        pass

    @abc.abstractmethod
    def get_estimator(
            self,
            input_data: InputData
    ) -> _Estimator_GLM:
        pass

    def basic_test(
            self,
            data,
            design_loc,
            design_scale
    ):
        input_data = self.input_data(
            data=data,
            design_loc=design_loc,
            design_scale=design_scale
        )
        estimator = self.get_estimator(input_data=input_data)
        return estimator.test_estimation()

    def _test_numpy(self, sparse):
        if sparse:
            X = scipy.sparse.csr_matrix(self.sim.X)

        success = self.basic_test(
            data=X,
            design_loc=self.sim.design_loc,
            design_scale=self.sim.design_scale
        )
        assert success, "_test_anndata with sparse=%s did not work" % sparse

    def _test_anndata(self, sparse):
        adata = self.sim.data_to_anndata()
        if sparse:
            adata.X = scipy.sparse.csr_matrix(adata.X)

        success = self.basic_test(
            data=adata,
            design_loc=self.sim.design_loc,
            design_scale=self.sim.design_scale
        )
        assert success, "_test_anndata with sparse=%s did not work" % sparse

    def _test_anndata_raw(self, sparse):
        adata = self.sim.data_to_anndata()
        if sparse:
            adata.X = scipy.sparse.csr_matrix(adata.X)

        adata.raw = adata
        success = self.basic_test(
            data=adata.raw,
            design_loc=self.sim.design_loc,
            design_scale=self.sim.design_scale
        )
        assert success, "_test_anndata with sparse=%s did not work" % sparse


if __name__ == '__main__':
    unittest.main()
