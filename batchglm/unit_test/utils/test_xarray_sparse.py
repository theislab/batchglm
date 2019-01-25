import logging
import unittest
import numpy as np
import scipy.sparse
import xarray as xr

import batchglm.api as glm
from batchglm.xarray_sparse import SparseXArrayDataArray, SparseXArrayDataSet

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class TestXarraySparse(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def simulate(self):
        nobs = 500
        nobs_per_group = 100
        nfeatures = 100
        X_dense = np.vstack([np.random.uniform(low=0, high=100*(i+1), size=[nobs_per_group, nfeatures]) for i in range(5)])
        feature_names = ["f"+str(i) for i in range(nfeatures)]
        obs_names = ["o" + str(i) for i in range(nobs)]

        self.xr = xr.Dataset(
            data_vars={
                "X": (("observations", "features"), X_dense)
            },
            coords={
                "features": feature_names,
                "observations": obs_names
            }
        )

        self.xr_sp = SparseXArrayDataSet(
            X=scipy.sparse.csr_matrix(X_dense),
            feature_names=feature_names,
            obs_names=obs_names
        )
        self.groups = [["e", "a", "c", "b", "d"][x] for x in np.arange(0, nobs) // 100]
        self.a = np.random.uniform(low=0, high=1000, size=[nobs, nfeatures])

    def _test_add(self):
        a_xr = self.xr["X"] + self.a
        a_xr_sp = self.xr_sp["X"].add(self.a)

        max_abs_dev = np.max(np.abs(a_xr.values - a_xr_sp.X.todense()))
        assert max_abs_dev < 1e-08
        return True

    def _test_means(self):
        self.simulate()

        means_xr = self.xr["X"].mean(self.xr["X"].dims[0]).values
        means_xr_sp = self.xr_sp["X"].mean(self.xr_sp["X"].dims[0])

        max_abs_dev = np.max(np.abs(means_xr - means_xr_sp))
        assert max_abs_dev < 1e-08
        return True

    def _test_vars(self):
        vars_xr = self.xr["X"].var(self.xr["X"].dims[0]).values
        vars_xr_sp = self.xr_sp["X"].var(self.xr_sp["X"].dims[0])

        max_abs_dev = np.max(np.abs(vars_xr - vars_xr_sp))

        assert max_abs_dev < 1e-08
        return True

    def _test_std(self):
        std_xr = self.xr["X"].std(self.xr["X"].dims[0]).values
        std_xr_sp = self.xr_sp["X"].std(self.xr_sp["X"].dims[0])

        max_abs_dev = np.max(np.abs(std_xr - std_xr_sp))

        assert max_abs_dev < 1e-08
        return True

    def _test_group_add(self):
        assert False, "not implemented"
        return True

    def _test_group_means(self):
        xr_grouped = self.xr["X"].assign_coords(group=((self.xr["X"].dims[0],), self.groups)).groupby("group")
        means_xr = xr_grouped.mean(self.xr["X"].dims[0]).values

        self.xr_sp["X"].assign_coords(("groups", self.groups))
        self.xr_sp["X"].groupby("groups")
        means_xr_sp = self.xr_sp["X"].group_means(self.xr_sp["X"].dims[0])

        max_abs_dev = np.max(np.abs(means_xr - means_xr_sp))
        assert max_abs_dev < 1e-08
        return True

    def _test_group_vars(self):
        xr_grouped = self.xr["X"].assign_coords(group=((self.xr["X"].dims[0],), self.groups)).groupby("group")
        vars_xr = xr_grouped.var(self.xr["X"].dims[0]).values

        self.xr_sp["X"].assign_coords(("groups", self.groups))
        self.xr_sp["X"].groupby("groups")
        vars_xr_sp = self.xr_sp["X"].group_vars(self.xr_sp["X"].dims[0])

        max_abs_dev = np.max(np.abs(vars_xr - vars_xr_sp))
        assert max_abs_dev < 1e-08
        return True

    def test_all_methods(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)

        logger.error("TestXarraySparse.test_all_methods()")

        self.simulate()
        self._test_add()
        self._test_means()
        self._test_vars()
        self._test_std()
        self._test_group_means()
        self._test_group_vars()


if __name__ == '__main__':
    unittest.main()
