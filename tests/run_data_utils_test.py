import logging
import unittest

import numpy as np
import pandas as pd

from batchglm.utils.data import constraint_system_from_star

logger = logging.getLogger("batchglm")


class TestConstraintSystemFromStar(unittest.TestCase):

    true_cmat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    true_cmat_list = np.array(
        [[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    true_cmat_array = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    true_dmat = np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    true_dmat_list = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )

    true_dmat_array = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

    true_terms = ["condition", "batch"]
    true_coefs = ["condition[0]", "condition[1]", "batch[0]", "batch[1]", "batch[2]", "batch[3]"]

    true_terms_list = ['condition[0]', 'condition[1]', 'batch[T.1]', 'batch[T.2]', 'batch[T.3]']
    true_coefs_list = ["condition[0]", "condition[1]", "batch[T.1]", "batch[T.2]", "batch[T.3]"]

    true_terms_array = ['condition[0]', 'condition[1]', 'batch[T.1]']
    true_coefs_array = ["condition[0]", "condition[1]", "batch[T.1]"]

    # dict tests

    def execute_test_dict(self, *args, **kwargs):
        dmat, coef_names, cmat, term_names = constraint_system_from_star(*args, **kwargs)
        assert term_names == self.true_terms
        assert coef_names == self.true_coefs
        assert np.all(np.equal(cmat, self.true_cmat))
        assert np.all(np.equal(dmat, self.true_dmat))

    def test_constraint_system_dict(self):
        formula = "~0 + condition + batch"
        sample_description = pd.DataFrame({"condition": [0, 1, 1, 0, 0, 1], "batch": [1, 2, 2, 0, 0, 3]})
        constraints = {"batch": "condition"}
        self.execute_test_dict(constraints, sample_description=sample_description, formula=formula)

    # list tests

    def execute_test_list(self, *args, **kwargs):
        dmat, coef_names, cmat, term_names = constraint_system_from_star(*args, **kwargs)
        assert term_names == self.true_terms_list
        assert coef_names == self.true_coefs_list
        assert np.all(np.equal(cmat, self.true_cmat_list))
        assert np.all(np.equal(dmat, self.true_dmat_list))

    def test_constraint_system_list(self):
        formula = "~0 + condition + batch"
        sample_description = pd.DataFrame({"condition": [0, 1, 1, 0, 0, 1], "batch": [1, 2, 2, 0, 0, 3]})
        constraints = ["condition[0] + condition[1] = 0"]
        self.execute_test_list(constraints, sample_description=sample_description, formula=formula)

    def test_constraint_system_list_with_dmat(self):
        constraints = ["condition[0] + condition[1] = 0"]
        dmat = pd.DataFrame(self.true_dmat_list, columns=self.true_coefs_list)
        self.execute_test_list(constraints, dmat=dmat)

    # array tests

    def execute_test_array(self, *args, **kwargs):
        dmat, coef_names, cmat, term_names = constraint_system_from_star(*args, **kwargs)
        assert term_names == self.true_terms_array
        assert coef_names == self.true_coefs_array
        assert np.all(np.equal(cmat, self.true_cmat_array))
        assert np.all(np.equal(dmat, self.true_dmat_array))

    def test_constraint_system_array(self):
        formula = "~0 + condition + batch"
        sample_description = pd.DataFrame({"condition": [0, 1, 0, 1], "batch": [1, 0, 0, 1]})
        constraints = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.execute_test_array(constraints, sample_description=sample_description, formula=formula)

    def test_constraint_system_array_with_dmat(self):
        constraints = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        dmat = pd.DataFrame(self.true_dmat_array, columns=self.true_coefs_array)
        self.execute_test_array(constraints, dmat=dmat)


if __name__ == "__main__":
    unittest.main()
