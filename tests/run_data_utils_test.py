from batchglm.api.utils.data import constraint_system_from_star
import pandas as pd
import logging
import numpy as np
import unittest

logger = logging.getLogger("batchglm")

class TestConstraintSystemFromStar(unittest.TestCase):

    def test_constraint_system_dict(self):
        formula = '~0 + condition + batch'
        sample_description = pd.DataFrame({ 'condition': [0, 1, 1, 0, 0, 1], 'batch': [1, 2, 2, 0, 0, 3] })
        constraints = {"batch": "condition"}
        dmat, coef_names, cmat, term_names = constraint_system_from_star(constraints, sample_description=sample_description, formula=formula)
        assert term_names == ['condition', 'batch']
        assert coef_names == ['condition[0]', 'condition[1]', 'batch[0]', 'batch[1]', 'batch[2]', 'batch[3]']
        assert np.all(np.equal(cmat, np.array([[ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0., -1.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  0., -1.],
            [ 0.,  0.,  0.,  1.]])))
        assert np.all(np.equal(dmat, np.array([[1., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 1., 0.],
            [0., 1., 0., 0., 1., 0.],
            [1., 0., 1., 0., 0., 0.],
            [1., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 1.]])))
    
    def test_constraint_system_list(self):
        formula = '~0 + condition + batch'
        sample_description = pd.DataFrame({ 'condition': [0, 1, 1, 0, 0, 1], 'batch': [1, 2, 2, 0, 0, 3] })
        constraints = ["condition[0] + condition[1] = 0"]
        dmat, coef_names, cmat, term_names = constraint_system_from_star(constraints, sample_description=sample_description, formula=formula)
        assert term_names == None
        assert coef_names == ['condition[0]', 'condition[1]', 'batch[T.1]', 'batch[T.2]', 'batch[T.3]']
        assert np.all(np.equal(cmat, np.array([[-1.,  0.,  0.,  0.],
                [ 1.,  0.,  0.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1.,  0.],
                [ 0.,  0.,  0.,  1.]])))
        assert np.all(np.equal(dmat, np.array([[1., 0., 1., 0., 0.],
                [0., 1., 0., 1., 0.],
                [0., 1., 0., 1., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 1.]])))
    
    def test_constraint_system_list(self):
        formula = '~0 + condition + batch'
        sample_description = pd.DataFrame({ 'condition': [0, 1, 1, 0, 0, 1], 'batch': [1, 2, 2, 0, 0, 3] })
        dmat = pd.DataFrame({ 'condition_0': [1, 0, 0, 1, 1, 0], 'condition_1': [0, 1, 1, 0, 0, 1], 'batch_0': [0, 0, 0, 1, 1, 0], 'batch_1': [1, 0, 0, 0, 0, 0], 'batch_2': [0, 1, 1, 0, 0, 0], 'batch_3': [0, 0, 0, 0, 0, 1] })
        constraints = np.array([[-1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        dmat, coef_names, cmat, term_names = constraint_system_from_star(constraints, sample_description=sample_description, formula=formula)
        assert term_names == None
        assert coef_names == ['condition[0]', 'condition[1]', 'batch[T.1]', 'batch[T.2]', 'batch[T.3]']
        assert np.all(np.equal(cmat, np.array([[-1.,  0.,  0.,  0.],
                [ 1.,  0.,  0.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1.,  0.],
                [ 0.,  0.,  0.,  1.]])))
        assert np.all(np.equal(dmat, np.array([[1., 0., 1., 0., 0.],
                [0., 1., 0., 1., 0.],
                [0., 1., 0., 1., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 1.]])))



if __name__ == "__main__":
    unittest.main()