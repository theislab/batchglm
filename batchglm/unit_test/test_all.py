import unittest
loader = unittest.TestLoader()

# Run tests on GLMs
start_dir = 'batchglm.unit_test.glm_all'
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)