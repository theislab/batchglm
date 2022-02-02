import unittest
loader = unittest.TestLoader()

# Run all unit tests.
# Tests are loaded from all subdirectories of batchglm.unit_test
start_dir = 'batchglm.unit_test'
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)
