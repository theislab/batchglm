import unittest

loader = unittest.TestLoader()

# Run all unit tests.
# Tests are loaded from all subdirectories of tests
start_dir = "tests"
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)
