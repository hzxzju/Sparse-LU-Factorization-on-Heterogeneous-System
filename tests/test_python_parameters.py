"""Check public argument forwarding without requiring the compiled extension."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


class ParameterTests(unittest.TestCase):
    def setUp(self):
        name = "_pyglu_parameter_test"
        backend = types.ModuleType(name + "._pyglu")
        self.factorization = Mock()
        backend.GLUFactorization = self.factorization
        path = Path(__file__).resolve().parents[1] / "pyglu" / "__init__.py"
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: self.module, backend.__name__: backend}):
            spec.loader.exec_module(self.module)
        self.matrix = ([2.0, 3.0], [0, 1], [0, 1, 2], (2, 2))

    def test_default_keeps_right_looking(self):
        self.module.splu(self.matrix)
        self.assertEqual(self.factorization.call_args.args[-3:], (False, "right-looking", "level"))

    def test_left_looking_with_perturbation(self):
        self.module.splu(self.matrix, perturb=True, update_strategy="left-looking")
        self.assertEqual(self.factorization.call_args.args[-3:], (True, "left-looking", "level"))

    def test_alias_reaches_native_parser(self):
        self.module.splu(self.matrix, update_strategy="ll")
        self.assertEqual(self.factorization.call_args.args[-2:], ("ll", "level"))

    def test_spsolve_forwards_strategy_and_rhs(self):
        result = self.module.spsolve(self.matrix, [4, 6], True, "left-looking")
        self.assertEqual(self.factorization.call_args.args[-3:], (True, "left-looking", "level"))
        np.testing.assert_array_equal(self.factorization.return_value.solve.call_args.args[0], [4., 6.])
        self.assertIs(result, self.factorization.return_value.solve.return_value)

    def test_sync_free_factorization_forwarding(self):
        self.module.splu(self.matrix, update_strategy="ll", scheduling="synchronization-free")
        self.assertEqual(self.factorization.call_args.args[-3:],
                         (False, "ll", "synchronization-free"))

    def test_sync_free_solve_forwarding(self):
        self.module.spsolve(self.matrix, [4, 6], True, "left-looking", "synchronization-free")
        self.assertEqual(self.factorization.call_args.args[-3:],
                         (True, "left-looking", "synchronization-free"))
        np.testing.assert_array_equal(self.factorization.return_value.solve.call_args.args[0], [4., 6.])


if __name__ == "__main__":
    unittest.main()
