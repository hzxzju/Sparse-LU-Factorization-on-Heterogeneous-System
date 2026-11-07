"""Check public argument forwarding without requiring the compiled extension."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

try:
    import numpy as np
except ImportError:
    np = None


class ParameterTests(unittest.TestCase):
    def setUp(self):
        name = "_pyglu_parameter_test"
        backend = types.ModuleType(name + "._pyglu")
        self.factorization = Mock()
        backend.GLUFactorization = self.factorization
        path = Path(__file__).resolve().parents[1] / "pyglu" / "__init__.py"
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
        self.module = importlib.util.module_from_spec(spec)
        modules = {name: self.module, backend.__name__: backend}
        if np is None:
            # These are argument forwarding checks. Array dtype/ABI and solves
            # belong to native integration tests, which require real NumPy.
            numpy_stub = types.ModuleType("numpy")
            numpy_stub.float64, numpy_stub.int32 = "float64", "int32"
            numpy_stub.ascontiguousarray = Mock(side_effect=lambda value, dtype: value)
            numpy_stub.asarray = Mock(side_effect=lambda value, dtype: value)
            modules["numpy"] = numpy_stub
            modules["scipy.sparse"] = None
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.module)
        self.matrix = ([2.0, 3.0], [0, 1], [0, 1, 2], (2, 2))

    def test_default_keeps_right_looking(self):
        self.module.splu(self.matrix)
        self.assertEqual(self.factorization.call_args[0][-3:], (False, "right-looking", "level"))
        self.assertEqual(self.factorization.call_args[1], dict(
            sf_workers=0, symbolic_backend="auto", symbolic_batch_cols=128,
            symbolic_block_size=256, symbolic_verify=False, symbolic_no_fallback=False,
            symbolic_sf_workers=0, symbolic_pool_entries=0, symbolic_max_pool_entries=0,
            numeric_sm_padding=False))

    def test_left_looking_with_perturbation(self):
        self.module.splu(self.matrix, perturb=True, update_strategy="left-looking")
        self.assertEqual(self.factorization.call_args[0][-3:], (True, "left-looking", "level"))

    def test_default_spsolve_uses_auto_symbolic(self):
        self.module.spsolve(self.matrix, [4, 6])
        self.assertEqual(self.factorization.call_args[1]["symbolic_backend"], "auto")

    def test_explicit_cpu_overrides_auto(self):
        self.module.splu(self.matrix, symbolic_backend="cpu")
        self.assertEqual(self.factorization.call_args[1]["symbolic_backend"], "cpu")

    def test_auto_forwards_gpu_options(self):
        self.module.splu(self.matrix, symbolic_backend="auto", symbolic_sf_workers=3,
                         symbolic_verify=True, symbolic_no_fallback=True)
        self.assertEqual(self.factorization.call_args[1]["symbolic_backend"], "auto")
        self.assertEqual(self.factorization.call_args[1]["symbolic_sf_workers"], 3)
        self.assertTrue(self.factorization.call_args[1]["symbolic_no_fallback"])

    def test_alias_reaches_native_parser(self):
        self.module.splu(self.matrix, update_strategy="ll")
        self.assertEqual(self.factorization.call_args[0][-2:], ("ll", "level"))

    def test_spsolve_forwards_strategy_and_rhs(self):
        result = self.module.spsolve(self.matrix, [4, 6], True, "left-looking")
        self.assertEqual(self.factorization.call_args[0][-3:], (True, "left-looking", "level"))
        self.assertEqual(list(self.factorization.return_value.solve.call_args[0][0]), [4., 6.])
        self.assertIs(result, self.factorization.return_value.solve.return_value)

    def test_sync_free_factorization_forwarding(self):
        self.module.splu(self.matrix, update_strategy="ll", scheduling="synchronization-free")
        self.assertEqual(self.factorization.call_args[0][-3:],
                         (False, "ll", "synchronization-free"))

    def test_sync_free_solve_forwarding(self):
        self.module.spsolve(self.matrix, [4, 6], True, "left-looking", "synchronization-free")
        self.assertEqual(self.factorization.call_args[0][-3:],
                         (True, "left-looking", "synchronization-free"))
        self.assertEqual(list(self.factorization.return_value.solve.call_args[0][0]), [4., 6.])

    def test_symbolic_options_reach_both_native_entry_points(self):
        options = dict(sf_workers=1, symbolic_backend="gpu-sf", symbolic_batch_cols=3,
                       symbolic_block_size=64, symbolic_verify=True, symbolic_no_fallback=True,
                       symbolic_sf_workers=2, symbolic_pool_entries=1, symbolic_max_pool_entries=128,
                       numeric_sm_padding=True)
        self.module.splu(self.matrix, update_strategy="ll", scheduling="synchronization-free", **options)
        factor_args, factor_kwargs = self.factorization.call_args
        self.assertEqual(factor_kwargs, options)
        self.module.spsolve(self.matrix, [4, 6], update_strategy="ll",
                            scheduling="synchronization-free", **options)
        self.assertEqual(self.factorization.call_args[1], options)
        self.assertEqual(self.factorization.call_args[0][3:], factor_args[3:])

    def test_new_options_are_keyword_only(self):
        with self.assertRaises(TypeError):
            self.module.splu(self.matrix, False, "right-looking", "level", "gpu-sf")
        self.factorization.assert_not_called()

    def test_symbolic_is_independent_of_numeric_scheduling(self):
        self.module.splu(self.matrix, symbolic_backend="gpu-sf", symbolic_sf_workers=2)
        self.assertEqual(self.factorization.call_args[0][-2:], ("right-looking", "level"))
        self.assertEqual(self.factorization.call_args[1]["symbolic_backend"], "gpu-sf")


if __name__ == "__main__":
    unittest.main()
