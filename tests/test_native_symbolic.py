"""Native parameter checks; opt in to hardware solves with GLU_TEST_GPU=1."""
import os
import unittest

try:
    import numpy as np
    import pyglu
    _IMPORT_ERROR = None
except ImportError as exc:
    _IMPORT_ERROR = str(exc)


@unittest.skipIf(_IMPORT_ERROR is not None, "native extension unavailable: " + str(_IMPORT_ERROR))
class NativeParameterTests(unittest.TestCase):
    def setUp(self):
        self.data = np.array([2., 3.], dtype=np.float64)
        self.indices = np.array([0, 1], dtype=np.int32)
        self.indptr = np.array([0, 1, 2], dtype=np.int32)

    def factorize(self, **options):
        return pyglu.GLUFactorization(self.data, self.indices, self.indptr, 2, **options)

    def test_invalid_configuration_fails_before_device_initialization(self):
        for options in (
                dict(symbolic_backend="bad"), dict(symbolic_backend="cpu", symbolic_verify=True),
                dict(symbolic_backend="cpu", symbolic_no_fallback=True),
                dict(symbolic_backend="cpu", symbolic_sf_workers=1),
                dict(symbolic_backend="gpu-sf", symbolic_block_size=96),
                dict(symbolic_backend="gpu-sf", symbolic_pool_entries=2, symbolic_max_pool_entries=1),
                dict(symbolic_batch_cols=0), dict(sf_workers=1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.factorize(**options)

    def test_invalid_csc_is_rejected_before_preprocessing(self):
        cases = [(self.data.reshape(1, 2), self.indices, self.indptr),
                 (self.data, np.array([0, 2], dtype=np.int32), self.indptr),
                 (self.data, self.indices, np.array([1, 1, 2], dtype=np.int32)),
                 (self.data, self.indices, np.array([0, 2, 1], dtype=np.int32)),
                 (self.data, self.indices, np.array([0, 1, 3], dtype=np.int32))]
        for arrays in cases:
            with self.subTest(arrays=arrays), self.assertRaises(ValueError):
                pyglu.GLUFactorization(*arrays, 2)


@unittest.skipUnless(_IMPORT_ERROR is None and os.environ.get("GLU_TEST_GPU") == "1",
                     "requires native extension and explicit GLU_TEST_GPU=1")
class NativeSymbolicGPUTests(unittest.TestCase):
    def test_all_backends_solve_and_retain_diagnostics(self):
        matrix = ([4., 1., .5, .5, 5., 1., 1., 6., .5, 1., .5, 7.],
                  [0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3], [0, 3, 6, 9, 12], (4, 4))
        expected = np.array([1., 1.25, 1.5, 1.75])
        rhs = np.zeros(4)
        for col in range(4):
            for p in range(matrix[2][col], matrix[2][col + 1]):
                rhs[matrix[1][p]] += matrix[0][p] * expected[col]
        for backend in ("auto", "cpu", "gpu", "gpu-sf"):
            with self.subTest(backend=backend):
                factor = pyglu.splu(matrix, update_strategy="ll", scheduling="synchronization-free",
                                    sf_workers=1, symbolic_backend=backend,
                                    symbolic_verify=backend != "cpu", symbolic_no_fallback=backend != "cpu",
                                    symbolic_sf_workers=2 if backend == "gpu-sf" else 0)
                actual_backend = "cpu" if backend == "auto" else backend
                self.assertEqual(factor.symbolic_backend, actual_backend)
                self.assertIn("Symbolic backend: " + actual_backend, factor.diagnostics)
                if backend == "auto":
                    self.assertIn("Symbolic auto selected backend: cpu", factor.diagnostics)
                for scale in (1., 2.):
                    np.testing.assert_allclose(factor.solve(rhs * scale), expected * scale, atol=1e-11)
                with self.assertRaises(ValueError):
                    factor.solve(rhs.reshape(2, 2))
                if backend == "gpu-sf":
                    self.assertIn("symbolic_gpu_compute_ms:", factor.diagnostics)

    def test_pool_failure_cleanup_and_actual_fallback_backend(self):
        matrix = ([2., 3.], [0, 1], [0, 1, 2], (2, 2))
        options = dict(symbolic_backend="gpu-sf", symbolic_pool_entries=1, symbolic_max_pool_entries=1)
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                pyglu.splu(matrix, symbolic_no_fallback=True, **options)
        factor = pyglu.splu(matrix, **options)
        self.assertEqual(factor.symbolic_backend, "cpu")
        self.assertIn("symbolic_cpu_fallback_count: 1", factor.diagnostics)
        np.testing.assert_allclose(factor.solve(np.array([4., 6.])), [2., 2.], atol=1e-11)


if __name__ == "__main__":
    unittest.main()
