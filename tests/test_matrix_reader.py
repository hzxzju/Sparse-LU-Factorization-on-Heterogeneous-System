"""Exercise the native CSR reader without requiring a GPU or Python bindings."""
import ctypes as ct
import itertools
from pathlib import Path
import subprocess
import tempfile
import unittest


class MatrixReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(__file__).resolve().parents[1]
        library = Path(cls.temp.name) / "reader.so"
        subprocess.run([
            "gcc", "-shared", "-fPIC", "-O2",
            "-I" + str(cls.root / "src/nicslu/include"),
            str(cls.root / "src/nicslu/util/nicslu_readfile.c"),
            "-o", str(library),
        ], check=True)
        cls.library = ct.CDLL(str(library))
        cls.reader = cls.library.NicsLU_ReadTripletRowToSparse
        cls.reader.argtypes = [ct.c_char_p, ct.POINTER(ct.c_uint),
                              ct.POINTER(ct.c_uint),
                              ct.POINTER(ct.POINTER(ct.c_double)),
                              ct.POINTER(ct.POINTER(ct.c_uint)),
                              ct.POINTER(ct.POINTER(ct.c_uint))]
        cls.reader.restype = ct.c_int
        cls.libc = ct.CDLL(None)
        cls.libc.free.argtypes = [ct.c_void_p]
        cls.libc.free.restype = None

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def read(self, path):
        n, nnz = ct.c_uint(), ct.c_uint()
        values = ct.POINTER(ct.c_double)()
        indices, offsets = ct.POINTER(ct.c_uint)(), ct.POINTER(ct.c_uint)()
        try:
            result = self.reader(str(path).encode(), ct.byref(n), ct.byref(nnz),
                                 ct.byref(values), ct.byref(indices), ct.byref(offsets))
            if result != 0:
                return result, [], []
            ap = list(offsets[:n.value + 1])
            self.assertEqual(ap[0], 0)
            self.assertEqual(ap[-1], nnz.value)
            self.assertEqual(ap, sorted(ap))
            entries = [(row + 1, indices[k] + 1, values[k])
                       for row in range(n.value)
                       for k in range(ap[row], ap[row + 1])]
            return result, ap, entries
        finally:
            for pointer in (values, indices, offsets):
                self.libc.free(pointer)

    def write(self, n, entries):
        path = Path(self.temp.name) / "matrix.mtx"
        path.write_text("%%MatrixMarket matrix coordinate real general\n"
                        + f"{n} {n} {len(entries)}\n"
                        + "".join(f"{r} {c} {v}\n" for r, c, v in entries))
        return path

    def test_all_triplet_orders_preserve_matrix(self):
        entries = [(1, 1, 8.0), (3, 1, -2.0), (1, 2, 3.5), (2, 3, 7.0)]
        for order in itertools.permutations(entries):
            with self.subTest(order=order):
                result, ap, actual = self.read(self.write(3, order))
                self.assertEqual(result, 0)
                self.assertEqual(ap, [0, 2, 3, 4])
                self.assertCountEqual(actual, entries)

    def test_leading_middle_and_trailing_empty_rows(self):
        entries = [(4, 1, 2.0), (2, 5, 3.0), (4, 2, 4.0)]
        result, ap, actual = self.read(self.write(5, entries))
        self.assertEqual(result, 0)
        self.assertEqual(ap, [0, 0, 1, 1, 3, 3])
        self.assertCountEqual(actual, entries)

    def test_invalid_coordinates_are_rejected(self):
        for entry in [(0, 1, 1), (4, 1, 1), (1, 0, 1), (1, 4, 1)]:
            with self.subTest(entry=entry):
                self.assertEqual(self.read(self.write(3, [entry]))[0], -7)

    def test_bundled_column_ordered_matrix(self):
        result, ap, entries = self.read(self.root / "src/nicslu/demo/ASIC_100k.mtx")
        self.assertEqual(result, 0)
        self.assertEqual(len(ap), 99341)
        self.assertEqual(len(entries), 954163)
        self.assertIn((11692, 1, -0.0003333333333333333), entries)


if __name__ == "__main__":
    unittest.main()
