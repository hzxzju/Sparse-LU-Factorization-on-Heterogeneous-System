"""Check fair backend invocation, timing summaries and failed measurements."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = '''import argparse
p = argparse.ArgumentParser()
p.add_argument('-i'); p.add_argument('-u'); p.add_argument('-s')
p.add_argument('--symbolic-backend'); p.add_argument('--symbolic-no-fallback', action='store_true')
p.add_argument('--symbolic-sf-workers', type=int, default=0)
a = p.parse_args()
assert a.u == 'left-looking' and a.s == 'synchronization-free'
assert a.symbolic_no_fallback == (a.symbolic_backend != 'cpu')
assert a.symbolic_backend != 'gpu-sf' or a.symbolic_sf_workers == 3
print('Symbolic time:', {'cpu': 12, 'gpu': 24, 'gpu-sf': 6}[a.symbolic_backend])
print('Matrix Row: 32\\nOriginal nonzero: 100\\nSymbolic nonzero: 200')
if a.symbolic_backend == 'gpu-sf':
    for k, v in dict(symbolic_gpu_total_ms=5, symbolic_gpu_compute_ms=2,
                     symbolic_gpu_pack_ms=1, symbolic_workers=3, symbolic_pred_merges=20,
                     symbolic_peak_workspace_bytes=4096, symbolic_predecessor_rows_read=100,
                     symbolic_task_parks=10).items():
        print(k + ':', v)
'''


class BenchmarkTests(unittest.TestCase):
    def run_case(self, fixture):
        with tempfile.TemporaryDirectory(prefix='glu-symbolic-benchmark-') as root:
            root = Path(root)
            matrix = root / 'a.mtx'; matrix.write_text('fixture')
            solver = root / 'solver'; solver.write_text('#!' + sys.executable + '\n' + fixture)
            solver.chmod(0o755)
            output = root / 'results'
            result = subprocess.run([sys.executable, str(ROOT / 'src/benchmark_symbolic_sf.py'),
                str(matrix), '--solver', str(solver), '--output-dir', str(output),
                '--repeats', '2', '--workers', '3'], capture_output=True, text=True, timeout=20)
            with (output / 'symbolic.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            return result, rows

    def test_backends_and_statistics(self):
        result, rows = self.run_case(FIXTURE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['backend'] for r in rows], ['cpu', 'gpu', 'gpu-sf'])
        self.assertEqual([r['samples'] for r in rows], ['2'] * 3)
        self.assertEqual([r['status'] for r in rows], ['ok'] * 3)
        self.assertEqual(float(rows[2]['speedup_vs_cpu']), 2)
        self.assertEqual(float(rows[2]['e2e_p90_ms']), 6)
        self.assertEqual(float(rows[2]['worker_median_ms']), 2)
        self.assertEqual(float(rows[2]['peak_workspace_bytes']), 4096)

    def test_failed_gpu_is_not_a_speedup(self):
        result, rows = self.run_case("import sys\nraise SystemExit(77)\n")
        self.assertEqual(result.returncode, 1)
        self.assertTrue(all(r['status'] == 'incomplete' and not r['speedup_vs_cpu'] for r in rows))

    def test_missing_pack_metric_is_reported_without_crashing(self):
        result, rows = self.run_case(FIXTURE.replace('symbolic_gpu_pack_ms=1,', ''))
        self.assertEqual(result.returncode, 1)
        self.assertNotIn('Traceback', result.stderr)
        self.assertEqual(rows[2]['status'], 'incomplete')
        self.assertIn('symbolic_gpu_pack_ms', rows[2]['note'])
        self.assertFalse(rows[2]['speedup_vs_cpu'])

    def test_nonfinite_gpu_compute_time_is_rejected(self):
        result, rows = self.run_case(FIXTURE.replace('symbolic_gpu_compute_ms=2',
                                                    'symbolic_gpu_compute_ms=float("nan")'))
        self.assertEqual(result.returncode, 1)
        self.assertIn('invalid metrics', rows[2]['note'])
        self.assertFalse(rows[2]['speedup_vs_cpu'])

    def test_different_symbolic_nnz_is_rejected(self):
        fixture = FIXTURE + "\nif a.symbolic_backend == 'gpu-sf': print('Symbolic nonzero: 201')\n"
        result, rows = self.run_case(fixture)
        self.assertEqual(result.returncode, 1)
        self.assertIn('differs from CPU', rows[2]['note'])
        self.assertFalse(rows[2]['speedup_vs_cpu'])

    def test_incomplete_cpu_baseline_produces_no_speedup(self):
        fixture = FIXTURE + "\nif a.symbolic_backend == 'cpu': raise SystemExit(2)\n"
        result, rows = self.run_case(fixture)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(rows[0]['status'], 'incomplete')
        self.assertTrue(all(not row['speedup_vs_cpu'] for row in rows))


if __name__ == '__main__':
    unittest.main()
