# GLU-accelerated Sparse Parallel LU factorization solver V3.0

Last update: March 19, 2026

## Authors:
Shaoyi Peng (speng004@ucr.edu)\
Kai He (khe004@ucr.edu)\
Sheldon Tan (stan@ece.ucr.edu)

Please contact Sheldon Tan for any question. 

Additional information: 
https://intra.ece.ucr.edu/~stan/project/glu/glu_proj.htm

GitHub Pages:
https://sheldonucr.github.io/GLU_public/

## License
USB 3-Clause License 

## Sub-directories
docs: contains some related document's and publications for GLU
src: contains all the source codes for GLU
pyglu: Python package (bindings for the GPU solver)

## Python Bindings (pyglu)

`pyglu` exposes the GLU solver as a Python library with an API similar to `scipy.sparse.linalg.splu` / `spsolve`.

### Requirements
- HTHPCC SDK; set `HPCC_PATH` if it is not installed under `/opt/hpcc`
- GCC / G++
- Python ≥ 3.9, pybind11 ≥ 2.11, NumPy ≥ 1.22
- SciPy ≥ 1.9 (optional, for passing sparse matrices directly)

### Installation
```bash
pip install pybind11
pip install -e . --no-build-isolation
```

### Usage
```python
import scipy.sparse as sp
import numpy as np
import pyglu

A = sp.random(1000, 1000, density=0.01, format='csc') + sp.eye(1000)
b = np.ones(1000)

# Factorize once, solve multiple times
lu = pyglu.splu(A)
x1 = lu.solve(b)
x2 = lu.solve(np.random.rand(1000))

# Or solve directly
x = pyglu.spsolve(A, b)

# Enable diagonal perturbation for near-singular matrices
x = pyglu.spsolve(A, b, perturb=True)

# Select left-looking updates through the module parameter
lu = pyglu.splu(A, update_strategy="left-looking")
x = lu.solve(b)
print(lu.update_strategy)  # left-looking
x = pyglu.spsolve(A, b, perturb=True, update_strategy="left-looking")

# CPU symbolic factorization + synchronization-free GPU numeric LL
lu = pyglu.splu(A, update_strategy="left-looking",
               scheduling="synchronization-free")
x = lu.solve(b)
print(lu.scheduling)  # synchronization-free

# Opt in to GPU symbolic SF independently of the numeric strategy
lu = pyglu.splu(A, update_strategy="left-looking", scheduling="synchronization-free",
               sf_workers=1, symbolic_backend="gpu-sf", symbolic_sf_workers=2,
               symbolic_verify=True, symbolic_no_fallback=True)
x = lu.solve(b)
print(lu.symbolic_backend)  # actual backend; cpu if fallback was enabled and used
print(lu.diagnostics)      # symbolic timing, pools, parks, merges and numeric diagnostics
```

`splu` accepts any scipy sparse matrix (any format) or a raw `(data, indices, indptr, shape)` tuple in CSC format. GLU's `REAL` type is `double`; the Python interface accepts and returns float64 arrays.

`update_strategy` defaults to `"right-looking"`; `"left-looking"` selects the
column-owned kernel in `src/left_looking.cuh`. The aliases `"right"`/`"rl"` and
`"left"`/`"ll"` are accepted. Both strategies default to automatic CPU/GPU-SF symbolic selection
and use the same diagonal perturbation rule. `scheduling="level"` remains the default;
`scheduling="synchronization-free"` is supported only with left-looking updates.
Left-looking collects
previous L-column contributions into the current column before dividing by its
pivot; it does not broadcast updates into future columns. Rebuild the extension
after changing the native code. `symbolic_backend="gpu"` selects the existing
graph search backend; `"gpu-sf"` selects bitmap union of published predecessor L
patterns. Both keep the existing CPU CSR/value preparation. All new options are
keyword-only in `splu`, `spsolve` and `GLUFactorization`; the existing positional
arguments retain their meaning. `sf_workers` controls numeric SF workers;
`symbolic_sf_workers` independently controls symbolic SF workers. Both default
to 0 for automatic selection. GPU symbolic failures fall back to CPU unless
`symbolic_no_fallback=True`; input/verification errors always raise. See the
[interface and default mapping](src/RUN_BATCH.md#符号分析接口一致性) for all options.

`symbolic_backend="auto"` is the default in C++, the CLI/batch runner and Python.
After preprocessing, let `m_j = ap[j+1] - ap[j]` and
`B = sum(m_j * max(0, m_j - (n-j)))` for zero-based column `j`. Auto selects
GPU-SF only when all four conditions hold: `n >= 20000`,
`max(m_j) / n >= 0.10`, `max(m_j) / (nnz/n) >= 64`, and `B >= 1e9`.
Otherwise it selects CPU. The decision takes O(n) time and O(1) extra space,
using 64-bit integer comparisons without starting the GPU. Explicit `cpu`,
`gpu` and `gpu-sf` override auto. Logs record the selection and its features;
`lu.symbolic_backend` reports the actual backend, including CPU fallback.
GPU verification, strict failure handling and SF worker/pool options apply
only when auto selects GPU-SF. Auto selecting CPU is not a GPU fallback.

The extension now compiles both `numeric.cu` and `symbolic_gpu.cu` with the same
HTHPCC barrier alias, toolchain flags and PIC. `GCC_TOOLCHAIN`, `GCC_VERSION`,
`HPCC_PATH`, `HTCC`, `CC` and `CXX` can override the build tools. The default GCC
toolchain matches `src/Makefile`: `$HOME/local/gcc-7.3.0/usr`.

The SF path uses `src/left_looking_sf.cuh`: persistent CTA workers atomically
claim increasing natural column IDs, consume each completed predecessor L
column in ascending U-row order, and publish completion after normalization
and all-lane factor writeback. One worker owns one dense workspace, reused
between columns. No level graph is generated for SF. The automatic worker count
starts at two workers per multiprocessor, bounded by the matrix size, grid limit,
and a workspace budget of min(256 MiB, total device memory / 16), with allocation
fallback down to one worker. This is a tuning heuristic, not an occupancy claim.

Ordered atomics are isolated in `src/sf_sync.cuh`. Production uses the SDK
`__hpcc_atomic_*` builtins with explicit `memory_scope_device` and acquire/release
publication. The compiler must provide these builtins; no relaxed fallback is
used. Run `tests/sf_sync.cu` on the target GPU to validate progress and visibility;
CUDA compatibility tests exercise a separate test-only adapter. For timeout
diagnosis, use CLI/batch `--progress` and compare automatic workers against
`--sf-workers 1`; see [src/RUN_BATCH.md](src/RUN_BATCH.md).

## Batch stage timings

Rebuild the native solver and run the batch script from the repository root:

```sh
make -C src MAIN
python3 src/run_batch.py --update-strategy left-looking \
  --scheduling synchronization-free --timeout 600
```

The solver records stage timings without per-kernel profiling. Batch runs write
`summary.csv`, per-matrix logs and solution files; see
[src/RUN_BATCH.md](src/RUN_BATCH.md) for timing definitions and options.

## CLI solution error and iterative refinement

After each GLU solve, `lu_cmd` reads the existing CPU KLU solution from
`src/KLU_standalone/test_result/<matrix_stem>_x.txt`; it does not run KLU.
Both solutions must correspond to the original matrix and `b = ones(n)`.
Here `x_GPU` means the final GLU solution (GPU LU plus CPU triangular solve
and any refinement). Logs report scientific notation with at least 17 digits:

```text
GPU-CPU absolute L2 error: ...
GPU-CPU relative L2 error: ...
```

These are `||x_GPU-x_CPU||_2` and `||x_GPU-x_CPU||_2 / ||x_CPU||_2`.
Reference-file reading and solution comparison pause all stage timers, so they
are excluded from `once_s`, `solve_s`, `post_s` and `total_solve_ms`.
The existing CPU files are only read. Missing, malformed, nonfinite or
wrong-length reference vectors cause an explicit error.

```sh
make -C src MAIN
./src/lu_cmd -i src/matrix/add32_csr.mtx
./src/lu_cmd -i src/matrix/add32_csr.mtx --cpu-reference /path/to/add32_csr_x.txt
python3 src/run_batch.py --cpu-reference-dir src/KLU_standalone/test_result
```

`--residual-tol` defaults to `1e-6` (finite and positive). Exit code 0 means the
absolute GPU/CPU L2 error meets this tolerance; 2 means it misses the target.
Relative L2 error is reported separately and is not the acceptance criterion.
For a zero CPU vector, relative error is 0 if both solutions are zero and
infinity otherwise. Batch CSV has 29 columns: `solution_error_l2` and
`relative_solution_error_l2` hold solution errors; `axb_residual_l1`,
`axb_residual_l2`, `axb_residual_linf` hold auxiliary equation residuals.
The old ambiguous `residual_l1/l2/linf` column names are removed.
A missed comparison target is recorded as `residual_not_converged`.

Iterative refinement still uses `r=b-Ax`, existing GLU factors and the same
`--residual-tol` as its equation-residual stopping tolerance. It does not use
the CPU reference to correct the solution. `--max-refine-iters` defaults to
`n` (matrix order); `0` disables corrections. Refinement stops on convergence,
the cap, stagnation or nonfinite arithmetic, preserving the best solution.
`Refinement status` describes this internal Ax-b check; the separate
`GPU-CPU comparison status` controls the final exit code. Updates and residual
accumulation retain CPU `long double` precision; factors and solves stay double.
`x.dat` preserves the final extended-precision GLU solution.

Refinement remains in `solve_s`; independent final Ax-b checks and output are
in `post_s`. `total_solve_ms = (0.005 * once_s + solve_s) * 1000`.
After cleanup, stdout contains only `once_s=... solve_s=... post_s=...`
(seconds, six decimals); comparison results and diagnostics go to stderr.
The Python `pyglu` API retains its existing behavior. See
[src/RUN_BATCH.md](src/RUN_BATCH.md) for all output and timing definitions.

## Publications: 
J1.  K. He, S. X.-D. Tan, H. Wang and G. Shi, “GPU-Accelerated Parallel Sparse LU Factorization Method for Fast Circuit Analysis”, IEEE Transactions on Very Large Scale Integrated Systems  (TVLSI), vol. 24, no.3, pp.1140-1150, March 2016.

J2. S. Peng and S. X.-D. Tan, “GLU3.0:  Fast GPU-based Parallel Sparse LU Factorization for Circuit Simulation”,  IEEE Design and Test (accepted in Feb 2020), pre-print is available at  http://arxiv.org/abs/1908.00204


## Some recent bug fixes by Codex, Feb 2026

CUDA sync bug that could deadlock kernels (__syncthreads() on divergent path)

Fixed in numeric.cu (line 270) (kernel RL_onecol_updateSubmat).
GPU resource/error-handling gaps (unchecked CUDA calls, leaked streams/events/tmp buffer, unsafe tmpMem sizing when free memory < 4GB)

Fixed in numeric.cu (line 347) onward (LUonDevice).
Ownership bug in preprocess failure path (freeing caller-owned SNicsLU*) + memory-management cleanup issues

Fixed in preprocess.c (line 102) onward.
CLI parse bug (-i missing value check off-by-one) + missing cleanup/return in main flow

Fixed in lu_cmd.cpp (line 43), lu_cmd.cpp (line 131).
Structural diagonal robustness for symbolic phase (prevents downstream invalid indexing assumptions)

Fixed in symbolic.cc (line 39).
 
