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
```

`splu` accepts any scipy sparse matrix (any format) or a raw `(data, indices, indptr, shape)` tuple in CSC format. GLU's `REAL` type is `double`; the Python interface accepts and returns float64 arrays.

`update_strategy` defaults to `"right-looking"`; `"left-looking"` selects the
column-owned kernel in `src/left_looking.cuh`. The aliases `"right"`/`"rl"` and
`"left"`/`"ll"` are accepted. Both strategies use CPU symbolic fill-in prediction
and the same diagonal perturbation rule. `scheduling="level"` remains the default;
`scheduling="synchronization-free"` is supported only with left-looking updates.
Left-looking collects
previous L-column contributions into the current column before dividing by its
pivot; it does not broadcast updates into future columns. Rebuild the extension
after changing the native code.

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

## CLI residual target and iterative refinement

`lu_cmd` and `src/run_batch.py` automatically check the absolute residual
`||Ax-b||_2 <= 1e-6` after the initial solve. If needed, they reuse the existing
LU factors for correction solves: `r=b-Ax`, `delta_x=solve_CSR(r)`,
`x += delta_x`. Residuals use the original CSR matrix and CPU `long double`
accumulation (extra precision depends on the platform); factors and solves
remain double precision.

```sh
make -C src MAIN
./src/lu_cmd -i src/matrix/ASIC_680ks_csr.mtx --residual-tol 1e-6 --max-refine-iters 10
python3 src/run_batch.py --residual-tol 1e-6 --max-refine-iters 10
```

`--residual-tol` defaults to `1e-6` (finite and positive).
`--max-refine-iters` defaults to `n`, the matrix order; use an unsigned integer
for a fixed cap or `0` to check without correcting. No correction solve runs
when the initial result already meets the target. Refinement stops on success,
the iteration limit, a residual that fails to decrease, or nonfinite arithmetic.
Rejected updates leave the best solution intact. The solver still writes that
solution to `x.dat`, returning 0 on convergence or 2 if the target is missed.
The target is an absolute residual, not relative solution error.

Residual checks and refinement are included in total solve time. Logs include
initial/per-iteration residuals, the limit, attempted iteration count, final
status and a separate refinement time; `triangular_solve_time` covers the initial
solve only. Batch CSV retains 23 columns, records a missed target as
`residual_not_converged`, and continues. Residuals and solution files retain at
least 17 significant digits. The additional cost is CPU work and O(n) temporary
memory, without repeating GPU factorization. Automatic refinement applies to
the CLI/batch path; the Python `pyglu` API retains its existing behavior.
See [BUILD_AND_RUN.md](BUILD_AND_RUN.md) and
[src/RUN_BATCH.md](src/RUN_BATCH.md) for timing and stopping details.

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
 
