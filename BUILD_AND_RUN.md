# GLU build and run

## Requirements

- Linux, GCC/G++ with C++11 support, GNU make, pthreads, libm and librt.
- The HTHPCC SDK installed under `/opt/hpcc`, or `HPCC_PATH` set to its
  installation directory. The runtime API guide's offline Makefile example
  uses `${HPCC_PATH}/htgpu_llvm/bin/htcc`, `-x hpcc`, and
  `${HPCC_PATH}/lib` in `LD_LIBRARY_PATH` (section 4.1.1, printed pages 49-51).

## Build

Run the commands below from the repository root
(`Sparse-LU-Factorization-on-Heterogeneous-System/`):

```sh
export HPCC_PATH=/opt/hpcc  # adjust if installed elsewhere
export LD_LIBRARY_PATH="$HPCC_PATH/lib:${LD_LIBRARY_PATH:-}"
make -C src clean
make -C src MAIN
```

The root Makefile also supports `make` (or `make MAIN`), `make clean` and
`make distclean`, forwarding these commands to `src/`.

The CPU preprocessing, symbolic and NICSLU sources remain on the GCC/G++ build
path. The GPU translation unit and final executable use `htcc`; its HTHPCC
language mode is marked in `src/Makefile`. `src/run.sh` sets the documented
SDK compiler and runtime paths before invoking the same Makefile target.

## Run

From the repository root, use the executable in `src/` and the bundled matrix
in `src/nicslu/demo/`:

```sh
./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx
./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx --update-strategy left-looking
./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx -u left-looking -p
```

If your terminal is already in `src/`, the corresponding command is:

```sh
./lu_cmd -i nicslu/demo/ASIC_100k.mtx
```

`-i` selects a square Matrix Market `coordinate real general` input containing
row, column and value triplets. The reader groups entries into CSR; the input
does not need to be sorted by row. The right-hand side is all ones; the solution
is written to `x.dat` in the terminal's current directory and the residual norms
are printed. To use your own matrix, replace the example path with its actual
path relative to the current directory, or an absolute path.

`-u` / `--update-strategy` selects `right-looking` (default) or `left-looking`.
Aliases `right`/`rl` and `left`/`ll` are accepted. `-p` enables the same static
diagonal perturbation for either strategy. The selected strategy is printed by
the GPU backend.

For CPU symbolic factorization and synchronization-free GPU numeric LL, run
from the repository root:

```sh
./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx -u left-looking --scheduling synchronization-free
```

`-s` / `--scheduling` accepts `level` (default) or `synchronization-free`.
SF requires LL and skips CPU level generation and GPU level/CSR uploads.
The CPU still generates fill-in, symbolic CSC/CSR and initial numeric values.

## Solution error and iterative refinement

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
relative GPU/CPU L2 error `||x_GPU-x_CPU||₂ / ||x_CPU||₂` is at most this
tolerance; 2 means it exceeds the target. Absolute L2 error is reported separately.
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

## Task 1 timing

The CLI uses `clock_gettime(CLOCK_MONOTONIC, ...)` for stage wall times.
After host/device cleanup, stdout contains exactly one timing line in seconds,
with six decimal places:

```text
once_s=7.767300 solve_s=0.555000 post_s=0.020000
```

The batch CSV and terminal `total_solve_ms` are the weighted result:
`(0.005 * once_s + solve_s) * 1000`. For the example above this is
`593.8365 ms`. Postprocessing is excluded from the score. The rounded stdout
values are authoritative for this calculation; historical unweighted
`Task 1 total solve time (T_gpu)` logs are not used.

| Aggregate | Exclusive stage fields (seconds, stderr labels end in `_s=`) |
| --- | --- |
| `once_s` | `read_parse`, `rhs_read`, `preprocess`, `ordering`, `symbolic`, `analysis_other`, `context_init`, `workspace_alloc`, `kernel_compile`, `plan_upload`, `cleanup` |
| `solve_s` | `value_h2d`, `numeric_factor`, `triangular_solve`, `solution_d2h`, `refinement`, `sync` |
| `post_s` | `residual_check`, `output_write`, `logging` |

`read_parse` reads/parses the matrix; `rhs_read` generates the all-ones RHS.
NICSLU MC64 matching/scaling and value packing belong to `preprocess`; AMD
ordering belongs to `ordering`. CSR construction, level scheduling and backend
selection belong to `analysis_other`. Device selection/probing belongs to
`context_init`, device buffer allocation/initialization to `workspace_alloc`,
and structural index/scheduling uploads to `plan_upload`. GPU symbolic work,
including structural result downloads, belongs to the once-only analysis.
There is no explicit runtime compiler in this ahead-of-time build, so
`kernel_compile` is zero.

`value_h2d` only uploads numeric values. `numeric_factor` only covers numeric
LU dispatch/computation and its completion waits. Device stages synchronize
before their timer stops. Waits already included in stages are not charged a
second time to `sync`, which is zero on the current paths. Nested phases pause
the parent counter rather than double counting it.

Triangular substitution runs on the CPU: `triangular_solve` covers the initial
forward/back substitutions, while host-path preparation (factor download,
permutation/scaling and solution conversion) is charged to `refinement`.
`factor_d2h_s` is a diagnostic subset of `refinement`, not an extra charge or a
solution-vector transfer. `solution_d2h` is zero because the CPU produces the
solution. Correction solves, updates and residuals needed to choose/stop
refinement all remain in `refinement`. Independent final residual verification
belongs to `residual_check`, and writing/closing `x.dat` to `output_write`.
GPU resource destruction and final NICSLU/host container destruction belong to
`cleanup`, even though they occur at different points in the execution.

Diagnostics go to stderr (batch merges them into `solver.log`). They include
all exclusive stage counters, `factor_d2h_s`, and weighted `total_solve_ms`.
Legacy millisecond stage descriptions remain for comparison; those inclusive
intervals overlap and must not be added to the exclusive counters. In
particular, `Numeric LU time (including GPU setup/transfers)` spans both once
and solve work. `Total GPU time` now also uses the monotonic host clock.
`--progress` adds stage markers on stderr. Diagnostic writes/flushes are
charged to `logging`; final timing-report serialization is excluded from its
own measurement to avoid a self-referential timing report.

Timing alone does not establish correctness. The final `Ax-b` residual and
convergence status are checked separately from the weighted score.

## Python extension

The `setup.py` build uses `HPCC_PATH` and `htcc` for the HTHPCC translation
unit and extension link step. Install Python dependencies, then run from the
repository root:

```sh
pip install pybind11
pip install -e . --no-build-isolation
```

After rebuilding the extension, select the strategy in either Python entry point:

```python
lu = pyglu.splu(A, update_strategy="left-looking")
x = lu.solve(b)
x = pyglu.spsolve(A, b, update_strategy="left-looking", perturb=True)
lu = pyglu.splu(A, update_strategy="left-looking",
               scheduling="synchronization-free")
```

The implementation is split between `src/left_looking.cuh` (column update and
factorization), `src/numeric.cu` (strategy dispatch and level/batch scheduling),
and `include/numeric.h` (C++ strategy parameter and parser).

`src/left_looking_sf.cuh` adds monotone column issuance and per-predecessor
waiting, sharing the LL arithmetic helpers with the level path. `src/sf_sync.cuh`
contains the ordered-atomic adapter. The native adapter uses the installed SDK
`__hpcc_atomic_*` builtins with explicit `memory_scope_device` (relaxed task
issuance, acquire completion loads, release completion stores). The compiler
must provide these builtins. Compile and run the publication test in
`tests/README.md` on the target GPU to validate progress and visibility. There is
no relaxed/volatile fallback.

The C++ `LUonDevice` entry point also accepts an optional final `sf_workers`
argument (0 = automatic), used by the GPU regressions to exercise one and three
workers. Requested counts are bounded by matrix size, memory budget and device
grid limit, and may be reduced on allocation failure. Python/CLI use automatic
selection by default. CLI/batch `--sf-workers N` exposes this limit for diagnosis,
and `--progress` flushes live stage diagnostics. Disable progress output for
performance comparisons because it adds logging overhead. `Total GPU time` still includes setup/transfers; it is not a standalone
numeric-kernel benchmark.

The solver reports stage timings and residuals without per-kernel profiling.
All RL, LL and LL-SF kernels launch directly; the code does not query occupancy,
kernel resource attributes or CE/RM counters, or write per-launch records.
The `--profile-kernels` and `--no-profile-kernels` options have been removed.

For batch stage timing results, rebuild and run from the repository root:

```sh
make -C src MAIN
python3 src/run_batch.py --update-strategy left-looking \
  --scheduling synchronization-free --timeout 600
```

Each run writes a 23-column `summary.csv`, per-matrix logs and solution files.
See [src/RUN_BATCH.md](src/RUN_BATCH.md) for timing boundaries and options.
No new kernel detail CSV or JSONL is produced; existing historical results are
preserved and may include profiling overhead. The CLI still reports aggregate
numeric compute wall time in addition to the full numeric-call time.

The HTHPCC guide documents executable Makefile builds, but does not specify
Python shared-extension linking, runtime library names, or whether `htcc`
accepts the existing `.cu` filename. Confirm these details against the
installed SDK before relying on the Python extension build. The native C++
Makefile also passes `numeric.cu` in `-x hpcc` mode; this filename/mode pairing
is not explicitly covered by the guide's `.cpp` example. The guide also does
not document whether `htcc` accepts `-fPIC`, which the extension build needs
for position-independent host code.
