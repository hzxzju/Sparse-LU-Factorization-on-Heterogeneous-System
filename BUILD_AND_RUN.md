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

## Residual target and iterative refinement

After the initial solve, `lu_cmd` checks the absolute residual
`||Ax-b||_2 <= 1e-6`. If needed, it computes `r=b-Ax` from the original CSR
matrix, reuses `A_sym.solve_CSR` to solve for the correction, and updates
`x += delta_x`. Permutations and scaling are handled by the same solve routine;
the GPU factors are not recomputed.

```sh
./src/lu_cmd -i src/matrix/ASIC_680ks_csr.mtx --residual-tol 1e-6 --max-refine-iters 10
python3 src/run_batch.py --residual-tol 1e-6 --max-refine-iters 10
```

`--residual-tol` defaults to `1e-6` and must be finite and positive.
`--max-refine-iters` defaults to `n` (the matrix order); an unsigned integer
sets a fixed correction limit, and `0` disables corrections while retaining the
residual check. An initially accurate solution needs no correction solves.
Residuals use CPU `long double` accumulation, whose extra precision depends on
the platform; factors, solutions and triangular solves remain double precision.
Each accepted correction must lower residualL2. Stagnation or nonfinite arithmetic
stops refinement early and preserves the best solution. Reaching the iteration
limit without meeting the target is also a failure. The solver writes the best
`x.dat` and returns **2** on a missed target, **0** on convergence.
Solution files use scientific notation with at least 17 significant digits.

Logs record the tolerance, resolved iteration limit, initial and per-iteration
residuals, attempted correction count, status (`converged`, `max_iterations`,
`stagnated`, `nonfinite`) and `Iterative refinement time (CPU)`.
Batch CSV remains at 23 columns; these diagnostics stay in `solver.log` and the
extra time is included in `total_solve_ms`. Batch `status=ok` also requires the
specified residual target; `residual_not_converged` records a missed target and
the batch continues. A small residual does not guarantee a small solution error.
Automatic refinement applies to the CLI/batch path, not the Python `pyglu` API.
Static pivot perturbation (`-p`) does not guarantee improved residuals.

## Task 1 timing

The contest guide (pages 6-7 and 9) scores speedup as
`SR_i,k = T_cpu,k / T_gpu,i,k`. `T_gpu` is the complete solve **wall-clock**
time, including CPU preparation, GPU setup, transfers, computation,
synchronization and result return. The CLI now prints these times in milliseconds:

| Output | Measured scope |
| --- | --- |
| `Input/preprocessing time` | The entire `preprocess` call: file reading, matrix backup/construction, NICSLU ordering/scaling and extraction of the prepared matrix. |
| `Symbolic time` | Symbolic matrix construction and fill-in analysis. |
| `CSR time`, `PredictLU time`, `Leveling time` | CSR generation, numeric value preparation and level scheduling; leveling is exactly zero for SF. |
| `Symbolic preparation time (total)` | All four CPU preparation stages above, measured as one interval. |
| `Numeric LU compute time (wall-clock)` | GPU numeric dispatch through completion of all kernels, including launches and synchronization across levels/streams; excludes allocation, initialization and transfers. |
| `Numeric LU time (including GPU setup/transfers)` | The entire `LUonDevice` call, including device setup, allocations, uploads, compute, synchronization, factor download and GPU cleanup. |
| `Triangular solve time (CPU)` | Initial `solve_CSR` only, including RHS permutation/scaling, forward/back substitution and restoring the original variable order. |
| `Iterative refinement time (CPU)` | Initial residual check and all correction solves, updates and candidate residual checks. |
| `Task 1 total solve time (T_gpu)` | From before NICSLU context allocation and input preparation until the final refined solution is available on the CPU. Use this value as the speedup denominator. |

All new measurements use `Timer::elapsedWallclockTime` with a monotonic
`std::chrono::steady_clock`. GPU initialization completes before the compute
timer starts, and every numeric path synchronizes kernel completion before it
stops. CPU user time does not account for GPU waits and is no longer used for
the CLI's symbolic timings. By default, solver diagnostics and timing summaries are printed
after the total timer stops. With `--progress`, diagnostics and stage markers
are flushed during execution (without per-kernel measurement); the existing C preprocessing progress messages
remain inside the preprocessing interval.

The total timer includes RHS creation, the CLI's file input, the initial residual
check and all refinement work; it stops before writing `x.dat`, reporting the
residuals and final CPU cleanup. The contest's
official interface/timing rules will be released with its cases (pages 9-10);
use the same input boundary for CPU comparisons. Substage times are nested, so
do not sum every printed line or substitute the legacy event-based `Total GPU
time` for `T_gpu`.

For a matching CPU baseline in milliseconds, compute
`SR = cpu_ms / total_solve_ms`. Per-matrix performance scoring is
`S2_i,k = 25 * min(SR_i,k / SR_best,k, 1)` for correct solutions, and zero for
incorrect solutions; the final S2 is the average across matrices (pages 7-8).
Timing alone does not establish correctness: S1 requires relative solution
error `||x_gpu - x_ref||_2 / ||x_ref||_2 <= 1e-6` against a reference solution.
The existing `Ax-b` residual output is a separate check.

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
