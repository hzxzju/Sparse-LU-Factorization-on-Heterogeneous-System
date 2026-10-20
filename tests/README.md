# Numeric strategy regression checks

The batch runner has a host-only regression using a subprocess solver fixture.
It checks stage timing and residual output, strategy/refinement option forwarding,
residual target failures and timeout handling, and the summary/log/solution files without requiring a GPU:

```sh
python3 -m unittest discover -s tests -p test_run_batch.py -v
```

The host-only refinement checks cover skipping an already accurate solution,
repeated corrections using inexact factors, absolute residual targets, iteration
caps (including zero), stagnation, rejecting worsening/nonfinite updates,
and extended-precision residual cancellation. Use a C++11 compiler (on the
HTHPCC host, the installed compiler is `$HOME/local/gcc-7.3.0/usr/bin/g++`):

```sh
"${CXX:-$HOME/local/gcc-7.3.0/usr/bin/g++}" -O2 -Wall -Wextra -std=c++11 -Iinclude tests/refinement.cpp -o /tmp/glu-refinement-test
/tmp/glu-refinement-test
```

For a target-GPU integration check on the two problematic matrices, build and
run each solver invocation from a separate output directory (each writes
`x.dat`). With default arguments, the CLI targets absolute residualL2 <= 1e-6
and uses the matrix order as the correction cap:

```sh
make -C src MAIN
GLU_TEST_ROOT="$PWD"
GLU_TEST_RESULTS=$(mktemp -d /tmp/glu-refinement-check.XXXXXX)
mkdir "$GLU_TEST_RESULTS/asic" "$GLU_TEST_RESULTS/twotone"
(cd "$GLU_TEST_RESULTS/asic" && "$GLU_TEST_ROOT/src/lu_cmd" -i "$GLU_TEST_ROOT/src/matrix/ASIC_680ks_csr.mtx")
(cd "$GLU_TEST_RESULTS/twotone" && "$GLU_TEST_ROOT/src/lu_cmd" -i "$GLU_TEST_ROOT/src/matrix/twotone_csr.mtx")
```

Inspect `Refinement iterations`, `Refinement status`, final `Ax-b (2-norm)`
and the exit code: 0 means convergence, 2 means a missed target with the best
solution retained. `--max-refine-iters 0` provides an initial-solve comparison;
`--max-refine-iters 1` checks the single-correction cap. The refinement time
includes the initial residual check and all correction work, and is included in
`T_gpu`. A stagnating or nonfinite candidate is rejected and stops refinement.
See [../src/RUN_BATCH.md](../src/RUN_BATCH.md) for all options and CSV status rules.

From the repository root, test the native Matrix Market reader (requires GCC
and Python, no GPU). This covers arbitrary triplet order, empty rows, invalid
coordinates, and the bundled column-ordered `ASIC_100k.mtx`:

```sh
python3 -m unittest discover -s tests -p test_matrix_reader.py -v
```

From `PublicCase/GLU_public`, run the public Python parameter checks:

```sh
python -m unittest discover -s tests -p test_python_parameters.py -v
```

On the HTHPCC machine, compile the actual numeric and symbolic sources with the
GPU regression driver:

First compile and run the ordered-atomic publication test. This also probes
whether the target SDK and GPU implement the production device-scope
`__hpcc_atomic_*` publication protocol correctly. Compilation alone cannot
validate cross-worker visibility or progress.

```sh
"$HPCC_PATH/htgpu_llvm/bin/htcc" -O2 -std=c++11 -x hpcc -D__sync_threads=__syncthreads -Isrc tests/sf_sync.cu -o /tmp/glu-sf-sync-test
timeout 60s /tmp/glu-sf-sync-test
```

It tests all-lane publication/consumption through the production SF helpers,
monotone issuance with 1/2/8/128/1024 workers, and four resets with different
data per worker count. The 257-task chain includes more workers than tasks. Each launch prints and
flushes its worker count and repeat number before starting, so a timeout shows
which configuration stalled. All five worker counts must pass; a pass with only
one worker does not validate inter-block publication.

Then run the numeric driver:

```sh
g++ -O2 -std=c++11 -Iinclude -Isrc/nicslu/include -c src/symbolic.cc -o /tmp/glu-symbolic-test.o
g++ -O2 -std=c++11 -Iinclude -c src/Timer.cpp -o /tmp/glu-timer-test.o
"$HPCC_PATH/htgpu_llvm/bin/htcc" -O2 -std=c++11 -x hpcc -D__sync_threads=__syncthreads -Iinclude -Isrc/nicslu/include -c src/numeric.cu -o /tmp/glu-numeric-test.o
g++ -O2 -std=c++11 -Iinclude -Isrc/nicslu/include -c tests/numeric_strategies.cpp -o /tmp/glu-strategies-test.o
"$HPCC_PATH/htgpu_llvm/bin/htcc" /tmp/glu-symbolic-test.o /tmp/glu-numeric-test.o /tmp/glu-strategies-test.o /tmp/glu-timer-test.o -o /tmp/glu-strategies-test
/tmp/glu-strategies-test
```

The driver checks RL-level, LL-level and LL-SF (one/automatic/three workers)
against an independent CPU factorization, reconstructs `L*U`, and checks solve
residuals across 17 matrices. SF cases deliberately omit level and CSR generation
and refactor the same CPU structure with two additional sets of scaled values.
Cases cover shared successors, fill-in, missing
structural diagonals, empty L columns, perturbed zero/negative small pivots,
workspace reuse, RL scheduling thresholds, and columns longer than one block.

For NVIDIA-only development machines, `cuda_compat/hc_runtime.h` maps the runtime
calls and block barriers to CUDA **for tests only**. Compile the same sources
with nvcc, adding `-Itests/cuda_compat` and `-arch=sm_70` or newer. The SF
test adapter uses device-scope PTX acquire loads/release stores; those require
sm_70+. Compile `tests/sf_sync.cu` with `-Isrc` as well.
The numeric backend now uses the POSIX `Timer` host implementation; these
regression build commands require Linux (including WSL).
This checks actual GPU arithmetic but does not validate the HTHPCC SDK build or
performance on its target hardware.
