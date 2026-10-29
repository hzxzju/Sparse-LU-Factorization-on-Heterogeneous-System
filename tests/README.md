# Numeric strategy regression checks

Symbolic and Numeric now use a shared GPU selector: initialization errors on one
enumerated device are logged and skipped, and Numeric prefers the device selected
by Symbolic. Test its failure handling without a GPU:

```sh
make -C src test-gpu-device-selection DEBUG=0
make -C src gpu_device_probe DEBUG=0
timeout 15s ./src/gpu_device_probe --count
timeout 15s ./src/gpu_device_probe
```

Enumeration must succeed before per-device fallback can run. If the SDK itself
blocks in `hcGetDeviceCount`, changing `hcSetDevice` cannot skip that earlier
step. See [the initialization diagnosis](../docs/GPU_Device_Initialization_Diagnostics.md)
for the observed GPU 2 queue stall and the separate missing-device-node issue in
the restricted execution environment. The probe prints its current step before
calling the runtime; timeout exit 124 means the call did not return.

GPU symbolic Phase 1 has a separate target using the real HTHPCC toolchain:

```sh
make -C src symbolic_gpu_test DEBUG=0 -j2
make -C src test-symbolic-host DEBUG=0
timeout 300s ./src/symbolic_gpu_test
```

The minimal target links only CPU symbolic, GPU symbolic and the test driver;
it does not compile numeric kernels or preprocessing libraries. The default test
requires GPU execution with fallback disabled: exit 0 means the elementwise
checks passed, 1 means failure, and 77 means no accessible GPU (SKIP, not a pass).
`--host-only` runs 66,066 exhaustive host checks of the graph path criterion,
CPU dependency/ordering checks, input validation and reuse checks. It does not
validate GPU concurrency. GPU cases compare all three CSC arrays, test
64/128/256-thread CTAs, 1/3/128 workspace slots, epoch re-enqueue, and recursive
scan boundaries; subsequent CSR/value/level comparisons still use CPU methods.
The optional `symbolic_gpu_numeric_test` target adds existing numeric integration
checks. See [the semantic audit and Phase 1 instructions](../docs/GPU_Symbolic_Factorization_Design.md#phase-1-原型构建与验收).

Symbolic defaults to `--symbolic-backend auto`, selecting CPU or GPU-SF from
post-preprocessing column pointers. Explicit `cpu`, `gpu` and `gpu-sf` override it.
Test all four policy boundaries, fractional means, early/late long columns,
empty input and 64-bit costs without the GPU runtime:

```sh
make -C src test-symbolic-auto DEBUG=0
```

The existing `test-symbolic-host` target also executes the production auto CPU
dispatch, including reuse, diagnostics and inactive GPU options, without device
initialization. GPU symbolic SF can be selected explicitly (`--symbolic-backend gpu-sf`).
It reuses the CPU golden, packing scan and downstream numeric consumers. Run:

```sh
make -C src test-symbolic-sf-host DEBUG=0
make -C src test-symbolic-sf-gpu DEBUG=0
timeout -k 3s 600s ./src/symbolic_gpu_test --sf --stress
timeout -k 3s 600s ./src/symbolic_gpu_numeric_test --sf
python3 -m unittest discover -s tests -p test_symbolic_benchmark.py -v
```

The SF host target calls production scheduler/bitmap/checkpoint/waiter helpers
with host acquire/release atomics: 1,000 randomized executions, reversed task
issuance, many waiters, exact predecessor read counts, sparse arenas, guarded
bounds, and queue/ticket errors. Additional 64/128/256-lane host barrier tests
cover cooperative bitmap export and scans. These simulations do not prove GPU
device scope or native wave semantics. The real GPU target disables fallback
and tests 64/128/256-thread blocks, dynamic predecessors, structured/random
patterns, pool restarts, strict failure, reuse, and oversubscribed worker grids.
The numeric target compares factors and verifies ordinary and transpose solves.
The stress option adds 1,000 inputs with two worker configurations each.
See [RUN_BATCH.md](../src/RUN_BATCH.md#gpu-符号分析-sf2026-10-26) for the
protocol, memory/timing definitions, benchmark commands and actual acceptance
status; GPU exit 77 is a skip, never a pass.

The batch runner has a host-only regression using a subprocess solver fixture.
It checks stage timing and residual output, strategy/refinement option forwarding,
residual target failures and timeout handling, and the summary/log/solution files without requiring a GPU:

```sh
python3 -m unittest discover -s tests -p test_run_batch.py -v
```

The host-only refinement checks cover skipping an already accurate solution,
repeated corrections using inexact factors, absolute residual targets, iteration
caps (including zero), stagnation, rejecting worsening/nonfinite updates,
extended-precision residual cancellation, and retaining sub-ULP corrections in
a `long double` solution. Use a C++11 compiler (on the
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

From the repository root, run the public Python parameter and build recipe checks:

```sh
python -m unittest discover -s tests -p test_python_parameters.py -v
python -m unittest discover -s tests -p test_extension_build.py -v
```

The forwarding checks mock the native factorization; if NumPy is absent they
also mock the two array conversion calls. They verify option/default forwarding
and keyword-only compatibility, not NumPy dtype/ABI or native compilation. The
build recipe check mocks subprocesses and verifies that symbolic and numeric
GPU objects use the same SDK flags and are both linked into the extension.

Native Python checks require Python >=3.9, NumPy, pybind11 and a built extension:

```sh
python -m pip install -e . --no-build-isolation
python -m unittest discover -s tests -p test_native_symbolic.py -v
GLU_TEST_GPU=1 python -m unittest discover -s tests -p test_native_symbolic.py -v
```

Without `GLU_TEST_GPU=1`, only parameter/CSC rejection tests run; they reject
before preprocessing or device initialization. Hardware tests cover all three
symbolic backends, repeated solves and persistent diagnostics, strict pool
failure cleanup, and reporting the actual CPU backend after fallback. Missing
dependencies are explicit skips. Once hardware execution is enabled, a runtime
failure is a test failure, not a skip.

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
