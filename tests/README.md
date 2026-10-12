# Numeric strategy regression checks

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
whether the target Clang/LLVM device compiler supports the production
`__atomic_*` adapter; the runtime guide alone does not establish that support.

```sh
"$HPCC_PATH/htgpu_llvm/bin/htcc" -O2 -std=c++11 -x hpcc -Isrc tests/sf_sync.cu -o /tmp/glu-sf-sync-test
/tmp/glu-sf-sync-test
```

It tests all-lane publication/consumption through the production SF helpers,
monotone issuance with 1/2/8/128/1024 workers, and four resets with different
data per worker count. The 257-task chain includes more workers than tasks.

Then run the numeric driver:

```sh
g++ -O2 -std=c++11 -Iinclude -Isrc/nicslu/include -c src/symbolic.cc -o /tmp/glu-symbolic-test.o
g++ -O2 -std=c++11 -Iinclude -c src/Timer.cpp -o /tmp/glu-timer-test.o
"$HPCC_PATH/htgpu_llvm/bin/htcc" -O2 -std=c++11 -x hpcc -Iinclude -Isrc/nicslu/include -c src/numeric.cu -o /tmp/glu-numeric-test.o
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
