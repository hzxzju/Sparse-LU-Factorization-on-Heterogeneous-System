# GLU build and run

## Requirements

- Linux, GCC/G++ with C++11 support, GNU make, pthreads, libm and librt.
- The HTHPCC SDK installed under `/opt/hpcc`, or `HPCC_PATH` set to its
  installation directory. The runtime API guide's offline Makefile example
  uses `${HPCC_PATH}/htgpu_llvm/bin/htcc`, `-x hpcc`, and
  `${HPCC_PATH}/lib` in `LD_LIBRARY_PATH` (section 4.1.1, printed pages 49-51).

## Build

From `PublicCase/GLU_public/src`:

```sh
export HPCC_PATH=/opt/hpcc  # adjust if installed elsewhere
export LD_LIBRARY_PATH="$HPCC_PATH/lib:$LD_LIBRARY_PATH"
make clean
make MAIN
```

The CPU preprocessing, symbolic and NICSLU sources remain on the GCC/G++ build
path. The GPU translation unit and final executable use `htcc`; its HTHPCC
language mode is marked in `src/Makefile`. `src/run.sh` sets the documented
SDK compiler and runtime paths before invoking the same Makefile target.

## Run

```sh
./lu_cmd -i matrix/add32_csr.mtx
./lu_cmd -i matrix/add32_csr.mtx --update-strategy left-looking
./lu_cmd -i matrix/add32_csr.mtx -u left-looking -p
```

`-i` selects a sorted CSR Matrix Market input. The right-hand side is all ones;
the solution is written to `x.dat` and the residual norms are printed.

`-u` / `--update-strategy` selects `right-looking` (default) or `left-looking`.
Aliases `right`/`rl` and `left`/`ll` are accepted. `-p` enables the same static
diagonal perturbation for either strategy. The selected strategy is printed by
the GPU backend.

For CPU symbolic factorization and synchronization-free GPU numeric LL:

```sh
./lu_cmd -i inputfile -u left-looking --scheduling synchronization-free
```

`-s` / `--scheduling` accepts `level` (default) or `synchronization-free`.
SF requires LL and skips CPU level generation and GPU level/CSR uploads.
The CPU still generates fill-in, symbolic CSC/CSR and initial numeric values.

## Python extension

The `setup.py` build uses `HPCC_PATH` and `htcc` for the HTHPCC translation
unit and extension link step. Install Python dependencies, then run:

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
contains the ordered-atomic adapter. The native adapter uses Clang lock-free
unsigned `__atomic_*` builtins (relaxed task issuance, acquire completion loads,
release completion stores). HTHPCC device lowering is not documented in the
supplied runtime guide: compile and run the publication test in `tests/README.md`
on the installed SDK before deployment. There is no relaxed/volatile fallback.

The C++ `LUonDevice` entry point also accepts an optional final `sf_workers`
argument (0 = automatic), used by the GPU regressions to exercise one and three
workers. Requested counts are bounded by matrix size, memory budget and device
grid limit, and may be reduced on allocation failure. Python/CLI use automatic
selection. `Total GPU time` still includes setup/transfers; it is not a standalone
numeric-kernel benchmark.

The HTHPCC guide documents executable Makefile builds, but does not specify
Python shared-extension linking, runtime library names, or whether `htcc`
accepts the existing `.cu` filename. Confirm these details against the
installed SDK before relying on the Python extension build. The native C++
Makefile also passes `numeric.cu` in `-x hpcc` mode; this filename/mode pairing
is not explicitly covered by the guide's `.cpp` example. The guide also does
not document whether `htcc` accepts `-fPIC`, which the extension build needs
for position-independent host code.
