"""
Build the pyglu Python extension.

Requires:
  - HTHPCC SDK. Set HPCC_PATH if it is not installed under /opt/hpcc.
  - GCC / G++ (for NICSLU and GLU CPU code).
  - pybind11 (pip install pybind11).

Install (editable, recommended during development):
    pip install -e . --no-build-isolation

Install (regular):
    pip install . --no-build-isolation
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import pybind11

# ── Directory layout ────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.resolve()
SRC     = ROOT / "src"
NICSLU  = SRC / "nicslu"
PREPROC = SRC / "preprocess"
INCLUDE = ROOT / "include"

# HTHPCC compiler location documented in Runtime API guide section 4.1.1.
HPCC_PATH = os.environ.get("HPCC_PATH", "/opt/hpcc")
HTCC = os.path.join(HPCC_PATH, "htgpu_llvm", "bin", "htcc")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _run(*args, **kwargs):
    """Run a subprocess command and raise on failure."""
    subprocess.check_call(list(args), **kwargs)


# ── Custom build_ext ─────────────────────────────────────────────────────────
class GLUBuildExt(build_ext):
    """
    Custom builder that:
      1. Builds NICSLU static libraries (nicslu.a, nicslu_util.a) via make.
      2. Compiles CPU objects with gcc/g++ and the HTHPCC source with htcc.
      3. Links through htcc, as in the guide's HTHPCC Makefile example.
    """

    def build_extensions(self):
        # Step 1 — NICSLU static libraries
        # Rebuild the static CPU archives as PIC because they are linked into
        # the Python shared extension. The command-line CFLAGS value propagates
        # through the NICSLU recursive makefiles.
        _run("make", "-B", "lib", "util", cwd=str(NICSLU))

        # Step 2 — Compile GLU CPU/GPU translation units
        build_tmp = Path(self.build_temp)
        build_tmp.mkdir(parents=True, exist_ok=True)

        inc_flags = [
            f"-I{INCLUDE}",
            f"-I{NICSLU / 'include'}",
            f"-I{NICSLU / 'util'}",
            f"-I{PREPROC}",
        ]

        # preprocess.c
        preproc_o = build_tmp / "preprocess.o"
        _run("gcc", "-O2", "-Wall", "-msse2", "-fPIC",
             *inc_flags, "-DNO_ATOMIC", "-DSSE2",
             "-c", str(PREPROC / "preprocess.c"), "-o", str(preproc_o))

        # preprocess_arrays.c
        preproc_arr_o = build_tmp / "preprocess_arrays.o"
        _run("gcc", "-O2", "-Wall", "-msse2", "-fPIC",
             *inc_flags, "-DNO_ATOMIC", "-DSSE2",
             "-c", str(PREPROC / "preprocess_arrays.c"),
             "-o", str(preproc_arr_o))

        # symbolic.cc
        symbolic_o = build_tmp / "symbolic.o"
        _run("g++", "-O3", "-m64", "-std=c++11", "-Wall", "-fPIC",
             *inc_flags,
             "-c", str(SRC / "symbolic.cc"), "-o", str(symbolic_o))

        # Timer.cpp
        timer_o = build_tmp / "Timer.o"
        _run("g++", "-O3", "-m64", "-std=c++11", "-Wall", "-fPIC",
             *inc_flags,
             "-c", str(SRC / "Timer.cpp"), "-o", str(timer_o))

        # HTHPCC translation unit compiled in -x hpcc mode.
        numeric_o = build_tmp / "numeric.o"
        _run(HTCC, "-x", "hpcc", "-O3", "-std=c++11",
             f"-I{INCLUDE}",
             f"-I{NICSLU / 'include'}",
             f"-I{NICSLU / 'util'}",
             f"-I{PREPROC}",
             # PIC is needed in a Python shared extension, but the guide does
             # not document htcc's support for this host compiler flag.
             "-fPIC",
             "-c", str(SRC / "numeric.cu"), "-o", str(numeric_o))

        # Attach pre-built objects to the extension so they get linked in
        extra_objs = [
            str(preproc_o),
            str(preproc_arr_o),
            str(symbolic_o),
            str(timer_o),
            str(numeric_o),
            str(NICSLU / "lib" / "nicslu.a"),
            str(NICSLU / "util" / "nicslu_util.a"),
        ]
        for ext in self.extensions:
            ext.extra_objects = extra_objs

        # setuptools' Unix linker uses linker_so_cxx for this C++ extension.
        # Keep -shared while routing that shared link through htcc.
        self.compiler.set_executable(
            "linker_exe", [HTCC])
        self.compiler.set_executable(
            "linker_so", [HTCC, "-shared"])
        self.compiler.set_executable(
            "linker_so_cxx", [HTCC, "-shared"])
        # UnixCCompiler substitutes compiler_cxx when target_lang is C++;
        # align that driver with htcc while compiler_so_cxx still compiles the
        # binding with the host C++ compiler.
        self.compiler.compiler_cxx = [HTCC]

        super().build_extensions()


# ── Extension definition ─────────────────────────────────────────────────────
glu_ext = Extension(
    name="pyglu._pyglu",
    sources=[str(SRC / "glu_binding.cpp")],
    depends=[str(SRC / name) for name in
             ("numeric.cu", "left_looking.cuh", "left_looking_sf.cuh", "sf_sync.cuh")]
            + [str(INCLUDE / "numeric.h"), str(INCLUDE / "symbolic.h")],
    include_dirs=[
        str(INCLUDE),
        str(NICSLU / "include"),
        str(NICSLU / "util"),
        str(PREPROC),
        pybind11.get_include(),
    ],
    # CUDA include/library paths and cudart linkage removed for the HTHPCC path.
    libraries=["m", "rt", "pthread"],
    extra_compile_args=["-O3", "-std=c++11", "-Wall", "-fPIC"],
    language="c++",
)

setup(
    name="pyglu",
    version="0.1.0",
    packages=["pyglu"],
    ext_modules=[glu_ext],
    cmdclass={"build_ext": GLUBuildExt},
    python_requires=">=3.9",
    install_requires=["numpy>=1.22"],
    extras_require={"scipy": ["scipy>=1.9"]},
)
