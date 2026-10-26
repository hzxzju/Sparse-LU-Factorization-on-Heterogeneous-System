"""Check extension build orchestration without SDK/Python development headers."""
import os
from pathlib import Path
import runpy
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class ExtensionBuildTests(unittest.TestCase):
    def test_symbolic_gpu_is_compiled_and_linked_with_the_numeric_runtime(self):
        setuptools = types.ModuleType("setuptools")
        setuptools.Extension = lambda **kwargs: types.SimpleNamespace(**kwargs)
        setuptools.setup = Mock()
        build_module = types.ModuleType("setuptools.command.build_ext")

        class BuildExt:
            def build_extensions(self):
                pass

        build_module.build_ext = BuildExt
        pybind = types.ModuleType("pybind11")
        pybind.get_include = lambda: "/test/pybind11/include"
        modules = {"setuptools": setuptools, "setuptools.command": types.ModuleType("setuptools.command"),
                   "setuptools.command.build_ext": build_module, "pybind11": pybind}
        env = dict(GCC_TOOLCHAIN="/test/gcc", GCC_VERSION="7.3.0", HPCC_PATH="/test/hpcc",
                   CC="/test/gcc/bin/gcc", CXX="/test/gcc/bin/g++", HTCC="/test/hpcc/bin/htcc")
        with patch.dict("sys.modules", modules), patch.dict(os.environ, env):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
        builder = namespace["GLUBuildExt"]()
        builder.extensions = [namespace["glu_ext"]]
        builder.compiler = types.SimpleNamespace(set_executable=Mock())
        # Patch the function's globals: runpy returns a separate dictionary.
        commands = Mock()
        with tempfile.TemporaryDirectory(prefix="glu-build-contract-") as directory:
            builder.build_temp = directory
            with patch.dict(builder.build_extensions.__globals__, {"_run": commands}):
                builder.build_extensions()
            for source in ("numeric.cu", "symbolic_gpu.cu"):
                matches = [call[0] for call in commands.call_args_list
                           if any(str(arg).endswith("/" + source) for arg in call[0])]
                self.assertEqual(len(matches), 1, source)
                command = matches[0]
                self.assertEqual(command[0], env["HTCC"])
                for flag in ("-fPIC", "-D__sync_threads=__syncthreads", "-gcc-toolchain=/test/gcc"):
                    self.assertIn(flag, command)
            self.assertIn(str(Path(directory) / "symbolic_gpu.o"), builder.extensions[0].extra_objects)
            self.assertIn(str(Path(directory) / "numeric.o"), builder.extensions[0].extra_objects)
        dependencies = {Path(path).name for path in namespace["glu_ext"].depends}
        self.assertTrue({"symbolic_gpu.cu", "symbolic_sf.cuh", "symbolic.h", "gpu_device.h"} <= dependencies)
        self.assertEqual(builder.compiler.compiler_cxx[0], env["HTCC"])
        self.assertEqual(setuptools.setup.call_args[1]["python_requires"], ">=3.9")


if __name__ == "__main__":
    unittest.main()
