#!/bin/bash
set -e

export HPCC_PATH=${HPCC_PATH:-/opt/hpcc}
export PATH=$HPCC_PATH/htgpu_llvm/bin:$PATH
# Runtime library path follows the native offline-build instructions.
export LD_LIBRARY_PATH=$HPCC_PATH/lib:$LD_LIBRARY_PATH
export CC=gcc CXX=g++

make MAIN
