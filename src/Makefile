# CPU and GPU builds use the same user-installed GCC toolchain.
DEBUG ?= 0
GCC_TOOLCHAIN ?= $(HOME)/local/gcc-7.3.0/usr
GCC_VERSION ?= 7.3.0
HPCC_PATH ?= /opt/hpcc
CC = $(GCC_TOOLCHAIN)/bin/gcc
CXX = $(GCC_TOOLCHAIN)/bin/g++
HTCC ?= $(HPCC_PATH)/htgpu_llvm/bin/htcc
export GCC_TOOLCHAIN GCC_VERSION

EXE = lu_cmd
OBJS = Timer.o numeric.o symbolic.o symbolic_gpu.o lu_cmd.o
PREPROCESS_LIBS = preprocess/preprocess.o nicslu/lib/nicslu.a nicslu/util/nicslu_util.a
LU_INC = -Ipreprocess -I../include -Inicslu/include
CPPFLAGS += $(LU_INC) -DGLU_DEBUG=$(DEBUG)
CXXFLAGS ?= -O3 -Wall -std=c++11
HTCCFLAGS ?= -O3 -std=c++11 -x hpcc
HTCC_GCC_FLAGS = -gcc-toolchain=$(GCC_TOOLCHAIN) -gcc-version=$(GCC_VERSION)
# This SDK spells the block synchronization intrinsic __syncthreads.
HTCC_CPPFLAGS = -D__sync_threads=__syncthreads
LDFLAGS += -Wl,-rpath,$(GCC_TOOLCHAIN)/lib64 -Wl,-rpath,$(HPCC_PATH)/lib
LDLIBS += -lpthread -lm -lrt -ldl

.PHONY: MAIN all clean distclean test-symbolic-gpu test-symbolic-host test-symbolic-auto test-symbolic-sf-gpu test-symbolic-sf-host test-gpu-device-selection
.DELETE_ON_ERROR:
MAIN: $(EXE)
all: MAIN

$(EXE): $(OBJS) $(PREPROCESS_LIBS) Makefile
	$(HTCC) $(HTCC_GCC_FLAGS) $(LDFLAGS) $(OBJS) $(PREPROCESS_LIBS) $(LDLIBS) -o $@

Timer.o: Timer.cpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

numeric.o: numeric.cu left_looking.cuh left_looking_sf.cuh sf_sync.cuh Makefile
	$(HTCC) $(CPPFLAGS) $(HTCC_CPPFLAGS) $(HTCCFLAGS) $(HTCC_GCC_FLAGS) -MMD -MP -c $< -o $@

symbolic.o: symbolic.cc Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

symbolic_gpu.o: symbolic_gpu.cu symbolic_sf.cuh sf_sync.cuh ../include/symbolic.h ../include/gpu_device.h Makefile
	$(HTCC) $(CPPFLAGS) $(HTCC_CPPFLAGS) $(HTCCFLAGS) $(HTCC_GCC_FLAGS) -MMD -MP -c $< -o $@

# Both GPU symbolic backends build independently of numeric kernels and preprocessing libraries.
symbolic_gpu_test: symbolic.o symbolic_gpu.o test_symbolic_gpu.o Makefile
	$(HTCC) $(HTCC_GCC_FLAGS) $(LDFLAGS) $(filter %.o,$^) $(LDLIBS) -o $@

test_symbolic_gpu.o: ../tests/symbolic_gpu.cpp Makefile
	$(HTCC) $(CPPFLAGS) $(HTCC_CPPFLAGS) $(HTCCFLAGS) $(HTCC_GCC_FLAGS) -MMD -MP -c $< -o $@

# Optional downstream numeric regression for either symbolic GPU backend.
symbolic_gpu_numeric_test: Timer.o numeric.o symbolic.o symbolic_gpu.o test_symbolic_gpu_numeric.o Makefile
	$(HTCC) $(HTCC_GCC_FLAGS) $(LDFLAGS) $(filter %.o,$^) $(LDLIBS) -o $@

test_symbolic_gpu_numeric.o: ../tests/symbolic_gpu.cpp Makefile
	$(HTCC) $(CPPFLAGS) $(HTCC_CPPFLAGS) $(HTCCFLAGS) $(HTCC_GCC_FLAGS) -DGLU_SYMBOLIC_TEST_NUMERIC=1 -MMD -MP -c $< -o $@

# Exit 77 (no accessible GPU) intentionally fails the GPU acceptance target.
test-symbolic-gpu: symbolic_gpu_test
	./symbolic_gpu_test

test-symbolic-sf-gpu: symbolic_gpu_test
	./symbolic_gpu_test --sf

test-symbolic-host: symbolic_gpu_test
	./symbolic_gpu_test --host-only

symbolic_auto_test: ../tests/symbolic_auto.cpp ../include/symbolic.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) ../tests/symbolic_auto.cpp -o $@

test-symbolic-auto: symbolic_auto_test
	./symbolic_auto_test

symbolic_sf_host_test: symbolic.o ../tests/symbolic_sf_host.cpp symbolic_sf.cuh ../include/symbolic.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread ../tests/symbolic_sf_host.cpp symbolic.o -o $@

test-symbolic-sf-host: symbolic.o symbolic_sf_host_test
	./symbolic_sf_host_test

gpu_device_selection_test: ../tests/gpu_device_selection.cpp ../include/gpu_device.h Makefile
	$(HTCC) $(CPPFLAGS) $(HTCCFLAGS) $(HTCC_GCC_FLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

test-gpu-device-selection: gpu_device_selection_test
	./gpu_device_selection_test

gpu_device_probe: ../tests/gpu_device_probe.cpp ../include/gpu_device.h Makefile
	$(HTCC) $(CPPFLAGS) $(HTCCFLAGS) $(HTCC_GCC_FLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

lu_cmd.o: lu_cmd.cpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

preprocess/preprocess.o: $(wildcard preprocess/*.c preprocess/*.h nicslu/include/*.h nicslu/util/*.h) preprocess/Makefile nicslu/make.inc
	$(MAKE) -C preprocess

nicslu/lib/nicslu.a: $(wildcard nicslu/source/*.c nicslu/source/*.h nicslu/include/*.h) nicslu/lib/Makefile nicslu/make.inc
	$(MAKE) -C nicslu lib

nicslu/util/nicslu_util.a: $(wildcard nicslu/util/*.c nicslu/util/*.h nicslu/include/*.h) nicslu/util/Makefile nicslu/make.inc
	$(MAKE) -C nicslu util

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(EXE) test_symbolic_gpu.o test_symbolic_gpu.d symbolic_gpu_test test_symbolic_gpu_numeric.o test_symbolic_gpu_numeric.d symbolic_gpu_numeric_test gpu_device_selection_test gpu_device_probe symbolic_sf_host_test symbolic_auto_test
	$(MAKE) -C preprocess clean
	$(MAKE) -C nicslu/lib clean
	$(MAKE) -C nicslu/util clean

distclean: clean

-include $(OBJS:.o=.d)
-include test_symbolic_gpu.d
-include test_symbolic_gpu_numeric.d
