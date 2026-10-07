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
OBJS = Timer.o numeric.o symbolic.o lu_cmd.o
PREPROCESS_LIBS = preprocess/preprocess.o nicslu/lib/nicslu.a nicslu/util/nicslu_util.a
LU_INC = -Ipreprocess -I../include -Inicslu/include
CPPFLAGS += $(LU_INC) -DGLU_DEBUG=$(DEBUG)
CXXFLAGS ?= -O3 -Wall -std=c++11
HTCCFLAGS ?= -O3 -std=c++11 -x hpcc
HTCC_GCC_FLAGS = -gcc-toolchain=$(GCC_TOOLCHAIN) -gcc-version=$(GCC_VERSION)
# This SDK spells the block synchronization intrinsic __syncthreads.
HTCC_CPPFLAGS = -D__sync_threads=__syncthreads
LDFLAGS += -Wl,-rpath,$(GCC_TOOLCHAIN)/lib64 -Wl,-rpath,$(HPCC_PATH)/lib
LDLIBS += -lpthread -lm -lrt

.PHONY: MAIN all clean distclean
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

lu_cmd.o: lu_cmd.cpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

preprocess/preprocess.o: $(wildcard preprocess/*.c preprocess/*.h nicslu/include/*.h nicslu/util/*.h) preprocess/Makefile nicslu/make.inc
	$(MAKE) -C preprocess

nicslu/lib/nicslu.a: $(wildcard nicslu/source/*.c nicslu/source/*.h nicslu/include/*.h) nicslu/lib/Makefile nicslu/make.inc
	$(MAKE) -C nicslu lib

nicslu/util/nicslu_util.a: $(wildcard nicslu/util/*.c nicslu/util/*.h nicslu/include/*.h) nicslu/util/Makefile nicslu/make.inc
	$(MAKE) -C nicslu util

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(EXE)
	$(MAKE) -C preprocess clean
	$(MAKE) -C nicslu/lib clean
	$(MAKE) -C nicslu/util clean

distclean: clean

-include $(OBJS:.o=.d)
