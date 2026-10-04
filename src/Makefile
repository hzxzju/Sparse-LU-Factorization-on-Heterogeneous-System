# Makefile
DEBUG = 0


LU_INC = -Ipreprocess -I../include -Inicslu/include

PREPROCESS_LIBS = preprocess/preprocess.o nicslu/lib/nicslu.a nicslu/util/nicslu_util.a

# HTHPCC offline compiler (Runtime API guide section 4.1.1, printed pp. 49-50).
HPCC_PATH ?= /opt/hpcc
HTCC = $(HPCC_PATH)/htgpu_llvm/bin/htcc
CPP = g++
EXE = lu_cmd

CFLAGS = -O3 -Wall -std=c++11 $(LU_INC) -DGLU_DEBUG=${DEBUG}
# HTHPCC language mode replaces the CUDA-specific nvcc flags.
HTCCFLAGS = -O3 -std=c++11 -x hpcc $(LU_INC)

MAIN: $(EXE)

lu_cmd: Timer.o numeric.o symbolic.o lu_cmd.o $(PREPROCESS_LIBS)
	$(HTCC) $^ -o $@

Timer.o: Timer.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

numeric.o: numeric.cu
	$(HTCC) $(HTCCFLAGS) -c $< -o $@

symbolic.o: symbolic.cc
	$(CPP) $(CFLAGS) -c $< -o $@

lu_cmd.o: lu_cmd.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

preprocess/preprocess.o:
	make -C preprocess/

nicslu/lib/nicslu.a:
	make -C nicslu/

nicslu/lib/nicslu_util.a:
	make -C nicslu/

clean:
	rm -f *.o *~ *.so $(EXE)
	make -C preprocess/ clean

distclean:
	rm -f *.o *~ *.so $(EXE)
	make -C nicslu/ clean
	make -C preprocess/ clean
