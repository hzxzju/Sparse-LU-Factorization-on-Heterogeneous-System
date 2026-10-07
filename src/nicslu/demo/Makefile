include ../make.inc
GCC_TOOLCHAIN ?= $(HOME)/local/gcc-7.3.0/usr
CC = $(GCC_TOOLCHAIN)/bin/gcc
export GCC_TOOLCHAIN
LIBS = ../lib/nicslu.a ../util/nicslu_util.a -lrt -lpthread -lm
INC = -I../include -I../util
PROGRAMS = demos demop test demos_time demop_time
OBJS = $(PROGRAMS:=.o)

.PHONY: all clean
.DELETE_ON_ERROR:
all: $(PROGRAMS)

$(PROGRAMS): %: %.o ../lib/nicslu.a ../util/nicslu_util.a
	$(CC) $(LFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

%.o: %.c Makefile ../make.inc
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INC) -MMD -MP -o $@ $<

../lib/nicslu.a: $(wildcard ../source/*.c ../source/*.h ../include/*.h) ../lib/Makefile ../make.inc
	$(MAKE) -C ../lib

../util/nicslu_util.a: $(wildcard ../util/*.c ../util/*.h ../include/*.h) ../util/Makefile ../make.inc
	$(MAKE) -C ../util

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(PROGRAMS)

-include $(OBJS:.o=.d)
