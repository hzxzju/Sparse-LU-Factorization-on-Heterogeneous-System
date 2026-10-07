include ../nicslu/make.inc
GCC_TOOLCHAIN ?= $(HOME)/local/gcc-7.3.0/usr
CC = $(GCC_TOOLCHAIN)/bin/gcc
COMPILE = $(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP
INC = -I../nicslu/include -I../nicslu/util

.PHONY: all clean
.DELETE_ON_ERROR:
all: preprocess.o

preprocess.o: preprocess.c Makefile ../nicslu/make.inc
	$(COMPILE) $(INC) $< -o $@

clean:
	rm -f preprocess.o preprocess.d

-include preprocess.d
