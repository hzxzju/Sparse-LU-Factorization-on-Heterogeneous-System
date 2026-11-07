# ---------------------------------------------------------------------------
# KLU standalone demo (Linux / gcc)
#
#   make            build libklu.a and the klu_demo executable
#   make clean      remove build products
#
# Outputs:
#   libklu.a        static library (KLU + AMD + COLAMD + BTF + SuiteSparse_config)
#   klu_demo        example driver that solves A x = b, b = ones(n)
# ---------------------------------------------------------------------------

CC      ?= gcc
CFLAGS  ?= -O2 -fPIC -w
INC     := -Iinclude
LIB     := libklu.a

SRCS := $(wildcard src/*.c)
OBJS := $(patsubst src/%.c,build/%.o,$(SRCS))

.PHONY: all clean

all: $(LIB) klu_demo

$(LIB): $(OBJS)
	ar rcs $@ $^

build/%.o: src/%.c | build
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

build:
	mkdir -p build

klu_demo: examples/klu_demo.c $(LIB)
	$(CC) $(CFLAGS) $(INC) $< $(LIB) -lm -o $@

clean:
	rm -rf build $(LIB) klu_demo x.txt
