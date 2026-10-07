include ../make.inc
GCC_TOOLCHAIN ?= $(HOME)/local/gcc-7.3.0/usr
CC = $(GCC_TOOLCHAIN)/bin/gcc
COMPILE = $(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP
AR = ar
ARFLAGS = rcs
INC = -I ../include

.PHONY: all clean
.DELETE_ON_ERROR:
all: nicslu_util.a

OBJS = nicslu_readfile.o

nicslu_util.a: $(OBJS)
	$(AR) $(ARFLAGS) $@ $^
	$(RANLIB) $@

%.o: %.c Makefile ../make.inc
	$(COMPILE) $(INC) $< -o $@

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) nicslu_util.a

-include $(OBJS:.o=.d)
