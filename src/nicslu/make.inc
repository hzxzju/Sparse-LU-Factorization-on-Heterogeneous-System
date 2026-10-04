CC = gcc
AR = ar -r
RANLIB = ranlib
MACROS = -DNO_ATOMIC #-DNICS_INT64 #-DNO_EXTENSION
ARCH = 
# PIC is required when NICSLU's static archives are linked into pyglu's .so.
CFLAGS = -c -g -O2 -Wall -Wextra -fPIC $(ARCH) $(MACROS)
LFLAGS = 
