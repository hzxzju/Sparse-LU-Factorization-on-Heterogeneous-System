GCC_TOOLCHAIN ?= $(HOME)/local/gcc-7.3.0/usr
export GCC_TOOLCHAIN

.PHONY: all lib util demo clean
all: lib util demo

lib util:
	$(MAKE) -C $@

demo: lib util
	$(MAKE) -C demo

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C util clean
	$(MAKE) -C demo clean
