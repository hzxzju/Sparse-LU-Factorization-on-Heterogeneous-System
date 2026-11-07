# Forward native build and cleanup commands to the source directory.
.PHONY: all MAIN clean distclean

all MAIN clean distclean:
	$(MAKE) -C src $@
