all:
	birch build --verbose

install:
	birch install --verbose

uninstall:
	birch uninstall --verbose

clean:
	rm -rf build autom4te.cache autogen.sh common.am configure.ac Makefile.am configure
