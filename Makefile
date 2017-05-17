all:
	birch build --verbose --disable-std

install:
	birch install --verbose --disable-std

uninstall:
	birch uninstall --verbose --disable-std

clean:
	rm -rf build autom4te.cache autogen.sh common.am configure.ac Makefile.am configure
