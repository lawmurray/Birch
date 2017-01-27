all:
	birch build --verbose --enable-warnings --enable-assert --enable-extra-debug

install:
	birch install --verbose --enable-warnings --enable-assert --enable-extra-debug

uninstall:
	birch uninstall --verbose --enable-warnings --enable-assert --enable-extra-debug

clean:
	rm -rf build autom4te.cache autogen.sh common.am configure.ac Makefile.am configure
