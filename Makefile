all:
	birch build --verbose --enable-warnings --disable-std

install:
	birch install --verbose --enable-warnings --disable-std

uninstall:
	birch uninstall --verbose --enable-warnings --disable-std

clean:
	rm -rf build autom4te.cache autogen.sh common.am configure.ac Makefile.am configure
