all:
	birch build --verbose --enable-debug --disable-std

install:
	birch install --verbose --enable-debug --disable-std

uninstall:
	birch uninstall --verbose --enable-debug --disable-std

clean:
	rm -rf 	build aclocal.m4 autom4te.cache autogen.sh common.am compile configure configure.ac config.guess config.sub depcomp install-sh ltmain.sh m4 Makefile.am Makefile.in missing
