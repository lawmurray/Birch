NUMBIRCH_VERSION := 0.0.0

build:
	cd jemalloc-*.*.* && make

install:
	cd jemalloc-*.*.* && make install DESTDIR=$$DESTDIR
	BINDIR=`grep 'BINDIR :=' jemalloc-*.*.*/Makefile | sed -E 's/BINDIR *:= *[$$]\(DESTDIR\)\///'`; \
	install dpcpp_wrapper nvcc_wrapper $$DESTDIR/$$BINDIR

dist:
	rm -rf numbirch-prereq-$(NUMBIRCH_VERSION)
	mkdir numbirch-prereq-$(NUMBIRCH_VERSION)
	cp -r dpcpp_wrapper nvcc_wrapper configure Makefile numbirch-prereq.spec debian PKGBUILD numbirch-prereq-$(NUMBIRCH_VERSION)
	tar czf numbirch-prereq-$(NUMBIRCH_VERSION).tar.gz numbirch-prereq-$(NUMBIRCH_VERSION)
	rm -rf numbirch-prereq-$(NUMBIRCH_VERSION)
