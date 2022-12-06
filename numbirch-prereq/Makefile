NUMBIRCH_VERSION=0.0.0
JEMALLOC_VERSION=5.3.0

build:

dist:
	rm -rf numbirch-prereq-$(NUMBIRCH_VERSION)
	mkdir numbirch-prereq-$(NUMBIRCH_VERSION)
	cd numbirch-prereq-$(NUMBIRCH_VERSION) && \
		curl -LO https://github.com/jemalloc/jemalloc/releases/download/$(JEMALLOC_VERSION)/jemalloc-$(JEMALLOC_VERSION).tar.bz2 && \
		tar xjf jemalloc-$(JEMALLOC_VERSION).tar.bz2 --strip-components=1 && \
		rm jemalloc-$(JEMALLOC_VERSION).tar.bz2
	cp -r dpcpp_wrapper nvcc_wrapper numbirch-prereq.spec debian PKGBUILD numbirch-prereq-$(NUMBIRCH_VERSION)
	tar czf numbirch-prereq-$(NUMBIRCH_VERSION).tar.gz numbirch-prereq-$(NUMBIRCH_VERSION)
	rm -rf numbirch-prereq-$(NUMBIRCH_VERSION)
