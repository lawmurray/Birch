lib_LTLIBRARIES =
if DEBUG
lib_LTLIBRARIES += libnumbirch-debug.la
endif
if TEST
lib_LTLIBRARIES += libnumbirch-test.la
endif
if RELEASE
lib_LTLIBRARIES += libnumbirch.la
endif

AM_CPPFLAGS = -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_DONT_PARALLELIZE=1
DP_CXXFLAGS = -qmkl=sequential -fsycl -Wno-unused-command-line-argument
NV_CXXFLAGS = -allow-unsupported-compiler

COMMON_SOURCES = numbirch/core.cpp
if ONEAPI
  COMMON_SOURCES += numbirch/core_oneapi.dpcpp
endif
if CUDA
  COMMON_SOURCES += numbirch/core_cuda.cu
endif

libnumbirch_debug_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O -g -flto
libnumbirch_debug_la_SOURCES = $(COMMON_SOURCES)

libnumbirch_test_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O -g -flto --coverage
libnumbirch_test_la_SOURCES = $(COMMON_SOURCES)

libnumbirch_la_CPPFLAGS = $(AM_CPPFLAGS) -DNDEBUG
libnumbirch_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O3 -flto
libnumbirch_la_SOURCES = $(COMMON_SOURCES)

include_HEADERS = \
  numbirch/numbirch.hpp

nobase_include_HEADERS = \
  numbirch/core.hpp

dist_noinst_DATA =  \
  Doxyfile \
  LICENSE

# compiler flags to transfer to dpcpp and nvcc, these can be obtained by
# searching for CXXCOMPILE and LTCXXCOMPILE in generated Makefile
CXXCOMPILEFLAGS = $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
  $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS)
LTCXXCOMPILEFLAGS = $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
  $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS)
SUFFIXES = .cu .dpcpp

# support for building *.dpcpp files with dpcpp
.dpcpp.o:
	depbase=`echo -n $@ | sed 's|[^/]*$$|.deps/&|;s|\.o$$||'` && \
	srcbase=`echo -n $@ | sed 's|/[^/]*$$||'` && \
	$(DPCPP) -M $(CXXCOMPILEFLAGS) $(DP_CXXFLAGS) -odir $$srcbase -o $$depbase.Po $< && \
	$(DPCPP) -c $(CXXCOMPILEFLAGS) $(DP_CXXFLAGS) -o $@ $<

.dpcpp.lo:
	depbase=`echo -n $@ | sed 's|[^/]*$$|.deps/&|;s|\.lo$$||'` && \
	srcbase=`echo -n $@ | sed 's|/[^/]*$$||'` && \
	$(DPCPP) -M $(CXXCOMPILEFLAGS) $(DP_CXXFLAGS) -odir $$srcbase -o $$depbase.Plo $< && \
	$(LIBTOOL) --tag=CC --mode=compile $(DPCPP) -c $(CXXCOMPILEFLAGS) $(DP_CXXFLAGS) -o $@ $<

# support for building *.cu files with nvcc
.cu.o:
	depbase=`echo -n $@ | sed 's|[^/]*$$|.deps/&|;s|\.o$$||'` && \
	srcbase=`echo -n $@ | sed 's|/[^/]*$$||'` && \
	$(NVCC) -M $(NV_CXXFLAGS) $(CXXCOMPILEFLAGS) -odir $$srcbase -o $$depbase.Po $< && \
	$(NVCC) -c $(NV_CXXFLAGS) $(CXXCOMPILEFLAGS) -o $@ $<

.cu.lo:
	depbase=`echo -n $@ | sed 's|[^/]*$$|.deps/&|;s|\.lo$$||'` && \
	srcbase=`echo -n $@ | sed 's|/[^/]*$$||'` && \
	$(NVCC) -M $(NV_CXXFLAGS) $(CXXCOMPILEFLAGS) -odir $$srcbase -o $$depbase.Plo $< && \
	$(NVCC) -c $(NV_CXXFLAGS) $(CXXCOMPILEFLAGS) -o $@ -Xcompiler="-DPIC -fPIC" $<
  # ^ libtool passes -DPIC and -fPIC to nvcc, which does not recognize them;
  #   instead don't use libtool, but pass these in with -Xcompiler
